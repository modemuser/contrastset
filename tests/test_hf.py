"""The HF adapter on a tiny randomly initialised model built in memory: no download."""

import math
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from contrastset.adapters import HFAdapter, Scorer  # noqa: E402

CHARS = list("abcdefghijklmnopqrstuvwxyz ?.:")


@pytest.fixture(scope="module")
def adapter() -> HFAdapter:
    vocab = {"<eos>": 0, **{c: i + 1 for i, c in enumerate(CHARS)}}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<eos>"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Split("", behavior="isolated")
    tok.decoder = tokenizers.decoders.Fuse()
    hf_tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", unk_token="<eos>")
    torch.manual_seed(0)
    config = transformers.GPT2Config(vocab_size=len(vocab), n_layer=1, n_head=2, n_embd=16, n_positions=128,
                                     eos_token_id=0, bos_token_id=0)
    model = transformers.GPT2LMHeadModel(config)
    model.generation_config.top_k = 3  # a model default that must NOT apply silently
    return HFAdapter(model, tokenizer=hf_tok, device="cpu")


def test_sampling_is_seeded_and_greedy_is_deterministic(adapter: HFAdapter) -> None:
    first = adapter.complete("q: abc?", n=3, temperature=1.0, seed=7, max_tokens=8)
    again = adapter.complete("q: abc?", n=3, temperature=1.0, seed=7, max_tokens=8)
    assert first == again and len(first) == 3
    assert adapter.complete("q: abc?", n=1, temperature=1.0, seed=8, max_tokens=8) == first[1:2]  # sample i: seed + i
    assert adapter.complete("q: abc?", max_tokens=8) == adapter.complete("q: abc?", max_tokens=8)


def test_score_matches_the_models_own_probabilities(adapter: HFAdapter) -> None:
    assert isinstance(adapter, Scorer)
    s = adapter.score("q: ab", "c")
    with torch.no_grad():
        ids = adapter.tokenizer("q: abc", return_tensors="pt", add_special_tokens=False).input_ids
        logits = adapter.model(ids).logits[0, -2]
    expected = torch.log_softmax(logits.float(), dim=-1)[ids[0, -1]].item()
    assert s.tokens == ["c"] and math.isclose(s.logprob, expected, rel_tol=1e-5)
    two = adapter.score("q: ab", "cd")
    assert len(two.token_logprobs) == 2 and math.isclose(two.logprob, sum(two.token_logprobs), rel_tol=1e-6)
    assert two.token_logprobs[0] == pytest.approx(s.logprob, rel=1e-5)


def test_describe_records_sampling_and_ignored_defaults(adapter: HFAdapter) -> None:
    d = adapter.describe()
    from contrastset.adapters.base import SAMPLING_ORDER

    assert d["adapter"] == "hf" and d["sampling"]["order"] == SAMPLING_ORDER
    assert d["sampling"]["top_k"] == "off" and d["sampling"]["top_p"] == "off"
    assert d["model_sampling_defaults_not_applied"]["top_k"] == 3


def test_run_returns_logprobs_and_saves_activations(adapter: HFAdapter, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json as _json

    import numpy as np

    out = adapter.run("q: abc?", max_tokens=4, logprobs=True, layers=[0, -1],
                      positions=["prompt_last", "answer:0", -1], save_to=tmp_path / "acts" / "p0")
    assert len(out["token_logprobs"]) == len(out["tokens"]) > 0
    # the reported logprob of the first answer token equals score() of that token
    first = adapter.tokenizer.convert_tokens_to_string([out["tokens"][0]])
    assert out["token_logprobs"][0] == pytest.approx(adapter.score("q: abc?", first).logprob, abs=1e-4)
    acts = np.load(tmp_path / "acts" / "p0.npy")
    side = _json.loads((tmp_path / "acts" / "p0.json").read_text())
    assert acts.shape == (2, 3, 16) and side["shape"] == [2, 3, 16] and side["layers"] == [0, 1]  # one block: embeddings + 1
    assert [p["index"] for p in side["positions"]][:2] == [side["n_prompt_tokens"] - 1, side["n_prompt_tokens"]]
    assert side["positions"][0]["token"] == "?"
    with pytest.raises(Exception, match="outside"):
        adapter.run("q: abc?", max_tokens=2, layers=[0], positions=[999])


def test_top_p_cuts_the_untempered_distribution(adapter: HFAdapter) -> None:
    """probabilities 0.50 / 0.30 / 0.15 / 0.05, top_p 0.9, temperature 0.5: truncating
    first keeps 3 tokens; tempering first (transformers' generate order) would keep 2."""
    logits = torch.log(torch.tensor([[0.50, 0.30, 0.15, 0.05]]))
    adapter.top_k, adapter.top_p = None, 0.9
    try:
        out = adapter._warpers(0.5)(torch.tensor([[0]]), logits.clone())
    finally:
        adapter.top_k, adapter.top_p = None, None
    assert int(torch.isfinite(out).sum()) == 3
    sharpened = torch.softmax(logits / 0.5, dim=-1)[0].tolist()
    assert sharpened[0] + sharpened[1] >= 0.9  # the other order would stop at 2


def test_score_many_equals_score(adapter: HFAdapter) -> None:
    conts = ["c", "cd", "xyz?", "a b"]
    batched = adapter.score_many("q: ab", conts)
    for c, b in zip(conts, batched):
        single = adapter.score("q: ab", c)
        assert b.tokens == single.tokens and b.logprob == pytest.approx(single.logprob, abs=1e-4)


def test_sampler_logprob_is_the_sampling_distribution(adapter: HFAdapter) -> None:
    """sampler_logprob: top-k/top-p on the untempered distribution, then temperature."""
    adapter.top_k, adapter.top_p = 5, 0.9
    try:
        [sc] = adapter.score_many("q: ab", ["c"], sampler_temperature=0.7)
        with torch.no_grad():
            ids = adapter.tokenizer("q: abc", return_tensors="pt", add_special_tokens=False).input_ids
            logits = adapter.model(ids).logits[0, -2].float()
        processed = adapter._warpers(0.7)(ids[:, :-1], logits.unsqueeze(0).clone())[0]
        expected = torch.log_softmax(processed, dim=-1)[ids[0, -1]].item()
    finally:
        adapter.top_k, adapter.top_p = None, None
    assert sc.sampler_logprob == pytest.approx(expected, abs=1e-4)
    assert sc.logprob == pytest.approx(adapter.score("q: ab", "c").logprob, abs=1e-4)


def test_prefix_mass_sums_over_every_segmentation() -> None:
    """With tokens a, b, ab, bc, abc, the text "ab" can start a completion as [ab],
    [abc], [a, b] or [a, bc]; the mass must be the sum of all four."""
    import math as _math

    vocab = {"<eos>": 0, "q": 1, ":": 2, "x": 3, "a": 4, "b": 5, "c": 6, "ab": 7, "bc": 8, "abc": 9}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<eos>"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Split("", behavior="isolated")
    tok.decoder = tokenizers.decoders.Fuse()
    hf_tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", unk_token="<eos>")
    torch.manual_seed(1)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=len(vocab), n_layer=1, n_head=2,
                                                                 n_embd=16, n_positions=64))
    a = HFAdapter(model, tokenizer=hf_tok, device="cpu")
    prompt = "q:x"
    a.PREFIX_PRUNE = 0.0  # nothing pruned: the sum must be exact
    m = a.prefix_mass(prompt, "ab")
    raw = m["raw"]
    assert raw is not None
    assert m["raw_pruned_max"] == 0.0

    def p_next(ids: list[int]) -> Any:
        with torch.no_grad():
            return torch.softmax(a.model(torch.tensor([ids])).logits[0, -1].float(), dim=-1)

    base = a.tokenize(prompt)
    p1 = p_next(base)
    p2 = p_next(base + [vocab["a"]])
    expected = float(p1[vocab["ab"]] + p1[vocab["abc"]] + p1[vocab["a"]] * (p2[vocab["b"]] + p2[vocab["bc"]]))
    assert raw == pytest.approx(expected, rel=1e-4)
    single = _math.exp(a.score(prompt, "ab").logprob)  # the canonical segmentation [a, b] alone
    assert raw > single

    a.PREFIX_PRUNE = 0.5  # prune every partial path: the bound must account for exactly what was dropped
    m2 = a.prefix_mass(prompt, "ab")
    dropped = float(p1[vocab["a"]])  # the [a, ...] path, the only partial path
    raw2, pruned2 = m2["raw"], m2["raw_pruned_max"]
    assert raw2 is not None and pruned2 is not None
    assert raw2 == pytest.approx(float(p1[vocab["ab"]] + p1[vocab["abc"]]), rel=1e-4)
    assert pruned2 == pytest.approx(dropped, rel=1e-4)
    assert raw2 <= raw <= raw2 + pruned2 + 1e-9
