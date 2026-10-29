"""HF (Hugging Face) transformers models, run in-process (`pip install contrastset[hf]`).

The adapter for experiments that need more than text: it can score a continuation
(log-probability of given tokens) as well as sample, and it runs the same weights
that an activation probe (TransformerLens, nnsight, ...) will load. Grade with the
weights you probe: a Q4 Ollama model and the bf16 HF model are different models.

The prompt is tokenised as-is, without the tokenizer's chat template, like the raw
Ollama adapter: the prompt template file renders the chat format.

Sampling follows the contrastset sampling order (adapters.base.SAMPLING_ORDER):
top-k, then top-p, both on the UNTEMPERED distribution, then temperature, then draw.
This is llama.cpp's order, so HF and Ollama grades sample the same way; transformers'
own generate() applies temperature first, which makes top-p cut a sharpened
distribution (fewer, more consistent answers: FINDINGS.md, parity). The model's
generation_config defaults (Qwen3 ships top_k = 20, top_p = 0.95) are NOT applied
silently; describe() records what is used and what was ignored.

Terms: logit, the model's raw score for each possible next token; log-probability
(logprob), the log of the softmax of the logits, so log P(token | everything before).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contrastset.adapters.base import SAMPLING_ORDER, AdapterError, sample_seed


@dataclass(frozen=True)
class Score:
    """log P(continuation | prompt), and per token."""

    logprob: float  # sum over the continuation's tokens, untempered (T = 1)
    tokens: list[str]
    token_logprobs: list[float]
    # The same under the sampling distribution (contrastset order: truncate the
    # untempered distribution with top-k/top-p, then temper), if asked for; -inf
    # where a token would have been truncated away.
    sampler_logprob: float | None = None


class HFAdapter:
    def __init__(
        self,
        model: str | Any,
        *,
        tokenizer: Any = None,
        device: str | None = None,
        dtype: str = "bfloat16",
        top_k: int | None = None,
        top_p: float | None = None,
        revision: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise AdapterError("the hf adapter needs `pip install contrastset[hf]`") from exc
        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model: Any  # transformers' own type hints don't fit generic causal LMs well
        self.tokenizer: Any
        if isinstance(model, str):
            self.name = model
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype), revision=revision)
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model, revision=revision)
        else:  # an already-built model, e.g. a tiny one in tests
            if tokenizer is None:
                raise AdapterError("pass a tokenizer with a model object")
            self.name = getattr(model.config, "name_or_path", "") or type(model).__name__
            self.model, self.tokenizer = model, tokenizer
        self.model.to(device).eval()
        self.top_k, self.top_p = top_k, top_p
        self._tbytes: list[bytes] | None = None
        self._token_trie: dict[str, Any] | None = None
        self.revision = revision or getattr(self.model.config, "_commit_hash", None) or self._cached_revision()

    def _warpers(self, temperature: float) -> Any:
        from transformers.generation.logits_process import (
            LogitsProcessorList,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        steps: list[Any] = []
        if self.top_k:
            steps.append(TopKLogitsWarper(top_k=self.top_k))
        if self.top_p is not None and self.top_p < 1.0:
            steps.append(TopPLogitsWarper(top_p=self.top_p))
        steps.append(TemperatureLogitsWarper(temperature))
        return LogitsProcessorList(steps)

    def _token_bytes(self) -> list[bytes] | None:
        """The bytes each token id stands for, for byte-level BPE tokenizers (GPT-2
        style, as in Qwen and Llama 3); None for others."""
        if self._tbytes is None:
            printable = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
                         + list(range(ord("®"), ord("ÿ") + 1)))
            bs, cs, n = printable[:], printable[:], 0
            for b in range(256):
                if b not in printable:
                    bs.append(b)
                    cs.append(256 + n)
                    n += 1
            decode = {chr(c): b for b, c in zip(bs, cs)}
            out, misses = [], 0
            for tok in self.tokenizer.convert_ids_to_tokens(list(range(len(self.tokenizer)))):
                try:
                    out.append(bytes(decode[ch] for ch in tok))
                except (KeyError, TypeError):
                    out.append(b"")
                    misses += 1
            self._tbytes = out if misses < 0.1 * len(out) else []
        return self._tbytes or None

    # Prefix-mass expansion limits, recorded in provenance (describe()["prefix_mass"]).
    PREFIX_PRUNE = 1e-5  # drop a partial path once its probability is below this
    PREFIX_MAX_PATHS = 512  # safety cap per step; overflow is counted in the bound, never silently lost

    def _trie(self) -> dict[str, Any]:
        """Byte trie over the vocabulary: node = {"ids": tokens ending here, "kids": {byte: node}}."""
        if self._token_trie is None:
            root: dict[str, Any] = {"ids": [], "kids": {}}
            for tok_id, b in enumerate(self._token_bytes() or []):
                if not b:
                    continue
                node = root
                for byte in b:
                    node = node["kids"].setdefault(byte, {"ids": [], "kids": {}})
                node["ids"].append(tok_id)
            self._token_trie = root
        return self._token_trie

    @staticmethod
    def _subtree_ids(node: dict[str, Any]) -> list[int]:
        if "all" not in node:
            ids, stack = [], [node]
            while stack:
                n = stack.pop()
                ids.extend(n["ids"])
                stack.extend(n["kids"].values())
            node["all"] = ids
        return node["all"]  # type: ignore[no-any-return]

    def prefix_mass(self, prompt: str, text: str, sampler_temperature: float | None = None
                    ) -> dict[str, float | None]:
        """P(the completion starts with `text`), summed over every token segmentation
        that spells it. Models often write a name in a segmentation the tokenizer
        would not choose (Qwen3-8B: "Rh·in·eland" 0.80 vs the canonical "R·hin·eland"
        0.03), so scoring one segmentation can miss most of the mass.

        Exact apart from mass pruning: at each step EVERY token consistent with the
        rest of `text` is considered (byte trie, no top-k), tokens that cover the rest
        finish a path (a token may overshoot: "Mur" includes "Mura..."), and a partial
        path is dropped only when its probability falls below PREFIX_PRUNE. Each
        dropped path's probability is added to `pruned_max`, an upper bound on the
        mass pruning could have lost: the true value lies in [mass, mass + pruned_max].

        Returns {"raw", "raw_pruned_max", "sampler", "sampler_pruned_max"}: the
        untempered mass and, if asked, the mass under the sampling distribution.
        Falls back to the canonical segmentation for non-byte-level tokenizers."""
        trie = self._trie() if self._token_bytes() is not None else None
        if trie is None:
            [sc] = self.score_many(prompt, [text], sampler_temperature=sampler_temperature)
            samp = math.exp(sc.sampler_logprob) if sc.sampler_logprob is not None else None
            return {"raw": math.exp(sc.logprob), "raw_pruned_max": None, "sampler": samp, "sampler_pruned_max": None}
        torch = self._torch
        target = text.encode("utf-8")
        prompt_ids = self._ids(prompt)[0]
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        use_sampler = sampler_temperature is not None
        frontier: list[tuple[list[int], float, float, int]] = [([], 0.0, 0.0, 0)]
        done = {"raw": 0.0, "sampler": 0.0}
        lost = {"raw": 0.0, "sampler": 0.0}
        while frontier:
            rows = [torch.cat([prompt_ids, torch.tensor(p, dtype=prompt_ids.dtype, device=self.device)])
                    for p, _, _, _ in frontier]
            width = max(len(r) for r in rows)
            ids = torch.full((len(rows), width), pad, dtype=prompt_ids.dtype, device=self.device)
            mask = torch.zeros((len(rows), width), dtype=torch.long, device=self.device)
            for i, r in enumerate(rows):
                ids[i, :len(r)] = r
                mask[i, :len(r)] = 1
            with torch.no_grad():
                logits = self.model(ids, attention_mask=mask).logits
            nxt: list[tuple[list[int], float, float, int]] = []
            for i, (path, lr, ls, used) in enumerate(frontier):
                row = logits[i, len(rows[i]) - 1].float()
                raw = torch.log_softmax(row, dim=-1)
                samp = self._sampler_logprobs(row, sampler_temperature) if sampler_temperature is not None else None
                rest = target[used:]
                node, continuing = trie, []
                for depth, byte in enumerate(rest):
                    node = node["kids"].get(byte)
                    if node is None:
                        break
                    if depth + 1 < len(rest):
                        continuing.extend(node["ids"])  # tokens spelling a proper prefix of the rest
                else:  # walked the whole rest: every token in this subtree covers it
                    finish = torch.tensor(self._subtree_ids(node), dtype=torch.long, device=raw.device)
                    if len(finish):
                        done["raw"] += math.exp(lr) * float(torch.exp(raw[finish]).sum())
                        if samp is not None:
                            done["sampler"] += math.exp(ls) * float(torch.exp(samp[finish]).sum())
                for tok in continuing:
                    r_lp = lr + float(raw[tok])
                    s_lp = ls + float(samp[tok]) if samp is not None else float("-inf")
                    step = len((self._token_bytes() or [])[tok])
                    if math.exp(r_lp) >= self.PREFIX_PRUNE:
                        nxt.append((path + [tok], r_lp, s_lp, used + step))
                    else:
                        lost["raw"] += math.exp(r_lp)
                        lost["sampler"] += math.exp(s_lp)
            del logits
            nxt.sort(key=lambda x: -x[1])
            for _, r_lp, s_lp, _ in nxt[self.PREFIX_MAX_PATHS:]:
                lost["raw"] += math.exp(r_lp)
                lost["sampler"] += math.exp(s_lp)
            frontier = nxt[:self.PREFIX_MAX_PATHS]
        if self.device == "mps":
            self._torch.mps.empty_cache()
        return {"raw": min(1.0, done["raw"]), "raw_pruned_max": lost["raw"],
                "sampler": min(1.0, done["sampler"]) if use_sampler else None,
                "sampler_pruned_max": lost["sampler"] if use_sampler else None}

    def tokenize(self, text: str) -> list[int]:
        return list(self._ids(text)[0].tolist())

    def _ids(self, text: str) -> Any:
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

    def complete(
        self,
        prompt: str,
        *,
        n: int = 1,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 64,
    ) -> list[str]:
        torch = self._torch
        ids = self._ids(prompt)
        out = []
        for i in range(n):
            kwargs: dict[str, Any] = {"max_new_tokens": max_tokens, "pad_token_id": self.tokenizer.pad_token_id
                                      or self.tokenizer.eos_token_id}
            if temperature == 0:
                kwargs |= {"do_sample": False, "top_k": None, "top_p": None, "temperature": None}
            else:
                s = sample_seed(seed, i)
                if s is not None:
                    torch.manual_seed(s)
                # generate()'s own warpers are switched off (temperature 1, top_k 0, top_p 1)
                # and ours run in the contrastset order: top-k, top-p, then temperature.
                kwargs |= {"do_sample": True, "temperature": 1.0, "top_k": 0, "top_p": 1.0,
                           "logits_processor": self._warpers(temperature)}
            with torch.no_grad():
                generated = self.model.generate(ids, **kwargs)
            out.append(str(self.tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True)))
        return out

    def score(self, prompt: str, continuation: str) -> Score:
        """log P(continuation | prompt). Prompt and continuation are tokenised
        separately and concatenated, so the boundary is exact."""
        torch = self._torch
        prompt_ids, cont_ids = self._ids(prompt), self._ids(continuation)
        if cont_ids.shape[1] == 0:
            raise AdapterError("cannot score an empty continuation")
        ids = torch.cat([prompt_ids, cont_ids], dim=1)
        with torch.no_grad():
            logits = self.model(ids).logits.float()
        logprobs = torch.log_softmax(logits[0, prompt_ids.shape[1] - 1:-1], dim=-1)
        picked = logprobs.gather(1, cont_ids[0].unsqueeze(1)).squeeze(1)
        tokens = self.tokenizer.convert_ids_to_tokens(cont_ids[0].tolist())
        return Score(logprob=float(picked.sum()), tokens=list(tokens), token_logprobs=picked.tolist())

    def _sampler_logprobs(self, logits: Any, temperature: float) -> Any:
        """log q for every token, q being the sampling distribution at each position."""
        torch = self._torch
        keep = torch.ones_like(logits, dtype=torch.bool)
        if self.top_k:
            kth = torch.topk(logits, min(self.top_k, logits.shape[-1]), dim=-1).values[..., -1:]
            keep &= logits >= kth
        if self.top_p is not None and self.top_p < 1.0:
            probs = torch.softmax(logits, dim=-1)
            sorted_p, order = torch.sort(probs, dim=-1, descending=True)
            before = torch.cumsum(sorted_p, dim=-1) - sorted_p  # mass of strictly higher-ranked tokens
            sorted_keep = before < self.top_p  # keep tokens until the cut is reached, including the one crossing it
            keep &= torch.zeros_like(keep).scatter(-1, order, sorted_keep)
        tempered = (logits / temperature).masked_fill(~keep, float("-inf"))
        return torch.log_softmax(tempered, dim=-1)

    def score_many(self, prompt: str, continuations: Sequence[str], chunk: int = 8,
                   sampler_temperature: float | None = None) -> list[Score]:
        """score() for several continuations of one prompt: batched forward passes of
        at most `chunk` rows (right-padded, attention-masked) that keep logits only
        for the continuation positions, so memory stays near the weights' own size.
        Same values as calling score() each."""
        torch = self._torch
        prompt_ids = self._ids(prompt)[0]
        conts = [self._ids(c)[0] for c in continuations]
        if any(len(c) == 0 for c in conts):
            raise AdapterError("cannot score an empty continuation")
        n_prompt = len(prompt_ids)
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        out: list[Score] = []
        for start in range(0, len(conts), chunk):
            part = conts[start:start + chunk]
            rows = [torch.cat([prompt_ids, c]) for c in part]
            width = max(len(r) for r in rows)
            ids = torch.full((len(rows), width), pad, dtype=prompt_ids.dtype, device=self.device)
            mask = torch.zeros((len(rows), width), dtype=torch.long, device=self.device)
            for i, r in enumerate(rows):
                ids[i, :len(r)] = r
                mask[i, :len(r)] = 1
            keep = width - n_prompt + 1  # positions n_prompt-1 .. end predict the continuation tokens
            with torch.no_grad():
                try:
                    logits = self.model(ids, attention_mask=mask, logits_to_keep=keep).logits
                except TypeError:  # models without logits_to_keep
                    logits = self.model(ids, attention_mask=mask).logits[:, -keep:]
            for i, c in enumerate(part):
                row = logits[i, :len(c)].float()
                picked = torch.log_softmax(row, dim=-1).gather(1, c.unsqueeze(1)).squeeze(1)
                sampler = None
                if sampler_temperature is not None:
                    q = self._sampler_logprobs(row, sampler_temperature).gather(1, c.unsqueeze(1)).squeeze(1)
                    sampler = float(q.sum())
                out.append(Score(logprob=float(picked.sum()),
                                 tokens=list(self.tokenizer.convert_ids_to_tokens(c.tolist())),
                                 token_logprobs=picked.tolist(), sampler_logprob=sampler))
            del logits
            if self.device == "mps":
                torch.mps.empty_cache()
        return out

    def run(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 64,
        logprobs: bool = False,
        layers: Sequence[int] = (),
        positions: Sequence[int | str] = (),
        save_to: str | Path | None = None,
    ) -> dict[str, Any]:
        """One completion, plus optional log-probabilities and activations.

        logprobs: log P(token | everything before) for each answer token, from the
          model's untempered distribution (temperature changes what is sampled, not
          what is reported).
        layers, positions: residual-stream activations, i.e. hidden_states, the
          running state each transformer block reads from and adds to. Layer 0 is
          the embedding output, layer i the output of block i; negative counts from
          the end. Positions index the full sequence (prompt + answer); negative
          counts from the end; "prompt_last" is the last prompt token (where the
          model decides how to answer), "answer:k" the k-th answer token.
        save_to: path stem; writes <stem>.npy, shape [layers, positions, d_model],
          float32, and <stem>.json with the resolved positions, their tokens, the
          model and the prompt hash.
        """
        torch = self._torch
        [text] = self.complete(prompt, n=1, temperature=temperature, seed=seed, max_tokens=max_tokens)
        prompt_ids = self._ids(prompt)
        answer_ids = self._ids(text) if text else prompt_ids[:, :0]
        ids = torch.cat([prompt_ids, answer_ids], dim=1)
        n_prompt, n_total = prompt_ids.shape[1], ids.shape[1]
        result: dict[str, Any] = {"text": text, "n_prompt_tokens": n_prompt,
                                  "tokens": self.tokenizer.convert_ids_to_tokens(answer_ids[0].tolist())}
        want_states = bool(layers) and bool(positions)
        with torch.no_grad():
            out = self.model(ids, output_hidden_states=want_states)
        if logprobs and answer_ids.shape[1]:
            lp = torch.log_softmax(out.logits[0, n_prompt - 1:-1].float(), dim=-1)
            result["token_logprobs"] = lp.gather(1, answer_ids[0].unsqueeze(1)).squeeze(1).tolist()
        if want_states:
            resolved = [self._position(p, n_prompt, n_total) for p in positions]
            n_layers = len(out.hidden_states)
            layer_idx = [layer if layer >= 0 else n_layers + layer for layer in layers]
            if any(not 0 <= layer < n_layers for layer in layer_idx):
                raise AdapterError(f"layers {list(layers)} out of range 0..{n_layers - 1}")
            acts = torch.stack([out.hidden_states[layer][0, resolved] for layer in layer_idx]).float().cpu().numpy()
            result["activations"] = acts
            sidecar = {
                "model": self.describe(), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "text": text, "n_prompt_tokens": n_prompt, "n_tokens": n_total,
                "layers": layer_idx, "positions": [
                    {"requested": p, "index": i, "token": self.tokenizer.convert_ids_to_tokens([int(ids[0, i])])[0]}
                    for p, i in zip(positions, resolved)],
                "shape": list(acts.shape), "dtype": "float32",
            }
            result["sidecar"] = sidecar
            if save_to is not None:
                import numpy as np

                stem = Path(save_to)
                stem.parent.mkdir(parents=True, exist_ok=True)
                np.save(stem.with_suffix(".npy"), acts)
                stem.with_suffix(".json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=1))
                result["files"] = [str(stem.with_suffix(".npy")), str(stem.with_suffix(".json"))]
        return result

    @staticmethod
    def _position(p: int | str, n_prompt: int, n_total: int) -> int:
        if p == "prompt_last":
            return n_prompt - 1
        if isinstance(p, str) and p.startswith("answer:"):
            k = int(p.split(":", 1)[1])
            index = n_prompt + k
        elif isinstance(p, int):
            index = p if p >= 0 else n_total + p
        else:
            raise AdapterError(f"unknown position {p!r}: use an int, 'prompt_last' or 'answer:k'")
        if not 0 <= index < n_total:
            raise AdapterError(f"position {p!r} is outside the {n_total}-token sequence")
        return index

    def _cached_revision(self) -> str | None:
        """The commit hash of the locally cached snapshot (.../snapshots/<sha>/)."""
        try:
            from huggingface_hub import try_to_load_from_cache

            path = try_to_load_from_cache(self.name, "config.json")
        except Exception:
            return None
        return Path(path).parent.name if isinstance(path, str) else None

    def describe(self) -> dict[str, Any]:
        gc = getattr(self.model, "generation_config", None)
        return {
            "adapter": "hf", "model": self.name, "revision": self.revision, "device": self.device,
            "dtype": str(next(self.model.parameters()).dtype).removeprefix("torch."),
            "sampling": {"order": SAMPLING_ORDER, "top_k": self.top_k or "off",
                         "top_p": self.top_p if self.top_p is not None and self.top_p < 1.0 else "off",
                         "min_p": "off", "repetition_penalty": "off", "temperature": "per call (provenance.sampling)"},
            "model_sampling_defaults_not_applied": {k: getattr(gc, k, None)
                                                    for k in ("top_k", "top_p", "temperature", "min_p",
                                                              "repetition_penalty")},
            "prefix_mass": {"candidates": "every token consistent with the text (byte trie)",
                            "prune_below": self.PREFIX_PRUNE, "max_paths_per_step": self.PREFIX_MAX_PATHS,
                            "bound": "logprob.p_correct_pruned_max: mass of dropped paths"},
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch"),
        }
