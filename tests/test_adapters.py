import json
import os

import httpx
import pytest

from contrastset.adapters import Adapter, FakeAdapter, OllamaAdapter, OpenAIAdapter, make_adapter


def test_fake_adapter_follows_seeding_contract() -> None:
    fake = FakeAdapter({"Düren": ["Germany", "Germany", "France"]})
    assert isinstance(fake, Adapter)
    assert fake.complete("Where is Düren?", n=6, temperature=0.7, seed=0) == ["Germany", "Germany", "France"] * 2
    assert fake.complete("Where is Düren?", temperature=0) == ["Germany"]
    assert fake.complete("Where is Atlantis?") == ["I don't know."]
    assert [c["seed"] for c in fake.calls[:3]] == [0, 1, 2]


def _recording_transport(reply: dict, seen: list[dict]) -> httpx.MockTransport:  # type: ignore[type-arg]
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"path": request.url.path, "body": json.loads(request.content)})
        return httpx.Response(200, json=reply)

    return httpx.MockTransport(handler)


def test_ollama_sends_raw_prompt_with_per_sample_seeds() -> None:
    seen: list[dict] = []  # type: ignore[type-arg]
    adapter = OllamaAdapter("qwen3:14b", transport=_recording_transport({"response": "Germany"}, seen))
    assert adapter.complete("<|im_start|>user\nq", n=3, temperature=0.7, seed=10, max_tokens=8) == ["Germany"] * 3
    assert all(s["path"] == "/api/generate" for s in seen)
    body = seen[0]["body"]
    assert body["raw"] is True and body["prompt"] == "<|im_start|>user\nq"
    assert body["options"] == {"temperature": 0.7, "num_predict": 8, "seed": 10}
    assert [s["body"]["options"]["seed"] for s in seen] == [10, 11, 12]


def test_openai_completions_mode_sends_prompt_unchanged() -> None:
    seen: list[dict] = []  # type: ignore[type-arg]
    transport = _recording_transport({"choices": [{"text": "Germany"}]}, seen)
    adapter = OpenAIAdapter("m", base_url="http://x/v1", api_key="", transport=transport)
    assert adapter.complete("raw prompt", n=2, seed=5) == ["Germany", "Germany"]
    assert seen[0]["path"] == "/v1/completions"
    assert seen[0]["body"]["prompt"] == "raw prompt"
    assert [s["body"]["seed"] for s in seen] == [5, 6]


def test_openai_chat_mode_wraps_prompt_as_user_message() -> None:
    seen: list[dict] = []  # type: ignore[type-arg]
    transport = _recording_transport({"choices": [{"message": {"content": "ok"}}]}, seen)
    adapter = OpenAIAdapter("m", base_url="http://x/v1", mode="chat", system="be terse", api_key="", transport=transport)
    assert adapter.complete("hi") == ["ok"]
    assert seen[0]["path"] == "/v1/chat/completions"
    assert seen[0]["body"]["messages"] == [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]


def test_make_adapter() -> None:
    assert isinstance(make_adapter("ollama", "qwen3:14b"), OllamaAdapter)
    assert make_adapter("openai-chat", "m", "http://x/v1").describe()["mode"] == "chat"
    with pytest.raises(Exception, match="base URL"):
        make_adapter("openai", "m")


# Live round trip against a real server. Opt in with e.g.
#   CONTRASTSET_LIVE_OLLAMA_MODEL=qwen3:14b uv run pytest -m live -s
QWEN3_NO_THINK = (
    "<|im_start|>user\nWhich country is Düren in? Answer with just the name.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


@pytest.mark.live
@pytest.mark.skipif("CONTRASTSET_LIVE_OLLAMA_MODEL" not in os.environ, reason="no live model configured")
def test_live_ollama_round_trip() -> None:
    adapter = OllamaAdapter(os.environ["CONTRASTSET_LIVE_OLLAMA_MODEL"],
                            base_url=os.environ.get("CONTRASTSET_LIVE_OLLAMA_URL", OllamaAdapter.default_url))
    greedy = adapter.complete(QWEN3_NO_THINK, temperature=0, max_tokens=16)
    sampled = adapter.complete(QWEN3_NO_THINK, n=3, temperature=0.7, seed=0, max_tokens=16)
    print("\n", adapter.describe(), "\n greedy:", greedy, "\n sampled:", sampled)
    assert "germany" in greedy[0].lower()
    assert "<think>" not in greedy[0]
    assert adapter.complete(QWEN3_NO_THINK, n=1, temperature=0.7, seed=0, max_tokens=16) == sampled[:1]


def test_anthropic_adapter_sends_no_temperature_or_seed() -> None:
    from types import SimpleNamespace

    from contrastset.adapters import AnthropicAdapter

    sent: list[dict] = []  # type: ignore[type-arg]

    class Messages:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            sent.append(kwargs)
            return SimpleNamespace(stop_reason="end_turn", stop_details=None,
                                   content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="ok")])

    adapter = AnthropicAdapter("claude-haiku-5-5", client=SimpleNamespace(messages=Messages()))
    assert adapter.complete("hi", n=2, temperature=0.7, seed=1) == ["ok", "ok"]
    assert "temperature" not in sent[0] and "seed" not in sent[0]
    assert sent[0]["output_config"] == {"effort": "low"}
    assert adapter.describe()["seed_supported"] is False
