"""The adapter interface."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable


class AdapterError(RuntimeError):
    pass


# The contrastset sampling order. Where an adapter samples itself (hf), it follows
# this exactly; where a server samples (ollama), the adapter records the server's
# settings and version, since the server's order is the server's. Matches
# llama.cpp, so HF and Ollama grades are comparable. Truncating before the
# temperature matters for top-p: after a temperature below 1 the distribution is
# sharper and the same top-p keeps fewer tokens.
SAMPLING_ORDER = "top_k -> top_p (both on untempered probabilities) -> temperature -> draw"


@runtime_checkable
class Adapter(Protocol):
    """Turns a prompt into n completions.

    Seeding contract: completion i is sampled with seed `seed + i`, one request per
    completion. That makes every sample individually reproducible and works with
    servers that ignore the OpenAI `n` parameter (Ollama does). With temperature 0
    (greedy decoding: always take the most likely next token) the seed is irrelevant.
    """

    def complete(
        self,
        prompt: str,
        *,
        n: int = 1,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 64,
    ) -> list[str]: ...

    def describe(self) -> dict[str, Any]:
        """Identify the model for provenance: adapter kind, model name, endpoint, and
        whatever the server reports about the exact weights (digest, quantisation)."""
        ...


@runtime_checkable
class Scorer(Protocol):
    """An adapter that can also score text: log P(continuation | prompt). Needs model
    internals (logits), so only in-process adapters (hf) implement it."""

    def score(self, prompt: str, continuation: str) -> Any: ...


def sample_seed(seed: int | None, i: int) -> int | None:
    return None if seed is None else seed + i


def make_adapter(kind: str, model: str, base_url: str | None = None, **options: Any) -> Adapter:
    """Build an adapter by name: "ollama", "openai" (raw completions), "openai-chat" or
    "anthropic" or "hf"."""
    from contrastset.adapters.anthropic import AnthropicAdapter
    from contrastset.adapters.hf import HFAdapter
    from contrastset.adapters.ollama import OllamaAdapter
    from contrastset.adapters.openai import OpenAIAdapter

    if kind == "ollama":
        return OllamaAdapter(model, base_url=base_url or OllamaAdapter.default_url, **options)
    if kind in ("openai", "openai-chat"):
        if base_url is None:
            raise AdapterError(f"adapter {kind!r} needs a base URL, e.g. http://localhost:8000/v1")
        mode: Literal["completions", "chat"] = "chat" if kind == "openai-chat" else "completions"
        return OpenAIAdapter(model, base_url=base_url, mode=mode, **options)
    if kind == "anthropic":
        return AnthropicAdapter(model, **options)
    if kind == "hf":
        return HFAdapter(model, **options)
    raise AdapterError(f"unknown adapter {kind!r}; expected ollama, openai, openai-chat, anthropic or hf")
