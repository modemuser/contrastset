"""Claude models through Anthropic's official SDK (`pip install contrastset[anthropic]`).

Meant for the generator model, not the subject: the API exposes no raw completions
and no internals. Two things differ from the other adapters and are recorded in
describe(), so provenance says it:
    - no temperature: the Messages API rejects temperature/top_p/top_k since
      2026-10-21, so `temperature` is ignored;
    - no seed: the API has none, so generator output is not reproducible from the
      seed. The pairs file keeps every generated text; that file is the record.
"""

from __future__ import annotations

from typing import Any

from contrastset.adapters.base import AdapterError


class AnthropicAdapter:
    def __init__(self, model: str, *, effort: str = "low", system: str | None = None,
                 client: Any = None) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise AdapterError("the anthropic adapter needs `pip install contrastset[anthropic]`") from exc
            client = anthropic.Anthropic()  # credentials from ANTHROPIC_API_KEY or an `ant auth login` profile
        self.model = model
        self.effort = effort  # how much the model thinks before answering: low | medium | high | xhigh | max
        self.system = system
        self._client = client

    def complete(
        self,
        prompt: str,
        *,
        n: int = 1,
        temperature: float = 0.0,  # ignored, see module docstring
        seed: int | None = None,  # ignored, see module docstring
        max_tokens: int = 4096,
    ) -> list[str]:
        out = []
        for _ in range(n):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "output_config": {"effort": self.effort},
            }
            if self.system:
                kwargs["system"] = self.system
            try:
                response = self._client.messages.create(**kwargs)
            except Exception as exc:  # SDK errors carry status and message; surface them as-is
                raise AdapterError(f"anthropic {self.model!r}: {exc}") from exc
            if response.stop_reason == "refusal":
                raise AdapterError(f"anthropic {self.model!r} declined the request: {response.stop_details}")
            out.append("".join(block.text for block in response.content if block.type == "text"))
        return out

    def describe(self) -> dict[str, Any]:
        return {"adapter": "anthropic", "model": self.model, "effort": self.effort,
                "temperature_supported": False, "seed_supported": False}
