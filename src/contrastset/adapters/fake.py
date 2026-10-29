"""A deterministic stand-in model for tests. No network, no weights."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from contrastset.adapters.base import sample_seed


class FakeAdapter:
    """Answers from a script.

    `script` maps a substring of the prompt to the answers the fake model gives;
    the first key found in the prompt wins. Greedy calls (temperature 0) return the
    first answer; sampled calls return answers[seed % len(answers)], so a script like
    ["Germany", "Germany", "France"] yields a known share of each answer. Alternatively
    pass a function (prompt, temperature, seed) -> str. Every call is logged in `calls`.
    """

    def __init__(
        self,
        script: dict[str, list[str]] | Callable[[str, float, int | None], str],
        *,
        default: str = "I don't know.",
        name: str = "fake",
    ) -> None:
        self.script = script
        self.default = default
        self.name = name
        self.calls: list[dict[str, Any]] = []

    def _one(self, prompt: str, temperature: float, seed: int | None) -> str:
        if callable(self.script):
            return self.script(prompt, temperature, seed)
        for key, answers in self.script.items():
            if key in prompt:
                if temperature == 0 or seed is None:
                    return answers[0]
                return answers[seed % len(answers)]
        return self.default

    def complete(
        self,
        prompt: str,
        *,
        n: int = 1,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 64,
    ) -> list[str]:
        out = []
        for i in range(n):
            s = sample_seed(seed, i)
            self.calls.append({"prompt": prompt, "temperature": temperature, "seed": s, "max_tokens": max_tokens})
            out.append(self._one(prompt, temperature, s))
        return out

    def describe(self) -> dict[str, Any]:
        return {"adapter": "fake", "model": self.name}
