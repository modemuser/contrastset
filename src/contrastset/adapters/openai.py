"""Any OpenAI-compatible HTTP endpoint: vLLM, llama.cpp server, most online APIs.

Two modes:
    completions  POST /completions with the prompt as-is. vLLM and llama.cpp apply no
                 chat template here, so this is the right mode for a subject model
                 whose prompt template file renders the full chat format.
    chat         POST /chat/completions with the prompt as one user message; the
                 server applies its own chat template. Fine for a generator model,
                 where the exact prompt format doesn't matter to the experiment.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import httpx

from contrastset.adapters.base import AdapterError, sample_seed


class OpenAIAdapter:
    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        mode: Literal["completions", "chat"] = "completions",
        api_key: str | None = None,
        system: str | None = None,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,  # tests inject a mock here
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.mode = mode
        self.system = system
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers, transport=transport)

    def _body(self, prompt: str, temperature: float, seed: int | None, max_tokens: int) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "temperature": temperature, "max_tokens": max_tokens}
        if seed is not None:
            body["seed"] = seed
        if self.mode == "chat":
            messages = [{"role": "system", "content": self.system}] if self.system else []
            body["messages"] = [*messages, {"role": "user", "content": prompt}]
        else:
            body["prompt"] = prompt
        return body

    def complete(
        self,
        prompt: str,
        *,
        n: int = 1,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 64,
    ) -> list[str]:
        path = "/chat/completions" if self.mode == "chat" else "/completions"
        out = []
        for i in range(n):
            body = self._body(prompt, temperature, sample_seed(seed, i), max_tokens)
            try:
                response = self._http.post(path, json=body)
                response.raise_for_status()
                choice = response.json()["choices"][0]
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                raise AdapterError(f"{self.base_url}{path} model {self.model!r}: {exc}") from exc
            out.append(choice["message"]["content"] if self.mode == "chat" else choice["text"])
        return out

    def describe(self) -> dict[str, Any]:
        return {"adapter": "openai", "mode": self.mode, "model": self.model, "endpoint": self.base_url}
