"""Ollama's native API in raw mode.

Why not Ollama's OpenAI-compatible /v1 endpoint: it wraps every prompt in the model's
chat template server-side (so a prompt you rendered yourself gets wrapped twice),
ignores `n`, and for Qwen3 runs hidden "thinking" before answering. Grading through
it measures a different prompt than the one you later probe. /api/generate with
raw=true sends exactly the string you give it, so the prompt template file is the
whole truth about what the model saw.
"""

from __future__ import annotations

from typing import Any

import httpx

from contrastset.adapters.base import AdapterError, sample_seed


class OllamaAdapter:
    default_url = "http://localhost:11434"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = default_url,
        timeout: float = 300.0,
        top_k: int | None = None,
        top_p: float | None = None,
        transport: httpx.BaseTransport | None = None,  # tests inject a mock here
    ) -> None:
        self.model = model
        # None: the server's Modelfile default applies (qwen3: top_k 20, top_p 0.95);
        # describe() records those defaults, so provenance shows what was used.
        self.top_k, self.top_p = top_k, top_p
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)
        self._described: dict[str, Any] | None = None

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
            options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
            if self.top_k is not None:
                options["top_k"] = self.top_k
            if self.top_p is not None:
                options["top_p"] = self.top_p
            s = sample_seed(seed, i)
            if s is not None:
                options["seed"] = s
            body = {"model": self.model, "prompt": prompt, "raw": True, "stream": False, "options": options}
            try:
                response = self._http.post("/api/generate", json=body)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise AdapterError(f"ollama {self.base_url} model {self.model!r}: {exc}") from exc
            out.append(response.json()["response"])
        return out

    def describe(self) -> dict[str, Any]:
        if self._described is None:
            info: dict[str, Any] = {"adapter": "ollama", "model": self.model, "endpoint": self.base_url, "raw": True}
            try:
                tags = self._http.get("/api/tags").json().get("models", [])
                match = next((m for m in tags if m.get("name") == self.model), None)
                if match:
                    info["digest"] = match.get("digest")
                    details = match.get("details", {})
                    info["parameter_size"] = details.get("parameter_size")
                    info["quantization"] = details.get("quantization_level")
                params = self._http.post("/api/show", json={"model": self.model}).json().get("parameters", "")
                defaults = {}
                for line in params.splitlines():
                    key, _, value = line.partition(" ")
                    if key in ("top_k", "top_p", "temperature", "repeat_penalty", "min_p"):
                        defaults[key] = value.strip()
                info["server_sampling_defaults"] = defaults
                info["sampling_overrides"] = {"top_k": self.top_k, "top_p": self.top_p}
                info["sampling_order"] = "the server's (Ollama/llama.cpp); see server_version"
                info["server_version"] = self._http.get("/api/version").json().get("version")
            except (httpx.HTTPError, ValueError):
                pass  # provenance is best-effort; complete() will fail loudly if the server is down
            self._described = info
        return self._described
