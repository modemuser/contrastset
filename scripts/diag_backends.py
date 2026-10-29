"""Do two backends see the same prompt and the same next-token distribution?

For a few questions, records how many tokens the prompt became and the top-20
probabilities of the first answer token, from the HF model or from Ollama (raw
mode, logprobs). Same token count and same distribution: the backends differ only
in how they sample. Different: they don't see the same thing (template,
tokenisation, numerics). Not part of the library.

    uv run python scripts/diag_backends.py hf Qwen/Qwen3-8B TEMPLATE -o hf.json
    uv run python scripts/diag_backends.py ollama qwen3:8b-fp16 TEMPLATE -o ollama.json
    uv run python scripts/diag_backends.py compare hf.json ollama.json
"""

import json
import math
import sys

import httpx

QUESTIONS = ["Which river flows through Ronda?", "Which river flows through Písek?",
             "Which river flows through Wels?", "Which river flows through Steyr?",
             "Which river flows through Bad Münstereifel?", "Which river flows through Heidelberg?"]


def hf(model: str, template_path: str) -> dict:
    import torch

    from contrastset.adapters import HFAdapter
    from contrastset.grade import Template

    a, t = HFAdapter(model), Template.load(template_path)
    out = {}
    for q in QUESTIONS:
        ids = a._ids(t.render(q, ""))
        with torch.no_grad():
            probs = torch.softmax(a.model(ids).logits[0, -1].float(), dim=-1)
        top = torch.topk(probs, 20)
        out[q] = {"prompt_tokens": int(ids.shape[1]),
                  "top": {a.tokenizer.decode([int(i)]): float(p) for p, i in zip(top.values, top.indices)}}
    return out


def ollama(model: str, template_path: str) -> dict:
    from contrastset.grade import Template

    t = Template.load(template_path)
    out = {}
    for q in QUESTIONS:
        r = httpx.post("http://localhost:11434/api/generate", timeout=300, json={
            "model": model, "prompt": t.render(q, ""), "raw": True, "stream": False,
            "logprobs": True, "top_logprobs": 20, "options": {"temperature": 0, "num_predict": 1}}).json()
        first = (r.get("logprobs") or [{}])[0]
        top = {c["token"]: math.exp(c["logprob"]) for c in first.get("top_logprobs", [])}
        out[q] = {"prompt_tokens": r.get("prompt_eval_count"), "top": top, "has_logprobs": bool(top)}
    return out


def compare(a_path: str, b_path: str) -> None:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    for q in QUESTIONS:
        ta, tb = a[q]["top"], b[q]["top"]
        shared = set(ta) & set(tb)
        diff = max((abs(ta[k] - tb[k]) for k in shared), default=float("nan"))
        entropy = lambda d: -sum(p * math.log(p) for p in d.values() if p > 0)
        print(f"{q}\n  prompt tokens {a[q]['prompt_tokens']} vs {b[q]['prompt_tokens']};  top-20 overlap "
              f"{len(shared)}/20;  max |p diff| on shared {diff:.4f};  top-20 entropy {entropy(ta):.3f} vs {entropy(tb):.3f}")
        print("  A: " + ", ".join(f"{k!r} {v:.3f}" for k, v in list(ta.items())[:5]))
        print("  B: " + ", ".join(f"{k!r} {v:.3f}" for k, v in list(tb.items())[:5]))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        result = (hf if mode == "hf" else ollama)(sys.argv[2], sys.argv[3])
        out = sys.argv[sys.argv.index("-o") + 1]
        json.dump(result, open(out, "w"), ensure_ascii=False, indent=1)
        print(f"wrote {out}", file=sys.stderr)
