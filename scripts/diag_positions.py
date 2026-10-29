"""Where do HF and Ollama samples split? 200 full answers per town on each backend,
compared token position by token position. Not part of the library.

    uv run python scripts/diag_positions.py ollama qwen3:8b-fp16 TEMPLATE -o runs/pos-ollama.json
    uv run python scripts/diag_positions.py hf Qwen/Qwen3-8B TEMPLATE -o runs/pos-hf.json
    uv run python scripts/diag_positions.py compare runs/pos-hf.json runs/pos-ollama.json
"""

import json
import sys
from collections import Counter

import httpx

TOWNS = ["Ronda", "Písek"]
N, MAX_TOKENS = 200, 12
SAMPLING = {"temperature": 0.7, "top_k": 20, "top_p": 0.95}


def ollama(model: str, template: str) -> dict:
    from contrastset.grade import Template

    t = Template.load(template)
    out = {}
    for town in TOWNS:
        prompt = t.render(f"Which river flows through {town}?", "")
        seqs = []
        for seed in range(N):
            r = httpx.post("http://localhost:11434/api/generate", timeout=120, json={
                "model": model, "prompt": prompt, "raw": True, "stream": False, "logprobs": True,
                "options": {**SAMPLING, "num_predict": MAX_TOKENS, "seed": seed}}).json()
            seqs.append([step["token"] for step in r.get("logprobs") or []])
            if seed % 50 == 49:
                print(f"ollama {town}: {seed + 1}/{N}", file=sys.stderr, flush=True)
        out[town] = seqs
    return out


def hf(model: str, template: str) -> dict:
    from contrastset.adapters import HFAdapter
    from contrastset.grade import Template

    a, t = HFAdapter(model, top_k=SAMPLING["top_k"], top_p=SAMPLING["top_p"]), Template.load(template)
    out = {}
    for town in TOWNS:
        ids = a._ids(t.render(f"Which river flows through {town}?", ""))
        seqs = []
        for seed in range(N):
            a._torch.manual_seed(seed)
            gen = a.model.generate(ids, max_new_tokens=MAX_TOKENS, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                   logits_processor=a._warpers(SAMPLING["temperature"]),
                                   pad_token_id=a.tokenizer.eos_token_id)
            new = gen[0, ids.shape[1]:].tolist()
            seqs.append([a.tokenizer.decode([i]) for i in new])
            if seed % 50 == 49:
                print(f"hf {town}: {seed + 1}/{N}", file=sys.stderr, flush=True)
        out[town] = seqs
    return out


def compare(a_path: str, b_path: str) -> None:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    for town in TOWNS:
        print(f"{town}  (A = {a_path}, B = {b_path})")
        for pos in range(4):
            # distribution of the token at `pos` given the same prefix (the modal prefix)
            ca = Counter(s[pos] for s in a[town] if len(s) > pos)
            cb = Counter(s[pos] for s in b[town] if len(s) > pos)
            na, nb = sum(ca.values()), sum(cb.values())
            tv = 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in set(ca) | set(cb)) if na and nb else float("nan")
            print(f"  token {pos + 1}: distinct A {len(ca):>2} B {len(cb):>2}  total variation {tv:.3f}  "
                  f"A {', '.join(f'{k!r}:{v}' for k, v in ca.most_common(4))}  |  B {', '.join(f'{k!r}:{v}' for k, v in cb.most_common(4))}")
        ends_a = Counter(len(s) for s in a[town])
        ends_b = Counter(len(s) for s in b[town])
        print(f"  answer length in tokens  A {dict(sorted(ends_a.items()))}  B {dict(sorted(ends_b.items()))}")
        full_a = Counter("".join(s).strip().splitlines()[0] if "".join(s).strip() else "" for s in a[town])
        full_b = Counter("".join(s).strip().splitlines()[0] if "".join(s).strip() else "" for s in b[town])
        print(f"  distinct whole answers A {len(full_a)} B {len(full_b)};  top A {full_a.most_common(3)}  top B {full_b.most_common(3)}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        data = (ollama if mode == "ollama" else hf)(sys.argv[2], sys.argv[3])
        json.dump(data, open(sys.argv[sys.argv.index("-o") + 1], "w"), ensure_ascii=False)
