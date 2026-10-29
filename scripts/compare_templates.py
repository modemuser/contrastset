"""Template A/B on the same entities (grade -n 8 --logprob): refusals, the
prefix-versus-anywhere gap, self-report openings, cell sizes. Not part of the library.

    uv run python scripts/compare_templates.py runs/template-A-29.jsonl runs/template-B-29.jsonl
"""

import json
import re
import sys
from collections import Counter
from statistics import mean

from contrastset.grade import fold_accents, normalise

SELF = re.compile(r"^\s*(yes|no)\b|\bi (do )?know\b|\bi'?m not sure\b|\bi am not sure\b", re.I)


def starts_with_form(text: str, answers: list[str]) -> bool:
    t = normalise(fold_accents(text.strip().lstrip("*").strip()))
    return any(t.startswith(normalise(fold_accents(a))) for a in answers if normalise(a))


for path in sys.argv[1:]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    m = rows[0]["provenance"]["template"]
    print(f"\n== {path}  (template {m['name']})")
    for qid in ("country", "region", "river"):
        qs = [q for r in rows for q in r["questions"] if q["id"] == qid]
        sb = Counter(q["bin"] for q in qs)
        has_lp = all(q.get("logprob_sampler") for q in qs)
        lb = Counter(q["logprob_sampler"]["bin"] for q in qs) if has_lp else Counter()
        samples = [s for q in qs for s in q["samples"] if not s["greedy"]]
        correct = [(s, q) for q in qs for s in q["samples"] if not s["greedy"] and s["correct"]]
        not_prefix = sum(not starts_with_form(s["text"], q["answers"]) for s, q in correct)
        greedy_ref = sum(bool(s["refusal_phrase"]) for q in qs for s in q["samples"] if s["greedy"])
        self_rep = sum(bool(SELF.search(s["text"])) for s in samples)
        print(f"  {qid:<8} sampled bins {dict(sb)}")
        if has_lp:
            print(f"  {'':<8} logprob bins (T=0.7) {dict(lb)}")
        print(f"  {'':<8} refusals: sampled-refused {sb.get('refused', 0)}, logprob-refused {lb.get('refused', 0)}, "
              f"greedy refusals {greedy_ref}, samples with a refusal phrase {sum(bool(s['refusal_phrase']) for s in samples)}"
              f"/{len(samples)}" + (f", mean p_refusal {mean(q['logprob']['p_refusal'] for q in qs):.3f}" if has_lp else ""))
        lp_part = (f" vs prefix p_correct(T=0.7) {mean(q['logprob_sampler']['p_correct'] for q in qs):.2f}"
                   if has_lp else "")
        print(f"  {'':<8} correct samples not starting with the name: {not_prefix}/{len(correct)};  "
              f"sampled correct_rate {mean(q['correct_rate'] for q in qs):.2f}{lp_part};  "
              f"self-report openings {self_rep}/{len(samples)}")
