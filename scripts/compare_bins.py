"""Sampled bins vs logprob bins within one `grade --logprob` file, per question.
Not part of the library.

    uv run python scripts/compare_bins.py GRADED
"""

import json
import sys
from collections import Counter
from statistics import correlation, mean

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
for qid in [q["id"] for q in rows[0]["questions"]]:
    qs = [(r["entity"], q) for r in rows for q in r["questions"] if q["id"] == qid and q.get("logprob")]
    agree = sum(q["bin"] == q["logprob"]["bin"] for _, q in qs)
    sampled = [q["correct_rate"] for _, q in qs]
    lp = [q["logprob"]["p_correct"] for _, q in qs]
    r = correlation(sampled, lp) if len(set(sampled)) > 1 and len(set(lp)) > 1 else float("nan")
    print(f"{qid}: sampled vs logprob bins agree {agree}/{len(qs)} = {agree / len(qs):.0%};  "
          f"correct_rate vs p_correct r = {r:.2f} (means {mean(sampled):.2f} / {mean(lp):.2f});  "
          f"mean p_refusal {mean(q['logprob']['p_refusal'] for _, q in qs):.3f}")
    conf = Counter((q["bin"], q["logprob"]["bin"]) for _, q in qs if q["bin"] != q["logprob"]["bin"])
    if conf:
        print("  sampled -> logprob: " + ", ".join(f"{a}->{b} x{n}" for (a, b), n in conf.most_common()))
    for e, q in qs:
        if q["bin"] != q["logprob"]["bin"]:
            l = q["logprob"]
            print(f"    {e:<17} sampled {q['n_correct']}/{q['n']} cons {q['consistency']}  |  p_correct {l['p_correct']:.2f} "
                  f"p_modal {l['p_modal']:.2f} ({l['modal_answer']!r}) cons {l['consistency']:.2f}")
