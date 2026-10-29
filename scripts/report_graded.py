"""Report on a graded file: bins and correct_rate histogram per question, gate pass
rate, and suspicious entries. Not part of the library.

Suspicious entries:
  - alias matches: correct samples that matched only an answer beyond the first two
    (labels), printed for review, junk-looking ones (digits, codes, very short) marked;
  - "no river": modal answers saying the question has no answer (e.g. "None");
  - default guess: per question, the most common modal answer over all entities and
    per country; entities whose modal answer is it are listed, wrong ones first. A
    model that falls back to the big river of the region (Rhine, Vltava) looks
    "consistent" without knowing the town.

    uv run python scripts/report_graded.py GRADED
"""

import json
import re
import sys
from collections import Counter, defaultdict

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
passed = [r for r in rows if not r.get("gate") or r["gate"]["passed"]]
print(f"{len(rows)} entities; gate {rows[0]['gate']['question'] if rows[0].get('gate') else '-'}: "
      f"{len(passed)}/{len(rows)} passed ({len(passed) / len(rows):.0%})")
by_country = defaultdict(lambda: [0, 0])
for r in rows:
    c = r.get("meta", {}).get("country_qid", "?")
    by_country[c][1] += 1
    by_country[c][0] += (not r.get("gate")) or r["gate"]["passed"]
print("  per country (passed/total): " + "  ".join(f"{c} {p}/{n}" for c, (p, n) in sorted(by_country.items())))

BINS = ["known", "confidently_wrong", "unstable", "refused"]
for qid in ["country", "region", "river"]:
    qs = [(r, q) for r in passed for q in r["questions"] if q["id"] == qid]
    if not qs:
        continue
    bins = Counter(q["bin"] for _, q in qs)
    hist = Counter(round(q["correct_rate"] * q["n"]) for _, q in qs)
    n = qs[0][1]["n"]
    print(f"\n{qid} ({len(qs)} towns past the gate)")
    print("  bins: " + "  ".join(f"{b}={bins.get(b, 0)}" for b in BINS))
    print(f"  correct_rate (k/{n}): " + "  ".join(f"{k}/{n}:{hist.get(k, 0)}" for k in range(n + 1)))
    print("  bands: <=1/8 {}  middle {}  >=7/8 {}".format(
        sum(q["correct_rate"] <= 0.125 for _, q in qs),
        sum(0.125 < q["correct_rate"] < 0.875 for _, q in qs),
        sum(q["correct_rate"] >= 0.875 for _, q in qs)))

print("\nalias matches (correct only via an answer beyond the first two; * = junk-looking):")
alias = Counter()
for r in passed:
    for q in r["questions"]:
        for s in q["samples"]:
            if s["matched_answer"] and q["answers"].index(s["matched_answer"]) >= 2:
                alias[(q["id"], s["matched_answer"])] += 1
for (qid, a), k in alias.most_common():
    junk = bool(re.search(r"\d", a)) or len(a) <= 3 or a.isupper()
    print(f"  {'*' if junk else ' '} {qid:<7} {a!r} x{k}")

print("\nmodal answers saying there is no answer:")
none_like = re.compile(r"^(none|no river|there is no|n/?a)\b", re.I)
hits = [(r["entity"], q["id"], q["modal_answer"]) for r in passed for q in r["questions"]
        if q["modal_answer"] and none_like.search(q["modal_answer"])]
print("  " + (", ".join(f"{e} ({qid}: {m!r})" for e, qid, m in hits) if hits else "none"))
print("  (every town has at least one river statement in Wikidata: the seed script only keeps those)")

print("\ndefault guesses (most common modal answer):")
for qid in ["region", "river"]:
    qs = [(r, q) for r in passed for q in r["questions"] if q["id"] == qid and q["modal_answer"]]
    top, k = Counter(q["modal_answer"] for _, q in qs).most_common(1)[0]
    wrong = [r["entity"] for r, q in qs if q["modal_answer"] == top and q["bin"] != "known"]
    print(f"  {qid}: overall {top!r} is the modal answer of {k} towns; wrong for {len(wrong)}: {', '.join(wrong[:12])}")
    per = defaultdict(list)
    for r, q in qs:
        per[r.get("meta", {}).get("country_qid", "?")].append((r, q))
    for c, items in sorted(per.items()):
        guess, kk = Counter(q["modal_answer"] for _, q in items).most_common(1)[0]
        wrong_c = [r["entity"] for r, q in items if q["modal_answer"] == guess and q["bin"] != "known"]
        if kk >= 3 and wrong_c:
            print(f"    {c}: {guess!r} for {kk}/{len(items)} towns, wrong for {len(wrong_c)}: {', '.join(wrong_c[:8])}")
