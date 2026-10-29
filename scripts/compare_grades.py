"""Compare two graded files of the same entities (e.g. Ollama vs HF): per question,
strict bin agreement, greedy-answer agreement, correct_rate correlation. Not part
of the library.

    uv run python scripts/compare_grades.py A.jsonl B.jsonl
"""

import json
import sys
from collections import Counter
from statistics import correlation, mean

from contrastset.grade import first_line, normalise


def load(path: str) -> dict[str, dict]:
    return {r["entity"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


a, b = load(sys.argv[1]), load(sys.argv[2])
shared = [e for e in a if e in b]
print(f"{len(shared)} shared entities ({sys.argv[1]} vs {sys.argv[2]})")
for side, recs in (("A", a), ("B", b)):
    m = next(iter(recs.values()))["provenance"]["model"]
    print(f"  {side}: {m.get('adapter')} {m.get('model')} {m.get('quantization') or m.get('dtype') or ''} "
          f"sampling {m.get('sampling') or m.get('sampling_overrides')}")
qids = [q["id"] for q in a[shared[0]]["questions"]]
for qid in qids:
    pairs = []
    for e in shared:
        qa = next((q for q in a[e]["questions"] if q["id"] == qid), None)
        qb = next((q for q in b[e]["questions"] if q["id"] == qid), None)
        if qa and qb:
            pairs.append((e, qa, qb))
    bins = sum(qa["bin"] == qb["bin"] for _, qa, qb in pairs)
    greedy = lambda q: normalise(first_line(next(s["text"] for s in q["samples"] if s["greedy"])))
    g_same = sum(greedy(qa) == greedy(qb) for _, qa, qb in pairs)
    ra, rb = [qa["correct_rate"] for _, qa, qb in pairs], [qb["correct_rate"] for _, qa, qb in pairs]
    r = correlation(ra, rb) if len(set(ra)) > 1 and len(set(rb)) > 1 else float("nan")
    print(f"\n{qid}: bins agree {bins}/{len(pairs)} = {bins / len(pairs):.0%}   greedy answers agree "
          f"{g_same}/{len(pairs)} = {g_same / len(pairs):.0%}   correct_rate r = {r:.2f}, "
          f"mean A {mean(ra):.2f} B {mean(rb):.2f}")
    confusion = Counter((qa["bin"], qb["bin"]) for _, qa, qb in pairs if qa["bin"] != qb["bin"])
    if confusion:
        print("  disagreements (A -> B): " + ", ".join(f"{x}->{y} x{n}" for (x, y), n in confusion.most_common()))
    for e, qa, qb in pairs:
        if greedy(qa) != greedy(qb):
            print(f"    greedy differs  {e:<17} A {greedy(qa)!r:<22} B {greedy(qb)!r}")
