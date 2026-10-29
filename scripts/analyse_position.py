"""Experiment 1 analysis: is the middle-band town answered more often in the second
clause? Reads probe_composite.py output. Not part of the library.

    uv run python scripts/analyse_position.py TWO_CLAUSE_PROBE [THREE_CLAUSE_PROBE]
"""

import json
import math
import re
import sys
from collections import defaultdict

from contrastset.grade import REFUSAL_PHRASES, detect_refusal, normalise


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def sign_test(diffs: list[float]) -> float:
    """Two-sided sign test on non-zero paired differences."""
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n, k = pos + neg, min(pos, neg)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def middle(row: dict) -> tuple[int, dict]:
    for i, c in enumerate(row["clauses"]):
        if c["slot"] == "middle":
            return i, c
    raise ValueError("no middle slot")


two = load(sys.argv[1])
per_town: dict[str, dict[str, list[bool]]] = defaultdict(lambda: {"first": [], "second": []})
partner_ok = {"first": [0, 0], "second": [0, 0]}
for r in two:
    i, c = middle(r)
    pos = "first" if i == 0 else "second"
    per_town[c["value"]][pos] += c["correct"]
    k = r["clauses"][1 - i]
    partner_ok[pos][0] += sum(k["correct"]); partner_ok[pos][1] += len(k["correct"])
print("two-clause: middle-band town correct, first vs second position (5 completions each)")
diffs, tot = [], {"first": [0, 0], "second": [0, 0]}
for town, d in sorted(per_town.items()):
    f, s = sum(d["first"]), sum(d["second"])
    tot["first"][0] += f; tot["first"][1] += len(d["first"])
    tot["second"][0] += s; tot["second"][1] += len(d["second"])
    diffs.append(s / len(d["second"]) - f / len(d["first"]))
    print(f"  {town:<26} first {f}/{len(d['first'])}  second {s}/{len(d['second'])}")
for pos in ("first", "second"):
    a, n = tot[pos]
    print(f"  total {pos:<6} {a}/{n} = {a / n:.2f}   (known partner in the other clause: "
          f"{partner_ok[pos][0]}/{partner_ok[pos][1]})")
print(f"  towns better second / same / better first: {sum(d > 0 for d in diffs)} / {sum(d == 0 for d in diffs)} / "
      f"{sum(d < 0 for d in diffs)};  sign test p = {sign_test(diffs):.3f}")

print("\ncontrol (b): raw first-position answers to the middle-band town's clause")
phrases = REFUSAL_PHRASES["en"]
classes: dict[str, int] = defaultdict(int)
for town in sorted(per_town)[:10]:
    for r in two:
        i, c = middle(r)
        if c["value"] != town or i != 0:
            continue
        shown = []
        for comp, ok in zip(r["completions"], c["correct"]):
            first = re.split(r"\n|;|,| and ", comp.strip(), maxsplit=1)[0].strip(" .*")
            if ok:
                kind = "correct"
            elif not first:
                kind = "empty"
            elif detect_refusal(comp, phrases) or re.search(r"\b(maybe|possibly|probably|i think)\b", comp, re.I):
                kind = "hedge"
            elif len(first.split()) > 5 or not re.search(r"[^\W\d_]", first):
                kind = "malformed"
            else:
                kind = "wrong river"
            classes[kind] += 1
            shown.append(f"{first!r}:{kind}")
        print(f"  {town:<26} " + "  ".join(shown))
print("  classes: " + ", ".join(f"{k} {v}" for k, v in sorted(classes.items())))

if len(sys.argv) > 2:
    three = load(sys.argv[2])
    names = ["first", "middle", "last"]
    pos_tot = {p: [0, 0] for p in names}
    town3: dict[str, dict[str, list[bool]]] = defaultdict(lambda: {p: [] for p in names})
    for r in three:
        i, c = middle(r)
        pos_tot[names[i]][0] += sum(c["correct"]); pos_tot[names[i]][1] += len(c["correct"])
        town3[c["value"]][names[i]] += c["correct"]
    print("\ncontrol (a): three clauses, middle-band town by position (each position twice per town)")
    for town, d in sorted(town3.items()):
        print(f"  {town:<26} " + "  ".join(f"{p} {sum(d[p])}/{len(d[p])}" for p in names))
    print("  total  " + "  ".join(f"{p} {a}/{n} = {a / n:.2f}" for p, (a, n) in pos_tot.items()))
