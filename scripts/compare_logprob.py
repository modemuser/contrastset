"""Log-prob grade (-n 0 --logprob) vs sampled grades of the same entities: bin
agreement at T = 1 and under the sampling distribution, and the correlation and
slope between p_correct and the sampled correct_rate, with the largest residuals.
Not part of the library.

    uv run python scripts/compare_logprob.py LOGPROB SAMPLED [SAMPLED ...]
"""

import json
import sys
from statistics import correlation, mean


def load(p: str) -> dict[str, dict]:
    return {r["entity"]: r for r in map(json.loads, open(p, encoding="utf-8"))}


def q(r: dict, qid: str) -> dict:
    return next(x for x in r["questions"] if x["id"] == qid)


lp = load(sys.argv[1])
bounds = [x[k] for r in lp.values() for qq in r["questions"] for x in (qq["logprob"], qq["logprob_sampler"])
          for k in ("p_correct_pruned_max", "p_modal_pruned_max") if x.get(k) is not None]
print(f"pruning bound over all questions: max {max(bounds):.1e}, mean {mean(bounds):.1e}")
for path in sys.argv[2:]:
    smp = load(path)
    m = next(iter(smp.values()))["provenance"]["model"]
    print(f"\nvs {path}  ({m.get('adapter')} {m.get('model')}, sampled n=8)")
    for qid in ("country", "region", "river"):
        es = [e for e in lp if e in smp]
        sr = [q(smp[e], qid)["correct_rate"] for e in es]
        out = []
        for key, label in (("logprob", "T=1  "), ("logprob_sampler", "T=0.7")):
            pc = [q(lp[e], qid)[key]["p_correct"] for e in es]
            agree = sum(q(lp[e], qid)[key]["bin"] == q(smp[e], qid)["bin"] for e in es)
            r = correlation(pc, sr) if len(set(pc)) > 1 and len(set(sr)) > 1 else float("nan")
            mx, my = mean(pc), mean(sr)
            slope = sum((x - mx) * (y - my) for x, y in zip(pc, sr)) / max(1e-12, sum((x - mx) ** 2 for x in pc))
            out.append(f"{label}: bins {agree}/{len(es)}, r {r:.2f}, slope {slope:.2f}, mean p {mx:.2f} vs sampled {my:.2f}")
        print(f"  {qid:<8} " + "  |  ".join(out))
        res = sorted(((abs(q(lp[e], qid)["logprob_sampler"]["p_correct"] - q(smp[e], qid)["correct_rate"]), e) for e in es),
                     reverse=True)[:3]
        print("           largest |p(T=0.7) - sampled|: " + ", ".join(
            f"{e} {q(lp[e], qid)['logprob_sampler']['p_correct']:.2f} vs {q(smp[e], qid)['correct_rate']:.2f}" for _, e in res))
