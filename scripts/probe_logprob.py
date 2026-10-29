"""Probability of the correct known-clause answer, KK versus KU. Needs the hf adapter.

The accuracy result in FINDINGS.md (an unknown neighbour costs the known clause
nothing) measures only which answer wins. Attention could shift and lower the
model's confidence in the right answer without changing the winner. This scores
log P(correct answer to the shared clause | prompt) on both sides of each pair.

Only pairs whose shared clause comes FIRST are scored: the answer to it is then the
first thing the model writes, right after an otherwise identical prompt. With the
shared clause second, the score would condition on the model's own (different)
answer to the neighbour first, which reintroduces the confound.

A reference answer can be written several ways (Soča, Isonzo); the score is the
log of the summed probability over all surface forms (log-sum-exp). Not part of the
library.

    uv run python scripts/probe_logprob.py PAIRS GRADED --model Qwen/Qwen3-8B -o OUT
"""

import argparse
import json
import math
import sys
import time

from contrastset import jsonl
from contrastset.adapters import HFAdapter
from contrastset.grade import require_same_template, Template
from contrastset.records import GradedEntity, Pair


def logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs")
    ap.add_argument("graded")
    ap.add_argument("--model", required=True, help="HF model id or local path, e.g. Qwen/Qwen3-8B")
    ap.add_argument("--template", required=True, help="the template the graded file was made with")
    ap.add_argument("--limit", type=int)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    answers = {(g.entity, q.id): q.answers for g in jsonl.read(args.graded, GradedEntity) for q in g.questions}
    pairs = [p for p in jsonl.read(args.pairs, Pair) if p.fills["a"][0].slot == "shared"][: args.limit]
    template = Template.load(args.template)
    require_same_template(template, jsonl.read(args.graded, GradedEntity))  # grade and probe with one string
    adapter = HFAdapter(args.model)
    print(f"{len(pairs)} pairs with the shared clause first; model {adapter.describe()}", file=sys.stderr)
    start = time.time()
    with open(args.output, "w", encoding="utf-8") as out:
        for i, pair in enumerate(pairs, start=1):
            shared = pair.fills["a"][0]
            forms = answers[(shared.value, shared.question)]
            record = {"pair_id": pair.pair_id, "shared": shared.value, "frame": pair.frame,
                      "factors": pair.factors, "forms": forms}
            for side in ("a", "b"):
                prompt = template.render(getattr(pair, side), "")
                scores = [adapter.score(prompt, form).logprob for form in forms]
                record[side] = {"text": getattr(pair, side), "logprob": logsumexp(scores), "per_form": scores}
            record["delta_ku_minus_kk"] = record["b"]["logprob"] - record["a"]["logprob"]
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            elapsed = time.time() - start
            print(f"[{i}/{len(pairs)}] {shared.value:<14} delta {record['delta_ku_minus_kk']:+.3f} nats  "
                  f"({elapsed:.0f}s, ~{elapsed / i * (len(pairs) - i):.0f}s left)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
