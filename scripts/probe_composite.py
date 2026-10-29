"""Ask the subject model both sides of every pair and score each clause separately.

Question this answers: does knowledge graded on single questions survive inside a
composite question? For a "one known, one unknown" side we want the known clause
still answered correctly and the unknown clause still guessed at; for "two known"
both answered. Not part of the library; a research probe on a pair file.

    uv run python scripts/probe_composite.py PAIRS GRADED --model qwen3:14b [--limit N] -o OUT

Note: runs before 2026-10-28 evening used seed 0 (the default then).
"""

import argparse
import json
import sys
import time

from contrastset import jsonl
from contrastset.adapters import OllamaAdapter
from contrastset.grade import require_same_template, REFUSAL_PHRASES, Template, detect_refusal, normalise
from contrastset.records import GradedEntity, Pair


def contains_any(text: str, answers: list[str]) -> bool:
    whole = f" {normalise(text)} "
    return any(f" {normalise(a)} " in whole for a in answers if normalise(a))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("graded")
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", required=True, help="the template the graded file was made with")
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=10_000,
                    help="sampling seeds start here; keep them disjoint from the grade's (default 10000)")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    answers = {}
    for g in jsonl.read(args.graded, GradedEntity):
        for q in g.questions:
            answers[(g.entity, q.id)] = q.answers
    pairs = list(jsonl.read(args.pairs, Pair))[: args.limit]
    adapter = OllamaAdapter(args.model)
    template = Template.load(args.template)
    require_same_template(template, jsonl.read(args.graded, GradedEntity))  # grade and probe with one string
    phrases = REFUSAL_PHRASES["en"]
    start = time.time()
    with open(args.output, "w", encoding="utf-8") as out:
        for i, pair in enumerate(pairs, start=1):
            for side in ("a", "b"):
                fills = pair.fills[side]
                text = getattr(pair, side)
                prompt = template.render(text, "")
                completions = [*adapter.complete(prompt, temperature=0, max_tokens=args.max_tokens),
                               *adapter.complete(prompt, n=args.n, temperature=0.7, seed=args.seed, max_tokens=args.max_tokens)]
                refs = [answers[(f.value, f.question)] for f in fills]
                # Clauses whose reference answers overlap (two towns on the same river)
                # cannot be told apart by string match.
                ambiguous = any(normalise(x) == normalise(y) for i, r1 in enumerate(refs) for r2 in refs[i + 1:]
                                for x in r1 for y in r2)
                clauses = [{"value": f.value, "slot": f.slot, "graded_bin": f.bin,
                            "correct": [contains_any(c, ref) for c in completions]}
                           for f, ref in zip(fills, refs)]
                record = {"pair_id": pair.pair_id, "side": side, "frame": pair.frame, "text": text,
                          "ambiguous": ambiguous, "clauses": clauses, "completions": completions,
                          "refusal": [detect_refusal(c, phrases) for c in completions]}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
            elapsed = time.time() - start
            print(f"[{i}/{len(pairs)}] {elapsed:.0f}s elapsed, ~{elapsed / i * (len(pairs) - i):.0f}s left",
                  file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
