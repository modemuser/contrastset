"""Command-line entry point: `contrastset grade|generate|check`.

Records go to stdout (or --output), one JSON line per item as soon as it is done.
Progress and reports go to stderr, so `contrastset grade ... | jq` stays clean.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import IO

from contrastset import __version__, jsonl
from contrastset.adapters import make_adapter
from contrastset.records import Entity, GradedEntity, GradedView, Thresholds, sha256_file


def _add_model_args(p: argparse.ArgumentParser, role: str, required: bool = True, prefix: str = "") -> None:
    p.add_argument(f"--{prefix}adapter", default="ollama", choices=["ollama", "openai", "openai-chat", "anthropic", "hf"],
                   help=f"how to reach the {role} model (default: ollama, raw mode)")
    p.add_argument(f"--{prefix}model", required=required, help=f"{role} model name, e.g. qwen3:14b")
    p.add_argument(f"--{prefix}url", help="server base URL (ollama default: http://localhost:11434; "
                                 "openai: e.g. http://localhost:8000/v1)")


def _priors(args: argparse.Namespace) -> int:
    from contrastset import grade

    records = list(jsonl.read(args.priors, GradedEntity))
    annotated = grade.annotate_priors(records, min_entities=args.prior_min, group=args.prior_group,
                                      source=args.prior_source)
    out: IO[str] = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for record in annotated:
            jsonl.write(record, out)
    finally:
        if out is not sys.stdout:
            out.close()
    marked: dict[str, int] = {}
    for r in annotated:
        for q in r.questions:
            if q.prior_guess:
                marked[f"{q.id}={q.prior_answer}"] = marked.get(f"{q.id}={q.prior_answer}", 0) + 1
    print(f"prior guesses (group {args.prior_group or 'all'}, >= {args.prior_min} entities): "
          + (", ".join(f"{k} x{v}" for k, v in sorted(marked.items(), key=lambda kv: -kv[1])) or "none"),
          file=sys.stderr)
    return 0


def _grade(args: argparse.Namespace) -> int:
    from contrastset import grade

    if args.priors:
        return _priors(args)
    if not args.entities or not args.model or not args.template:
        print("grade needs ENTITIES, --model and --template (or --priors GRADED)", file=sys.stderr)
        return 2

    template = grade.Template.load(args.template)
    if args.refusal_phrases:
        phrases: tuple[str, ...] | None = tuple(
            line.strip() for line in Path(args.refusal_phrases).read_text().splitlines() if line.strip())
        source = f"file:{args.refusal_phrases}"
    elif template.refusal_phrases is not None:
        phrases, source = template.refusal_phrases, f"template:{template.name}"
    else:
        phrases, source = None, "built-in"
        print(f"template {template.name} lists no refusal_phrases: using the built-in list for {args.language}, "
              "which was not written against this template's wording", file=sys.stderr)
    cfg = grade.GradeConfig(
        n=args.n, temperature=args.temperature, seed=args.seed, max_tokens=args.max_tokens,
        thresholds=Thresholds(refusal_min=args.refusal_min, correct_min=args.correct_min,
                              consistency_min=args.consistency_min),
        languages=tuple(args.language.split(",")), refusal_phrases=phrases, refusal_source=source, gate=args.gate,
        logprob=args.logprob,
    )
    sampling = {k: v for k, v in (("top_k", args.top_k), ("top_p", args.top_p)) if v is not None}
    if sampling and args.adapter not in ("ollama", "hf"):
        print(f"--top-k/--top-p are supported by the ollama and hf adapters, not {args.adapter}", file=sys.stderr)
        return 2
    adapter = make_adapter(args.adapter, args.model, args.url, **sampling)
    if args.logprob and not hasattr(adapter, "score"):
        print(f"--logprob needs an adapter that exposes log-probabilities (hf), not {args.adapter}", file=sys.stderr)
        return 2
    entities = list(itertools.islice(jsonl.read(args.entities, Entity), args.limit))
    entities_sha = None if args.entities == "-" else sha256_file(args.entities)

    out: IO[str] = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    done: list[GradedEntity] = []
    print(f"grading {len(entities)} entities with {args.model} via {args.adapter}, template {template.name}, "
          f"n={cfg.n} T={cfg.temperature} + greedy", file=sys.stderr)
    try:
        for i, graded in enumerate(grade.grade(adapter, template, entities, cfg, entities_sha), start=1):
            jsonl.write(graded, out)
            done.append(graded)
            print(grade.progress_line(i, len(entities), graded), file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print(f"\ninterrupted after {len(done)} entities; output so far is complete JSONL", file=sys.stderr)
    finally:
        if out is not sys.stdout:
            out.close()
    print("\n" + grade.summary(done), file=sys.stderr)
    return 0


def _write_candidates(args: argparse.Namespace) -> int:
    from contrastset import generate

    if args.graded is None:
        print("--candidates needs --graded", file=sys.stderr)
        return 2
    if args.slot:
        if not args.spec:
            print("--slot needs a spec", file=sys.stderr)
            return 2
        spec, _ = generate.load_spec(args.spec)
        rows = generate.load_pools(spec, args.spec, args.graded)[args.slot]
        what = f"slot {args.slot} after its filters"
    else:
        rows = generate.graded_candidates(list(jsonl.read(args.graded, GradedView)))
        what = "every (entity, question) pair, unfiltered"
    out: IO[str] = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if out is not sys.stdout:
            out.close()
    print(f"{len(rows)} candidates ({what}); each line is a valid `file` slot source line", file=sys.stderr)
    return 0


def _generate(args: argparse.Namespace) -> int:
    from contrastset import generate

    if args.candidates:
        return _write_candidates(args)
    if not args.spec:
        print("generate needs a spec (or --candidates)", file=sys.stderr)
        return 2
    adapter = make_adapter(args.adapter, args.model, args.url) if args.model else None
    result, pairs = generate.run(adapter, args.spec, args.graded, n_frames=args.frames, limit=args.limit,
                                 seed=args.seed, frames_from=args.frames_from)
    origin = (f"generator {args.model} via {args.adapter} ({generate.FRAME_PROMPT_VERSION})" if result.calls
              else "frames from --frames-from" if args.frames_from else "hand-written frames in the spec")
    print(f"{len(result.frames)} frames, {origin}", file=sys.stderr)
    for i, frame in enumerate(result.frames, start=1):
        print(f"  frame {i}: {frame}", file=sys.stderr)
    for frame, reason in result.rejected:
        print(f"  rejected ({reason}): {frame}", file=sys.stderr)
    if result.cells:
        print("  combinations per factor cell: " + ", ".join(f"{k}: {v}" for k, v in result.cells.items()),
              file=sys.stderr)
    out: IO[str] = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    n = 0
    try:
        for n, pair in enumerate(pairs, start=1):
            jsonl.write(pair, out)
            print(f"[{n:>5}] a: {pair.a}\n        b: {pair.b}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print(f"\ninterrupted after {n} pairs; output so far is complete JSONL", file=sys.stderr)
    finally:
        if out is not sys.stdout:
            out.close()
    print(f"\n{n} pairs from {len(result.frames)} frames ({len(result.rejected)} rejected, "
          f"{result.calls} generator calls)", file=sys.stderr)
    return 0


def _check(args: argparse.Namespace) -> int:
    from contrastset import checks, generate, grade
    from contrastset.records import CheckFlag, CheckRef, PairView, Provenance

    pairs = list(jsonl.read(args.pairs, PairView))
    spec = generate.load_spec(args.spec)[0] if args.spec else None
    if args.spec:  # your own checks, named by dotted path, may live next to the spec
        sys.path.insert(0, str(Path(args.spec).resolve().parent))
    inputs = {"pairs": sha256_file(args.pairs)} if args.pairs != "-" else {}
    if args.spec:
        inputs["spec"] = sha256_file(args.spec)
    if args.checks:
        by_use = {ref.use: ref for ref in (spec.checks or [])} if spec else {}
        refs = [by_use.get(name, CheckRef(use=name)) for name in args.checks.split(",")]
    elif spec and spec.checks is not None:
        refs = list(spec.checks)
    else:
        refs = [CheckRef(use=name) for name in checks.DEFAULT_CHECKS]
    res = checks.Resources()
    graded_views = list(jsonl.read(args.graded, GradedView)) if args.graded else []
    if args.graded:
        res.graded = {(g.entity, q.id): {"answers": q.answers, "correct_rate": q.correct_rate}
                      for g in graded_views for q in g.questions}
        inputs["graded"] = sha256_file(args.graded)
    if args.subject_model:
        if not args.template:
            print("a subject model needs --template (the chat format it was graded with)", file=sys.stderr)
            return 2
        res.template = grade.Template.load(args.template)
        if graded_views and not args.allow_template_mismatch:
            try:
                grade.require_same_template(res.template, graded_views)
            except ValueError as exc:
                print(f"{exc} (override with --allow-template-mismatch)", file=sys.stderr)
                return 2
        res.subject = make_adapter(args.subject_adapter, args.subject_model, args.subject_url)
    actives = checks.prepare(refs, pairs, spec)
    unknown_required = set(args.require) - {a.id for a in actives}
    if unknown_required:
        print(f"--require names checks that are not active: {sorted(unknown_required)}", file=sys.stderr)
        return 2
    prov = Provenance(verb="check", model=res.subject.describe() if res.subject else None, inputs=inputs,
                      config={"checks": [r.model_dump(exclude_defaults=True) for r in refs],
                              "require": args.require, "max_fail_rate": args.max_fail_rate,
                              **({"allow_template_mismatch": True} if args.allow_template_mismatch else {})})
    print(f"checking {len(pairs)} pairs with {', '.join(a.id for a in actives)}", file=sys.stderr)
    out: IO[str] = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    failed_pairs = 0
    try:
        for i, pair in enumerate(pairs, start=1):
            verdicts, skipped = [], {}
            for active in actives:
                verdict, why = checks.run_one(active, pair, spec, res)
                if why:
                    skipped[active.id] = why
                elif verdict is not None:
                    verdicts.append(verdict)
            worst = "fail" if any(v.severity == "fail" for v in verdicts) else "warn" if verdicts else "ok"
            failed_pairs += worst == "fail"
            jsonl.write(CheckFlag(pair_id=pair.pair_id, verdict=worst, verdicts=verdicts, skipped=skipped,
                                  provenance=prov), out)
            detail = "; ".join(f"{v.check_id}: {v.reason}" for v in verdicts)
            print(f"[{i:>5}/{len(pairs)}] {pair.pair_id} {worst:<4} {detail}", file=sys.stderr, flush=True)
    finally:
        if out is not sys.stdout:
            out.close()

    width = max(20, *(len(a.id) + 1 for a in actives))
    print(f"\n{'check':<{width}} {'ok':>5} {'warn':>5} {'fail':>5} {'skipped':>8}", file=sys.stderr)
    for a in actives:
        c = a.counts
        print(f"{a.id:<{width}} {c['ok']:>5} {c['warn']:>5} {c['fail']:>5} {c['skipped']:>8}", file=sys.stderr)
    problems = []
    for a in actives:
        if a.counts["skipped"]:
            reasons = ", ".join(f"{why} ({n})" for why, n in a.skip_reasons.items())
            print(f"skipped {a.id} on {a.counts['skipped']} pairs: {reasons}", file=sys.stderr)
            if a.id in args.require:
                problems.append(f"required check {a.id} was skipped on {a.counts['skipped']} pairs")
        summary = getattr(a.check, "summary", None)
        if summary is not None:
            lines, set_failed = summary()
            for line in lines:
                print(line, file=sys.stderr)
            if set_failed:
                problems.append(f"{a.id} failed over the whole set")
    rate = failed_pairs / len(pairs) if pairs else 0.0
    if rate > args.max_fail_rate:
        problems.append(f"{failed_pairs}/{len(pairs)} pairs failed ({rate:.1%} > --max-fail-rate {args.max_fail_rate:.1%})")
    outcome = "FAIL: " + "; ".join(problems) if problems else f"PASS: {failed_pairs}/{len(pairs)} pairs failed"
    print(outcome, file=sys.stderr)
    return 1 if problems else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="contrastset", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"contrastset {__version__}")
    sub = parser.add_subparsers(dest="verb", required=True)

    g = sub.add_parser("grade", help="census of what the subject model knows, graded behaviourally")
    g.add_argument("entities", nargs="?", help="entity JSONL file, or - for stdin")
    _add_model_args(g, "subject", required=False)
    g.add_argument("--template",
                   help="prompt template: a file path, or a built-in chat format (qwen3-nothink). No default: "
                        "the template decides what the model sees, so it is always an explicit choice")
    g.add_argument("-o", "--output", help="write graded JSONL here instead of stdout")
    g.add_argument("--limit", type=int, help="grade only the first N entities")
    g.add_argument("-n", type=int, default=8, help="sampled completions per question (default: 8); 0 with --logprob "
                                                   "grades from log-probabilities alone")
    g.add_argument("--temperature", type=float, default=0.7, help="sampling temperature (default: 0.7)")
    g.add_argument("--seed", type=int, default=0, help="sample i uses seed + i (default: 0)")
    g.add_argument("--max-tokens", type=int, default=32, help="completion length cap (default: 32)")
    defaults = Thresholds()
    g.add_argument("--correct-min", type=float, default=defaults.correct_min,
                   help=f"known if correct share >= this (default: {defaults.correct_min})")
    g.add_argument("--consistency-min", type=float, default=defaults.consistency_min,
                   help=f"confidently_wrong if modal wrong answer share >= this (default: {defaults.consistency_min})")
    g.add_argument("--refusal-min", type=float, default=defaults.refusal_min,
                   help=f"refused if refusal share >= this (default: {defaults.refusal_min})")
    g.add_argument("--language", default="en", help="comma-separated languages for articles and refusal phrases")
    g.add_argument("--refusal-phrases", help="file with one refusal phrase per line; overrides the template's list")
    g.add_argument("--top-k", type=int, help="sample only from the k most likely tokens (ollama, hf); default: "
                                             "the server's/model's own setting for ollama, off for hf")
    g.add_argument("--top-p", type=float, help="nucleus sampling cut-off (ollama, hf); defaults as for --top-k")
    g.add_argument("--logprob", action="store_true",
                   help="also judge each question from untempered log-probabilities (mass of the correct, modal and "
                        "refusal answers), binned with the same thresholds; needs --adapter hf")
    g.add_argument("--priors", metavar="GRADED", help="instead of grading: mark default guesses (prior_guess) "
                                                      "in a finished graded file and write it out")
    g.add_argument("--prior-group", metavar="FIELD", help="group for --priors, e.g. answer.country or meta.country_qid")
    g.add_argument("--prior-min", type=int, default=4, help="entities that must share a wrong answer (default: 4)")
    g.add_argument("--prior-source", choices=["samples", "logprob"], default="samples",
                   help="modal answer from the samples, or the greedy answer of a --logprob grade (default: samples)")
    g.add_argument("--gate", metavar="QUESTION_ID",
                   help="ask this question first and the rest only if it is known, e.g. --gate country")
    g.set_defaults(func=_grade)

    gen = sub.add_parser("generate", help="surface-matched pairs from a spec")
    gen.add_argument("spec", nargs="?", help="spec file, YAML or JSON")
    gen.add_argument("--graded", help="graded JSONL (from `grade` or any tool meeting records.GradedView); "
                                      "needed if any slot uses source 'graded'")
    _add_model_args(gen, "generator", required=False)
    gen.add_argument("--frames-from", metavar="PAIRS", help="reuse the frames of an existing pair file instead of "
                                                           "calling the generator (reproduces Haiku-made pairs)")
    gen.add_argument("--candidates", action="store_true",
                     help="print the flattened (entity, question) candidates of --graded as JSONL and exit; "
                          "each line is a valid `file` slot source line, so filter with jq and feed it back")
    gen.add_argument("--slot", help="with --candidates and a spec: only this slot's candidates, after its filters")
    gen.add_argument("--frames", type=int, default=8, help="frames to ask the generator for (default: 8)")
    gen.add_argument("-o", "--output", help="write pair JSONL here instead of stdout")
    gen.add_argument("--limit", type=int, help="emit at most N pairs")
    gen.add_argument("--seed", type=int, default=0, help="seed for choosing slot combinations (default: 0)")
    gen.set_defaults(func=_generate)

    c = sub.add_parser("check", help="leak detector for any pair file (only a and b are required)")
    c.add_argument("pairs", help="pair JSONL (from `generate` or anywhere: {\"a\": ..., \"b\": ...} per line), or -")
    c.add_argument("--spec", help="the spec: activates its `checks` and enables the example checks")
    c.add_argument("--checks", help="comma-separated checks to run (built-in names or dotted paths); "
                                    f"default: the spec's, else {','.join(['length', 'overlap', 'slot_diff', 'duplicates', 'example_similarity', 'example_reuse'])}")
    c.add_argument("--require", action="append", default=[], metavar="CHECK_ID",
                   help="fail the run if this check was skipped on any pair (repeatable); a skip is not a pass")
    c.add_argument("--max-fail-rate", type=float, default=0.0,
                   help="exit 1 if more than this share of pairs fail (default: 0, any failure)")
    c.add_argument("--graded", help="graded JSONL, for checks that need answers (behaviour)")
    _add_model_args(c, "subject", required=False, prefix="subject-")
    c.add_argument("--template", help="subject prompt template, for checks that ask the subject model; must be the "
                                      "one the --graded file was made with")
    c.add_argument("--allow-template-mismatch", action="store_true",
                   help="ask the subject with a different template than it was graded with (recorded in provenance)")
    c.add_argument("-o", "--output", help="write check JSONL here instead of stdout")
    c.set_defaults(func=_check)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
