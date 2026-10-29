"""`generate`: surface-matched pairs from a spec.

Two stages, so that a and b can only differ where the contrast is:

1. Frames. The generator model writes entity-free frames: sentences with [[X1]],
   [[X2]], ... placeholders, one per clause position. It sees the contrast, the
   example pairs and what each position asks about, but never the entities being
   contrasted and never what the subject model knows.
2. Fill. For each combination of slot values, one frame is filled with side a's
   values and again with side b's. Everything except the placeholders is identical
   by construction.

The frame prompt is part of the library, versioned (FRAME_PROMPT_VERSION), and
recorded with its hash in every pair. Users write the spec, not the prompt.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from contrastset import jsonl
from contrastset.adapters import Adapter
from contrastset.records import (
    Factor,
    Fill,
    FileSource,
    GradedSource,
    GradedView,
    InlineSource,
    Pair,
    PairView,
    Provenance,
    Range,
    SlotDefaults,
    Slots,
    SlotSource,
    Spec,
    Where,
    sha256_text,
)

# v2: frames must not ask about anyone's knowledge or certainty. v1 produced "Do you
# know which river flows through [[X1]] ...", which turns the stimulus into a
# self-report question, the very variable under study.
# v3: one clause per placeholder, no lists, no extra task. v2 produced "Which rivers
# flow through [[X1]] and [[X2]], respectively?" (one clause), "For [[X1]], [[X2]] and
# [[X3]], name the river ..." (a list) and "Does the river through [[X1]] differ from
# ..." (an added comparison), which break clause positions as a controlled variable.
# v4: the clause rule of v3 is opt-in (spec constraints.one_clause_per_slot), since it
# is wrong for statement-style contrasts; the knowledge rule no longer assumes the
# frame is a question.
FRAME_PROMPT_VERSION = "frames/v4"
FRAME_PROMPT = """\
You are writing sentence frames for a dataset of matched text pairs, used in an \
interpretability experiment on a small language model.

The two sides of every pair must differ in exactly one thing: {contrast}.
To guarantee that, each frame is filled twice with different values, and nothing \
else changes. Your frames must therefore read naturally whatever value goes into \
each placeholder, and must not hint at any property of the values (famous or \
obscure, real or invented, easy or hard). Never ask or mention whether anyone \
knows, remembers, is sure of, or can tell something.

Each frame is a single sentence that uses each of these placeholders exactly once, \
in this order:
{positions}
{clause_rule}
Example pairs, for style only. Do not reuse any of their names:
{examples}
{constraints}
Write {k} frames that differ from each other in wording and sentence structure, \
while staying plain and natural. Return only a JSON array of {k} strings, nothing else."""

CLAUSE_RULE = """
Give every placeholder its own clause, and let each clause ask its question in full \
(a verb and the question word in every clause). Never put two placeholders in one \
clause, never list them ("[[X1]], [[X2]] and [[X3]]"), and never add a task beyond \
the questions, such as comparing the answers.
"""

PLACEHOLDER = re.compile(r"\[\[X(\d+)\]\]")


def placeholder(i: int) -> str:
    return f"[[X{i}]]"


# --- spec and candidates ------------------------------------------------------------


def load_spec(path: str | Path) -> tuple[Spec, str]:
    """Load a YAML or JSON spec; return it with the SHA-256 of the file's bytes."""
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw)  # YAML is a superset of JSON, so this reads both
    return Spec.model_validate(data), hashlib.sha256(raw).hexdigest()


Candidate = dict[str, Any]  # flat fields; always has "value"


_SKIP_EXTRA = {"samples", "provenance", "thresholds", "bin_counts", "aggregation", "gate", "self_report"}


def graded_candidates(graded: Sequence[GradedView]) -> list[Candidate]:
    """One candidate per (entity, question) pair: the flattened view selectors and
    `same` / factors refer to (field list in records.GradedSource). This is also
    exactly what `generate --candidates` prints, one JSON line per candidate, so the
    view can be inspected and filtered with jq and fed back as a `file` slot source.

    Unknown fields of the graded line and of the question pass through (except the
    bulky raw evidence: samples, provenance)."""
    out = []
    for g in graded:
        others = {f"q.{q.id}": q.bin.value for q in g.questions}
        others |= {f"answer.{q.id}": q.answers[0] for q in g.questions if q.answers}
        meta = {f"meta.{k}": v for k, v in g.meta.items()}
        top_extra = {k: v for k, v in (g.model_extra or {}).items() if k not in _SKIP_EXTRA}
        for q in g.questions:
            q_extra = {k: v for k, v in (q.model_extra or {}).items() if k not in _SKIP_EXTRA}
            core = {"value": g.entity, "entity": g.entity, "question": q.id, "question_text": q.q,
                    "bin": q.bin.value, "entity_bin": g.bin.value if g.bin else None, "type": g.type,
                    "correct_rate": q.correct_rate, "consistency": q.consistency, "refusal_rate": q.refusal_rate,
                    "n_correct": q.n_correct, "n": q.n, "prior_guess": q.prior_guess}
            out.append({**top_extra, **q_extra, **meta, **others, **core})
    return out


def _value_candidates(values: Sequence[str | dict[str, Any]]) -> list[Candidate]:
    out = []
    for v in values:
        candidate = {"value": v} if isinstance(v, str) else dict(v)
        if "value" not in candidate:
            raise ValueError(f"slot value {v!r} has no 'value' key")
        out.append(candidate)
    return out


def _matches(candidate: Candidate, where: Where) -> bool:
    for key, allowed in where.items():
        if isinstance(allowed, Range):
            if not allowed.contains(candidate.get(key)):
                return False
            continue
        if isinstance(allowed, bool):  # e.g. {prior_guess: true}; unset (None) never matches
            if candidate.get(key) is not allowed:
                return False
            continue
        options = allowed if isinstance(allowed, list) else [allowed]
        if str(candidate.get(key)) not in options:
            return False
    return True


def _familiar(candidate: Candidate, familiar: bool | list[str]) -> bool:
    """Every other question of the candidate's entity (or the listed ones) is known."""
    if familiar is False:
        return True
    own = f"q.{candidate['question']}"
    keys = [k for k in candidate if k.startswith("q.") and k != own] if familiar is True \
        else [f"q.{qid}" for qid in familiar if f"q.{qid}" != own]
    return all(candidate.get(k) == "known" for k in keys)


def slot_candidates(source: SlotSource, graded: list[Candidate] | None, spec_dir: Path,
                    defaults: SlotDefaults | None = None) -> list[Candidate]:
    if isinstance(source, GradedSource):
        if graded is None:
            raise ValueError("a slot uses source 'graded' but no --graded file was given")
        familiar = source.familiar if source.familiar is not None else (defaults or SlotDefaults()).familiar
        pool = [c for c in graded if _familiar(c, familiar)]
    elif isinstance(source, InlineSource):
        pool = _value_candidates(source.values)
    elif isinstance(source, FileSource):
        path = Path(source.path) if Path(source.path).is_absolute() else spec_dir / source.path
        lines = path.read_text(encoding="utf-8").splitlines()
        pool = _value_candidates([json.loads(line) for line in lines if line.strip()])
    return [c for c in pool if _matches(c, source.where)]


# --- combinations -------------------------------------------------------------------

Combo = dict[str, Candidate]  # slot name -> chosen candidate


def _valid(combo: Combo, same: Sequence[str]) -> bool:
    values = [c["value"] for c in combo.values()]
    if len(set(values)) != len(values):
        return False
    return all(len({str(c.get(key)) for c in combo.values()}) == 1 for key in same)


def factor_levels(combo: Combo, factors: dict[str, Factor]) -> dict[str, str] | None:
    """The level of every factor for this combination, or None if some factor is
    neither uniformly same nor uniformly different, or at a level not asked for."""
    levels: dict[str, str] = {}
    for name, factor in factors.items():
        equal = {str(combo[x].get(factor.field)) == str(combo[y].get(factor.field)) for x, y in factor.between}
        if len(equal) != 1:
            return None
        level = "same" if equal.pop() else "different"
        if level not in factor.levels:
            return None
        levels[name] = level
    return levels


def combos(pools: dict[str, list[Candidate]], same: Sequence[str], rng: random.Random,
           limit: int | None, factors: dict[str, Factor] | None = None,
           max_enumerate: int = 200_000) -> Iterator[tuple[Combo, dict[str, str]]]:
    """Distinct valid slot combinations with their factor levels, in random (seeded)
    order, dealt round-robin across factor cells so every cell gets an equal share
    for as long as it has combinations left.

    Small spaces are enumerated; large ones are sampled, which needs a limit."""
    factors = factors or {}
    names = list(pools)
    size = 1
    for pool in pools.values():
        size *= len(pool)
    if size == 0:
        empty = [n for n, p in pools.items() if not p]
        raise ValueError(f"no candidates for slots {empty}: check their `where` filters")
    cells: dict[tuple[str, ...], list[tuple[Combo, dict[str, str]]]] = {}
    if size <= max_enumerate:
        for choice in itertools.product(*pools.values()):
            combo = dict(zip(names, choice))
            levels = factor_levels(combo, factors) if _valid(combo, same) else None
            if levels is not None:
                cells.setdefault(tuple(levels.values()), []).append((combo, levels))
    else:
        if limit is None:
            raise ValueError(f"{size} possible combinations: pass --limit to sample from them")
        seen: set[tuple[str, ...]] = set()
        for _ in range(limit * 1000):
            combo = {n: rng.choice(pools[n]) for n in names}
            key = tuple(f"{c['value']}|{c.get('question')}" for c in combo.values())
            levels = factor_levels(combo, factors) if key not in seen and _valid(combo, same) else None
            if levels is not None:
                seen.add(key)
                cells.setdefault(tuple(levels.values()), []).append((combo, levels))
                if len(seen) >= limit * 4:
                    break
    if not cells:
        raise ValueError("no slot combination satisfies `same`, distinct values and the factors")
    queues = []
    for key in sorted(cells):
        rng.shuffle(cells[key])
        queues.append(iter(cells[key]))
    emitted = 0
    while queues:
        for queue in list(queues):
            item = next(queue, None)
            if item is None:
                queues.remove(queue)
                continue
            yield item
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def cell_sizes(pools: dict[str, list[Candidate]], spec: Spec) -> dict[str, int]:
    """How many combinations each factor cell has (enumerable spaces only), for the log."""
    counts: dict[str, int] = {}
    for combo, levels in combos(pools, slots_of(spec).same, random.Random(0), None, slots_of(spec).factors):
        key = ", ".join(f"{k}={v}" for k, v in levels.items()) or "all"
        counts[key] = counts.get(key, 0) + 1
    return counts


# --- frames -------------------------------------------------------------------------


def _describe_position(i: int, slot_names: Sequence[str], pools: dict[str, list[Candidate]]) -> str:
    """What the clause at position i is about, without naming any candidate entity."""
    descriptions = set()
    for name in slot_names:
        sample = pools[name][0]
        if "question_text" in sample:
            text = sample["question_text"].replace(sample["value"], placeholder(i))
            descriptions.add(f'a {sample.get("type", "name")}; the clause asks: "{text}"')
        else:
            descriptions.add("one of the slot values")
    if len(descriptions) != 1:
        raise ValueError(f"position {i} holds slots asking different things ({sorted(descriptions)}); "
                         "frames need every side to ask the same thing at the same position")
    return f"- {placeholder(i)}: replaced by {descriptions.pop()}"


def frame_prompt(spec: Spec, pools: dict[str, list[Candidate]], k: int) -> str:
    sides = slots_of(spec).resolved_sides()
    width = len(sides["a"])
    positions = []
    for i in range(width):
        at_i = {sides[side][perm[i]] for perm in clause_orders(width, slots_of(spec).order) for side in ("a", "b")}
        positions.append(_describe_position(i + 1, sorted(at_i), pools))
    examples = "\n".join(f"a: {e.a}\nb: {e.b}\n" for e in spec.examples)
    c = spec.constraints
    rules = []
    if c.language:
        rules.append(f"Language: {c.language}.")
    if c.register_:
        rules.append(f"Register: {c.register_}.")
    if c.min_words or c.max_words:
        rules.append(f"Length: {c.min_words or 1} to {c.max_words or 'any number of'} words, "
                     "counting each placeholder as one word.")
    if c.forbidden_words:
        rules.append("Never use these words: " + ", ".join(c.forbidden_words) + ".")
    constraints = ("Constraints:\n" + "\n".join(f"- {r}" for r in rules) + "\n") if rules else ""
    clause_rule = CLAUSE_RULE if c.one_clause_per_slot else ""
    return FRAME_PROMPT.format(contrast=spec.contrast, positions="\n".join(positions), examples=examples,
                               constraints=constraints, k=k, clause_rule=clause_rule)


def example_names(spec: Spec) -> set[str]:
    """Capitalised words inside the example pairs (not sentence-initial): the examples'
    entities, which must not leak into frames."""
    names = set()
    for e in spec.examples:
        for text in (e.a, e.b):
            words = re.findall(r"[\w'-]+", text)
            names |= {w for w in words[1:] if w[:1].isupper()}
    return names


def frame_problem(frame: str, width: int, spec: Spec, banned: set[str]) -> str | None:
    """Why a frame is unusable, or None if it is fine."""
    found = [int(m) for m in PLACEHOLDER.findall(frame)]
    if found != list(range(1, width + 1)):
        return f"placeholders {found}, expected {list(range(1, width + 1))} once each in order"
    if "[[" in PLACEHOLDER.sub("", frame):
        return "stray [[...]] marker"
    if spec.constraints.one_clause_per_slot:
        for gap in re.split(r"\[\[X\d+\]\]", frame)[1:-1]:
            if len(re.findall(r"[\w'-]+", gap)) < 2:  # "[[X1]] and [[X2]]", "[[X1]], [[X2]]"
                return f"placeholders share a clause (only {gap.strip()!r} between them)"
    words = re.findall(r"[\w'-]+", PLACEHOLDER.sub("X", frame))
    c = spec.constraints
    if c.min_words and len(words) < c.min_words:
        return f"{len(words)} words < {c.min_words}"
    if c.max_words and len(words) > c.max_words:
        return f"{len(words)} words > {c.max_words}"
    lowered = {w.casefold() for w in words}
    hits = [w for w in c.forbidden_words if w.casefold() in lowered]
    if hits:
        return f"forbidden words {hits}"
    leaked = sorted(set(words) & banned)
    if leaked:
        return f"reuses example names {leaked}"
    return None


def parse_frames(text: str) -> list[str]:
    """The JSON array in the generator's reply, tolerating a ```json fence."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"generator reply has no JSON array: {text[:200]!r}")
    frames = json.loads(match.group(0))
    if not all(isinstance(f, str) for f in frames):
        raise ValueError("generator reply is not a list of strings")
    return [f.strip() for f in frames]


@dataclass
class FrameResult:
    frames: list[str]
    rejected: list[tuple[str, str]]  # (frame, reason)
    prompt: str
    calls: int
    cells: dict[str, int] = field(default_factory=dict)  # combinations per factor cell


def make_frames(adapter: Adapter, spec: Spec, pools: dict[str, list[Candidate]], k: int,
                max_rounds: int = 3, max_tokens: int = 4096) -> FrameResult:
    width = len(slots_of(spec).resolved_sides()["a"])
    prompt = frame_prompt(spec, pools, k)
    banned = example_names(spec)
    frames: list[str] = []
    rejected: list[tuple[str, str]] = []
    calls = 0
    while len(frames) < k and calls < max_rounds:
        calls += 1
        [reply] = adapter.complete(prompt, n=1, max_tokens=max_tokens)
        for frame in parse_frames(reply):
            problem = frame_problem(frame, width, spec, banned)
            if problem is None and frame in frames:
                problem = "duplicate"
            if problem:
                rejected.append((frame, problem))
            elif len(frames) < k:
                frames.append(frame)
    if not frames:
        raise ValueError(f"no usable frames after {calls} generator calls; rejected: {rejected}")
    return FrameResult(frames, rejected, prompt, calls)


# --- pairs --------------------------------------------------------------------------


def clause_orders(width: int, order: str) -> list[list[int]]:
    """Index orders applied to both sides: as written, plus reversed ("both") or
    every rotation ("rotations": with 3 clauses, each slot once first, middle, last)."""
    identity = list(range(width))
    if order == "both":
        return [identity, identity[::-1]]
    if order == "rotations":
        return [identity[k:] + identity[:k] for k in range(width)]
    return [identity]


def _order_name(order: str, k: int) -> str:
    if order == "both":
        return "as_written" if k == 0 else "reversed"
    return f"rotation_{k}"


def fill(frame: str, values: Sequence[str]) -> str:
    return PLACEHOLDER.sub(lambda m: values[int(m.group(1)) - 1], frame)


_FILL_KEYS = ("question", "question_text", "bin", "entity_bin", "type")


def _fill_record(slot: str, c: Candidate, source: Literal["graded", "inline", "file"]) -> Fill:
    """Known keys (question, bin, ...) become Fill fields whatever the source, so
    candidates piped through jq into a file source keep them; the rest goes to fields."""
    lifted = {k: c[k] for k in _FILL_KEYS if c.get(k) is not None}
    rest = {k: v for k, v in c.items() if k not in _FILL_KEYS and k != "value"}
    if source == "graded":
        rest = {}  # the graded file is the record of everything else
    return Fill(slot=slot, value=c["value"], source=source, fields=rest, **lifted)


def pairs(spec: Spec, spec_sha256: str, frames: Sequence[str],
          combo_iter: Iterator[tuple[Combo, dict[str, str]]], prov: Provenance,
          limit: int | None = None) -> Iterator[Pair]:
    sides = slots_of(spec).resolved_sides()
    orders = clause_orders(len(sides["a"]), slots_of(spec).order)
    sources = {name: src.source for name, src in slots_of(spec).select.items()}
    emitted = 0
    for i, (combo, levels) in enumerate(combo_iter):
        frame = frames[i % len(frames)]
        for k, perm in enumerate(orders):
            names = {side: [slots[j] for j in perm] for side, slots in sides.items()}
            if len(orders) > 1:  # which clause order this pair uses is a design factor too
                levels = {**levels, "clause_order": _order_name(slots_of(spec).order, k)}
            texts = {side: fill(frame, [combo[n]["value"] for n in names[side]]) for side in ("a", "b")}
            fills = {side: [_fill_record(n, combo[n], sources[n]) for n in names[side]] for side in ("a", "b")}
            pair_id = sha256_text("\n".join([spec_sha256, frame, texts["a"], texts["b"]]))[:16]
            yield Pair(pair_id=pair_id, a=texts["a"], b=texts["b"], frame=frame, factors=levels,
                       fills=fills,
                       contrast=spec.contrast, spec_sha256=spec_sha256, provenance=prov)
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def frames_from_pairs(path: str | Path) -> list[str]:
    """The distinct frames of a pair file, in order of first appearance."""
    seen: dict[str, None] = {}
    for pair in jsonl.read(path, PairView):
        if pair.frame:
            seen.setdefault(pair.frame, None)
    if not seen:
        raise ValueError(f"{path}: no pair has a frame")
    return list(seen)


def slots_of(spec: Spec) -> Slots:
    if spec.slots is None:
        raise ValueError("this spec has no `slots`: it can configure `check`, but `generate` needs slots")
    return spec.slots


def load_pools(spec: Spec, spec_path: str | Path, graded_path: str | None) -> dict[str, list[Candidate]]:
    slots_of(spec)
    graded = None
    if graded_path is not None:
        graded = graded_candidates(list(jsonl.read(graded_path, GradedView)))
    elif slots_of(spec).needs_graded():
        raise ValueError("this spec selects slots from a graded file: pass --graded")
    spec_dir = Path(spec_path).parent
    return {name: slot_candidates(src, graded, spec_dir, spec.slot_defaults) for name, src in slots_of(spec).select.items()}


def run(adapter: Adapter | None, spec_path: str | Path, graded_path: str | None, *, n_frames: int = 8,
        limit: int | None = None, seed: int = 0,
        frames_from: str | None = None) -> tuple[FrameResult, Iterator[Pair]]:
    """Get frames (hand-written in the spec, reused from a pair file, or from the
    generator model), then return them with a lazy iterator of pairs."""
    spec, spec_sha = load_spec(spec_path)
    inputs = {"spec": spec_sha}
    if graded_path is not None:
        inputs["graded"] = hashlib.sha256(Path(graded_path).read_bytes()).hexdigest()
    pools = load_pools(spec, spec_path, graded_path)
    width = len(slots_of(spec).resolved_sides()["a"])
    if spec.frames is not None or frames_from is not None:
        frames = spec.frames if spec.frames is not None else frames_from_pairs(frames_from or "")
        problems = [(f, frame_problem(f, width, spec, set())) for f in frames]
        bad = [(f, why) for f, why in problems if why]
        if bad:
            raise ValueError(f"unusable frames: {bad}")
        result = FrameResult(list(frames), [], "", 0)
        source = "spec.frames" if spec.frames is not None else "frames-from"
        template = {"name": source, "sha256": sha256_text("\n".join(frames))}
        model = None
        if frames_from is not None:
            inputs["frames_from"] = hashlib.sha256(Path(frames_from).read_bytes()).hexdigest()
    else:
        if adapter is None:
            raise ValueError("this spec has no `frames`: pass a generator model (--adapter/--model) "
                             "or reuse frames with --frames-from")
        result = make_frames(adapter, spec, pools, n_frames)
        template = {"name": FRAME_PROMPT_VERSION, "sha256": sha256_text(FRAME_PROMPT + CLAUSE_RULE)}
        model = adapter.describe()
    if slots_of(spec).factors:
        try:
            result.cells = cell_sizes(pools, spec)
        except ValueError:  # too many combinations to count; sampling still works
            pass
    config: dict[str, Any] = {"pool_sizes": {name: len(pool) for name, pool in pools.items()}}
    if result.prompt:
        config["frame_prompt_sha256"] = sha256_text(result.prompt)
    prov = Provenance(
        verb="generate", model=model, template=template,
        sampling={"frames": len(result.frames), "frames_rejected": len(result.rejected),
                  "generator_calls": result.calls},
        seed=seed, inputs=inputs, config=config,
    )
    rng = random.Random(seed)
    per_combo = len(clause_orders(width, slots_of(spec).order))
    combo_limit = None if limit is None else -(-limit // per_combo)
    combo_iter = combos(pools, slots_of(spec).same, rng, combo_limit, slots_of(spec).factors)
    return result, pairs(spec, spec_sha, result.frames, combo_iter, prov, limit)
