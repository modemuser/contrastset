"""The check protocol and runner: does a pair differ only in the contrast?

A check is any callable

    check(pair, spec, /, **options) -> Verdict | None

returning None when the pair passes. `pair` is a records.PairView: only `a` and `b`
are guaranteed, so `check` works on pair files from anywhere. `spec` is None when no
spec was given. Optional attributes refine it:

    id        name in reports (default: the name it was referenced by)
    requires  what it needs beyond a and b, any of:
                  "frame", "fills"             pair fields
                  "spec.examples"              a spec with example pairs
                  "graded", "subject"          resources from the command line
              Where one is missing the check is SKIPPED for that pair, with the
              reason recorded. A skip is not a pass: the summary counts skips, and
              `check --require ID` turns a skip of ID into a failure.
    fit(pairs, spec)       called once with every pair first (set-level checks)
    summary() -> (lines, failed)
                           set-level report after the per-pair calls; failed=True
                           fails the run (e.g. a blind judge beating chance)

Resources a check requires arrive as keyword options: `graded` (dict
(entity, question id) -> {"answers": [...], "correct_rate": ...}) and `subject`
(an Adapter) plus `template` (grade.Template).

A spec activates checks by name, built-in or dotted path to your own:

    checks:
      - use: length
        severity: fail          # overrides the severity the check reports
        options: {fail_words: 2}
      - use: mychecks.no_intent_words
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from contrastset.records import CheckRef, PairView, Spec, Verdict


@runtime_checkable
class Check(Protocol):
    def __call__(self, pair: PairView, spec: Spec | None, /, **options: Any) -> Verdict | None: ...


def builtins() -> dict[str, Check]:
    """Fresh instances of every built-in check (set-level checks keep state)."""
    from contrastset.checks import model, surface

    return {name: factory() for name, factory in {**surface.FACTORIES, **model.FACTORIES}.items()}


DEFAULT_CHECKS = ["length", "overlap", "slot_diff", "duplicates", "example_similarity", "example_reuse"]


def resolve(ref: CheckRef) -> Check:
    """Find the check a spec entry names: a built-in name, or "module.attribute"."""
    known = builtins()
    if ref.use in known:
        return known[ref.use]
    module_name, _, attribute = ref.use.rpartition(".")
    if not module_name:
        raise ValueError(f"unknown check {ref.use!r}: not a built-in ({', '.join(known)}) and not a dotted path")
    obj = getattr(importlib.import_module(module_name), attribute)
    if not callable(obj):
        raise TypeError(f"check {ref.use!r} is not callable")
    return obj  # type: ignore[no-any-return]


def check_id(check: Check, ref: CheckRef) -> str:
    return str(getattr(check, "id", ref.use))


@dataclass
class Resources:
    graded: dict[tuple[str, str], dict[str, Any]] | None = None
    subject: Any = None  # Adapter
    template: Any = None  # grade.Template


def missing(check: Check, pair: PairView, spec: Spec | None, res: Resources) -> str | None:
    """Why `check` cannot run on `pair`, or None if it can."""
    for need in getattr(check, "requires", ()):
        if need == "frame" and not pair.frame:
            return "pair has no frame"
        if need == "fills" and not (pair.fills.get("a") and pair.fills.get("b")):
            return "pair has no fills"
        if need == "spec.examples" and not (spec and spec.examples):
            return "no spec with examples (--spec)"
        if need == "graded" and res.graded is None:
            return "no graded file (--graded)"
        if need == "subject" and res.subject is None:
            return "no subject model (--subject-model)"
    return None


@dataclass
class Active:
    check: Check
    ref: CheckRef
    id: str
    counts: dict[str, int] = field(default_factory=lambda: {"ok": 0, "warn": 0, "fail": 0, "skipped": 0})
    skip_reasons: dict[str, int] = field(default_factory=dict)


def run_one(active: Active, pair: PairView, spec: Spec | None, res: Resources) -> tuple[Verdict | None, str | None]:
    """(verdict, skip reason) for one check on one pair."""
    why = missing(active.check, pair, spec, res)
    if why:
        active.counts["skipped"] += 1
        active.skip_reasons[why] = active.skip_reasons.get(why, 0) + 1
        return None, why
    options = dict(active.ref.options)
    for need in getattr(active.check, "requires", ()):
        if need in ("graded", "subject"):
            options[need] = getattr(res, need)
            if need == "subject":
                options["template"] = res.template
    verdict = active.check(pair, spec, **options)
    if verdict is not None:
        updates: dict[str, Any] = {"check_id": active.id}
        if active.ref.severity is not None:
            updates["severity"] = active.ref.severity
        verdict = verdict.model_copy(update=updates)
        active.counts[verdict.severity] += 1
    else:
        active.counts["ok"] += 1
    return verdict, None


def prepare(refs: Sequence[CheckRef], pairs: Sequence[PairView], spec: Spec | None) -> list[Active]:
    actives = []
    for ref in refs:
        check = resolve(ref)
        fit = getattr(check, "fit", None)
        if fit is not None:
            fit(pairs, spec)
        actives.append(Active(check, ref, check_id(check, ref)))
    return actives
