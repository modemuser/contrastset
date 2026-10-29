"""Surface checks: no model, only the text. Cheap leak statistics for any pair file.

Words are runs of letters, digits, apostrophes and hyphens, a stand-in for model
tokens (no tokenizer is loaded); counts in characters are reported alongside.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from collections.abc import Sequence
from statistics import mean
from typing import Any

from contrastset.records import PairView, Spec, Verdict

WORD = re.compile(r"[\w'-]+")
PLACEHOLDER = re.compile(r"\[\[X\d+\]\]")


def words(text: str) -> list[str]:
    return WORD.findall(text)


def jaccard(x: set[str], y: set[str]) -> float:
    return len(x & y) / len(x | y) if x | y else 1.0


class Length:
    """Word and character counts per side. Warns when the sides differ by more than
    `warn_words` words, fails beyond `fail_words`. The summary prints the
    distribution of the word difference over the whole file."""

    id = "length"
    requires: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.diffs: Counter[int] = Counter()
        self.char_diffs: list[int] = []

    def fit(self, pairs: Sequence[PairView], spec: Spec | None) -> None:
        self.diffs = Counter(abs(len(words(p.a)) - len(words(p.b))) for p in pairs)
        self.char_diffs = [abs(len(p.a) - len(p.b)) for p in pairs]

    def __call__(self, pair: PairView, spec: Spec | None, /, *, warn_words: int = 1, fail_words: int = 3,
                 **_: Any) -> Verdict | None:
        wa, wb = len(words(pair.a)), len(words(pair.b))
        diff = abs(wa - wb)
        metrics = {"words_a": wa, "words_b": wb, "chars_a": len(pair.a), "chars_b": len(pair.b)}
        if diff > fail_words:
            return Verdict(check_id=self.id, severity="fail", reason=f"sides differ by {diff} words", metrics=metrics)
        if diff > warn_words:
            return Verdict(check_id=self.id, severity="warn", reason=f"sides differ by {diff} words", metrics=metrics)
        return None

    def summary(self) -> tuple[list[str], bool]:
        if not self.char_diffs:
            return [], False
        dist = ", ".join(f"{d}: {n}" for d, n in sorted(self.diffs.items()))
        return [f"length: word difference a-b over all pairs (difference: pairs) {dist}; "
                f"mean character difference {mean(self.char_diffs):.1f}"], False


class Overlap:
    """Vocabulary overlap of the two sides: the share of words they have in common,
    counted with repeats and case-folded, over the longer side. A one-word swap in a
    six-word sentence scores 5/6. Should be high apart from the slots."""

    id = "overlap"
    requires: tuple[str, ...] = ()

    def __call__(self, pair: PairView, spec: Spec | None, /, *, warn_below: float = 0.7, fail_below: float = 0.5,
                 **_: Any) -> Verdict | None:
        wa = Counter(w.casefold() for w in words(pair.a))
        wb = Counter(w.casefold() for w in words(pair.b))
        longer = max(sum(wa.values()), sum(wb.values()))
        score = sum((wa & wb).values()) / longer if longer else 1.0
        if score < fail_below:
            return Verdict(check_id=self.id, severity="fail", reason=f"word overlap {score:.2f}", metrics={"overlap": score})
        if score < warn_below:
            return Verdict(check_id=self.id, severity="warn", reason=f"word overlap {score:.2f}", metrics={"overlap": score})
        return None


class SlotDiff:
    """Position-aligned word diff. With fills, every differing word must belong to a
    slot value of its side (anything else fails). Without fills, more than
    `max_spans` separate differences warns: a contrast should sit in one place."""

    id = "slot_diff"
    requires: tuple[str, ...] = ()

    def __call__(self, pair: PairView, spec: Spec | None, /, *, max_spans: int = 1, **_: Any) -> Verdict | None:
        wa, wb = words(pair.a), words(pair.b)
        ops = [op for op in difflib.SequenceMatcher(a=wa, b=wb, autojunk=False).get_opcodes() if op[0] != "equal"]
        if pair.fills.get("a") and pair.fills.get("b"):
            slot_words = {side: {w for f in pair.fills[side] for w in words(f.value)} for side in ("a", "b")}
            outside_a = [w for _, i1, i2, _, _ in ops for w in wa[i1:i2] if w not in slot_words["a"]]
            outside_b = [w for _, _, _, j1, j2 in ops for w in wb[j1:j2] if w not in slot_words["b"]]
            if outside_a or outside_b:
                return Verdict(check_id=self.id, severity="fail",
                               reason=f"difference outside the slots: a {outside_a}, b {outside_b}",
                               metrics={"spans": len(ops)})
            return None
        if len(ops) > max_spans:
            return Verdict(check_id=self.id, severity="warn", reason=f"{len(ops)} separate differences",
                           metrics={"spans": len(ops)})
        return None


class Duplicates:
    """Near-duplicate detection across the file (set-level). Fails a pair whose two
    sides are identical or that appears more than once; warns when one of its sides
    is reused verbatim in other pairs (the probe then sees that text more often)."""

    id = "duplicates"
    requires: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.pairs: Counter[tuple[str, str]] = Counter()
        self.sides: Counter[str] = Counter()

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(w.casefold() for w in words(text))

    def fit(self, pairs: Sequence[PairView], spec: Spec | None) -> None:
        self.pairs = Counter((self._norm(p.a), self._norm(p.b)) for p in pairs)
        self.sides = Counter(self._norm(t) for p in pairs for t in (p.a, p.b))

    def __call__(self, pair: PairView, spec: Spec | None, /, **_: Any) -> Verdict | None:
        a, b = self._norm(pair.a), self._norm(pair.b)
        if a == b:
            return Verdict(check_id=self.id, severity="fail", reason="sides a and b are identical")
        if self.pairs[(a, b)] > 1:
            return Verdict(check_id=self.id, severity="fail", reason=f"pair appears {self.pairs[(a, b)]} times")
        reused = {side: self.sides[t] - 1 for side, t in (("a", a), ("b", b)) if self.sides[t] > 1}
        if reused:
            detail = ", ".join(f"side {s} in {n} other pair(s)" for s, n in reused.items())
            return Verdict(check_id=self.id, severity="warn", reason=f"reused text: {detail}",
                           metrics={f"reuse_{s}": n for s, n in reused.items()})
        return None


def example_names(spec: Spec) -> set[str]:
    """Capitalised words inside the example pairs (not sentence-initial): the examples'
    names. A heuristic that suits languages that capitalise names."""
    names = set()
    for e in spec.examples:
        for text in (e.a, e.b):
            ws = words(text)
            names |= {w for w in ws[1:] if w[:1].isupper()}
    return names


class ExampleSimilarity:
    """Similarity to the spec's example pairs: a generator copies example quirks,
    and that is a leak too. Compares the frame (or the texts) with the examples,
    names and placeholders removed; warns at `warn_above` or more."""

    id = "example_similarity"
    requires = ("spec.examples",)

    def __call__(self, pair: PairView, spec: Spec | None, /, *, warn_above: float = 0.9, **_: Any) -> Verdict | None:
        assert spec is not None
        names = example_names(spec)
        texts = [PLACEHOLDER.sub(" ", pair.frame)] if pair.frame else [pair.a, pair.b]
        own = [{w.casefold() for w in words(t) if w not in names} for t in texts]
        best = 0.0
        for e in spec.examples:
            for et in (e.a, e.b):
                ex = {w.casefold() for w in words(et) if w not in names}
                best = max(best, *(jaccard(o, ex) for o in own))
        if best >= warn_above:
            return Verdict(check_id=self.id, severity="warn", reason=f"near-copy of an example (similarity {best:.2f})",
                           metrics={"similarity": best})
        return None


class ExampleReuse:
    """Warns when a pair reuses a name from the spec's examples: examples are style
    guidance only, never slot values, and the subject may or may not know them."""

    id = "example_reuse"
    requires = ("spec.examples",)

    def __call__(self, pair: PairView, spec: Spec | None, /, **_: Any) -> Verdict | None:
        assert spec is not None
        reused = sorted(example_names(spec) & (set(words(pair.a)) | set(words(pair.b))))
        if reused:
            return Verdict(check_id=self.id, severity="warn", reason=f"reuses example names {reused}")
        return None


# Built-in surface checks by name; instantiated fresh per run (set-level state).
FACTORIES: dict[str, Any] = {"length": Length, "overlap": Overlap, "slot_diff": SlotDiff,
                             "duplicates": Duplicates, "example_similarity": ExampleSimilarity,
                             "example_reuse": ExampleReuse}
