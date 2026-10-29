"""Checks that call a model. Kept apart from the surface checks: they need
adapters, cost time or money, and are tested with the fake adapter only.

Both build their adapters through `make_adapter` in this module, the single point
tests replace.
"""

from __future__ import annotations

import math
import random
from typing import Any

from contrastset.adapters import make_adapter as _make_adapter
from contrastset.grade import Template, match_answer, question_seed
from contrastset.records import PairView, Spec, Verdict

make_adapter = _make_adapter  # replaced in tests; never called by the surface checks


class Behaviour:
    """Asks the SUBJECT model both sides of the pair and scores every clause
    separately: the composite probe as a check. Single-question knowledge does not
    always survive inside a composite question (FINDINGS.md), and surface statistics
    can't see that; this can.

    A fill whose graded correct_rate is at least `known_min` is expected to be
    answered (correct in at least `answered_min` of the completions); any other fill
    is expected not to be. Sampling seeds start from `seed` (default 10000) plus a hash
    of the pair and side, so pairs don't share random streams and the check never
    reuses a grade's samples.
    Clauses whose reference answers overlap (two towns on one river) can't be told
    apart by string match: such pairs get a warning, not a verdict on behaviour."""

    id = "behaviour"
    requires = ("fills", "graded", "subject")

    def __call__(self, pair: PairView, spec: Spec | None, /, *, graded: dict[tuple[str, str], dict[str, Any]],
                 subject: Any, template: Template, n: int = 4, seed: int = 10_000, temperature: float = 0.7,
                 known_min: float = 0.875, answered_min: float = 0.75, max_tokens: int = 64,
                 **_: Any) -> Verdict | None:
        problems: list[str] = []
        metrics: dict[str, float | int | str | None] = {}
        for side in ("a", "b"):
            fills = pair.fills[side]
            refs = []
            for f in fills:
                info = graded.get((f.value, f.question or ""))
                if info is None:
                    return Verdict(check_id=self.id, severity="warn",
                                   reason=f"{f.value!r} / {f.question!r} not in the graded file")
                refs.append(info)
            answers = [r["answers"] for r in refs]
            if any(set(map(str.casefold, x)) & set(map(str.casefold, y))
                   for i, x in enumerate(answers) for y in answers[i + 1:]):
                return Verdict(check_id=self.id, severity="warn", reason=f"side {side}: clauses share an answer")
            prompt = template.render(getattr(pair, side), "")
            side_seed = question_seed(seed, pair.pair_id, side)  # own random streams per pair and side
            completions = [*subject.complete(prompt, n=1, temperature=0.0, max_tokens=max_tokens),
                           *subject.complete(prompt, n=n, temperature=temperature, seed=side_seed,
                                             max_tokens=max_tokens)]
            for f, ref in zip(fills, refs):
                share = sum(match_answer(c, ref["answers"]) is not None for c in completions) / len(completions)
                metrics[f"{side}:{f.value}"] = round(share, 3)
                expected_known = (ref.get("correct_rate") or 0.0) >= known_min
                if expected_known and share < answered_min:
                    problems.append(f"side {side}: known clause {f.value!r} answered {share:.2f}")
                if not expected_known and share >= answered_min:
                    problems.append(f"side {side}: not-known clause {f.value!r} answered {share:.2f}")
        if problems:
            return Verdict(check_id=self.id, severity="fail", reason="; ".join(problems), metrics=metrics)
        return None


JUDGE_PROMPT = """\
Below are two short texts, 1 and 2. Exactly one of them {target}.
Judge from the texts alone. Answer with the single digit 1 or 2.

1: {first}
2: {second}"""


class BlindJudge:
    """Asks a JUDGE model (not the subject) which side of the pair is the target,
    from surface form alone, with the sides in seeded random order. If a judge can
    tell the sides apart well above chance, the set leaks: a probe could learn the
    surface cue instead of the contrast. Per pair: a warning when the judge picks
    the target side. Over the set: the run fails if accuracy beats chance at
    p < `alpha` (one-sided binomial test).

    Options: `target` (required: what the target side does, e.g. "describes an
    intentional action"), `target_side` ("b"), `adapter`, `model`, `seed`."""

    id = "blind_judge"
    requires: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.correct = 0
        self.total = 0
        self.alpha = 0.01
        self._adapters: dict[tuple[str, str], Any] = {}

    def __call__(self, pair: PairView, spec: Spec | None, /, *, target: str, target_side: str = "b",
                 adapter: str = "anthropic", model: str = "claude-haiku-5-5", seed: int = 0, alpha: float = 0.01,
                 **_: Any) -> Verdict | None:
        self.alpha = alpha
        key = (adapter, model)
        if key not in self._adapters:
            self._adapters[key] = make_adapter(adapter, model)
        swap = random.Random(f"{seed}:{pair.pair_id}").random() < 0.5
        first, second = (pair.b, pair.a) if swap else (pair.a, pair.b)
        reply = self._adapters[key].complete(JUDGE_PROMPT.format(target=target, first=first, second=second),
                                             n=1, max_tokens=8)[0].strip()
        picked_first = reply.startswith("1")
        picked = ("b" if swap else "a") if picked_first else ("a" if swap else "b")
        self.total += 1
        if picked == target_side:
            self.correct += 1
            return Verdict(check_id=self.id, severity="warn", reason="judge picked the target side",
                           metrics={"judge_reply": reply[:8]})
        return None

    def summary(self) -> tuple[list[str], bool]:
        if not self.total:
            return [], False
        # P(X >= correct) for X ~ Binomial(total, 0.5)
        p = sum(math.comb(self.total, k) for k in range(self.correct, self.total + 1)) / 2 ** self.total
        leak = p < self.alpha
        return [f"blind_judge: {self.correct}/{self.total} = {self.correct / self.total:.2f} correct, "
                f"one-sided p = {p:.3g} against chance{' -> LEAK' if leak else ''}"], leak


FACTORIES: dict[str, Any] = {"behaviour": Behaviour, "blind_judge": BlindJudge}
