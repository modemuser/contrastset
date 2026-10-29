"""A user check for example 04, referenced from spec.yaml by dotted path
(`intent_checks.no_intent_words`). `check` puts the spec's folder on the import
path, so this module needs no installing.

The contrast must be carried by the scenario, not by a marker word: if one side
says "deliberately" or "by accident", a probe can learn the word instead of the
intention.
"""

import re

from contrastset.records import PairView, Spec, Verdict

DEFAULT_WORDS = ["deliberately", "intentionally", "on purpose", "by accident", "accidentally", "meant to",
                 "unintentionally", "purposely"]


def no_intent_words(pair: PairView, spec: Spec | None, /, *, words: list[str] = DEFAULT_WORDS,
                    **_: object) -> Verdict | None:
    found = {side: [w for w in words if re.search(rf"\b{re.escape(w)}\b", getattr(pair, side), re.I)]
             for side in ("a", "b")}
    if found["a"] or found["b"]:
        return Verdict(check_id="no_intent_words", severity="fail",
                       reason=f"explicit intent words: a {found['a']}, b {found['b']}")
    return None
