"""The check verb and the surface checks. No model anywhere in this file; the
model-calling checks are tested in test_model_checks.py with the fake adapter."""

import json
import sys
import types
from pathlib import Path

import pytest

from contrastset import cli
from contrastset.checks import surface
from contrastset.records import PairView, Verdict


def write_pairs(tmp_path: Path, rows: list[dict]) -> Path:  # type: ignore[type-arg]
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def run(argv: list[str], capsys) -> tuple[int, list[dict], str]:  # type: ignore[no-untyped-def, type-arg]
    code = cli.main(argv)
    out, err = capsys.readouterr()
    return code, [json.loads(line) for line in out.splitlines()], err


def test_pair_view_needs_only_a_and_b_and_passes_the_rest_through() -> None:
    p = PairView.model_validate({"a": "The cat sat.", "b": "The dog sat.", "my_label": 3})
    assert len(p.pair_id) == 16 and p.model_extra == {"my_label": 3}
    with pytest.raises(ValueError, match="graded"):
        PairView.model_validate({"record": "contrastset.graded/1", "a": "x", "b": "y"})


def test_foreign_pair_file_runs_surface_checks_and_counts_skips(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    pairs = write_pairs(tmp_path, [{"a": "The cat sat on the mat.", "b": "The dog sat on the mat."},
                                   {"a": "Short.", "b": "A much longer sentence with many more words in it."}])
    code, flags, err = run(["check", str(pairs)], capsys)
    assert code == 1  # the second pair fails length and overlap
    assert [f["verdict"] for f in flags] == ["ok", "fail"]
    assert flags[0]["skipped"] == {"example_similarity": "no spec with examples (--spec)",
                                   "example_reuse": "no spec with examples (--spec)"}
    assert "skipped example_similarity on 2 pairs" in err and "FAIL" in err
    code, flags, err = run(["check", str(pairs), "--checks", "length", "--max-fail-rate", "0.5"], capsys)
    assert code == 0 and "PASS" in err


def test_require_turns_a_skip_into_a_failure(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    pairs = write_pairs(tmp_path, [{"a": "The cat sat.", "b": "The dog sat."}])
    code, _, err = run(["check", str(pairs), "--checks", "length,example_reuse"], capsys)
    assert code == 0  # a skip alone does not fail ...
    code, _, err = run(["check", str(pairs), "--checks", "length,example_reuse", "--require", "example_reuse"], capsys)
    assert code == 1 and "required check example_reuse was skipped on 1 pairs" in err  # ... unless required


def test_slot_diff_uses_fills_when_present() -> None:
    check = surface.SlotDiff()
    clean = PairView.model_validate({"a": "Which river flows through Bonn?", "b": "Which river flows through Ulm?",
                                     "fills": {"a": [{"value": "Bonn"}], "b": [{"value": "Ulm"}]}})
    assert check(clean, None) is None
    leaky = PairView.model_validate({"a": "Which river flows through Bonn?", "b": "What river runs past Ulm?",
                                     "fills": {"a": [{"value": "Bonn"}], "b": [{"value": "Ulm"}]}})
    verdict = check(leaky, None)
    assert verdict is not None and verdict.severity == "fail" and "outside the slots" in verdict.reason


def test_duplicates_is_set_level() -> None:
    rows = [PairView.model_validate(r) for r in [{"a": "x one", "b": "y one"}, {"a": "x one", "b": "y two"},
                                                 {"a": "z", "b": "z"}]]
    check = surface.Duplicates()
    check.fit(rows, None)
    assert check(rows[0], None).reason == "reused text: side a in 1 other pair(s)"  # type: ignore[union-attr]
    assert check(rows[2], None).severity == "fail"  # type: ignore[union-attr]


def no_cat(pair: PairView, spec: object, /, *, word: str = "cat", **_: object) -> Verdict | None:
    if word in pair.a or word in pair.b:
        return Verdict(check_id="no_cat", severity="warn", reason=f"mentions {word}")
    return None


def test_user_check_by_dotted_path_with_options_and_severity(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    module = types.ModuleType("user_checks")
    module.no_cat = no_cat  # type: ignore[attr-defined]
    sys.modules["user_checks"] = module
    pairs = write_pairs(tmp_path, [{"a": "The cat sat.", "b": "The dog sat."}])
    spec = tmp_path / "spec.yaml"
    spec.write_text("""
contrast: c
frames: ["[[X1]]"]
slots: {select: {a: {source: inline, values: [x]}, b: {source: inline, values: [y]}}}
checks:
  - {use: user_checks.no_cat, severity: fail, options: {word: dog}}
""")
    code, flags, err = run(["check", str(pairs), "--spec", str(spec)], capsys)
    assert code == 1 and flags[0]["verdicts"][0] == {"check_id": "user_checks.no_cat", "severity": "fail",
                                                     "reason": "mentions dog", "metrics": {}}
