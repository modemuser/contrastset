"""Model-calling checks (behaviour, blind_judge), with the fake adapter only.
Deliberately separate from test_checks.py: these go through adapters."""

import json
from pathlib import Path

import pytest

from contrastset import cli
from contrastset.adapters import FakeAdapter
from contrastset.checks import model
from contrastset.records import PairView


def graded_file(tmp_path: Path) -> Path:
    from contrastset.grade import Template

    sha = Template.load("qwen3-nothink").sha256
    rows = [{"entity": town, "questions": [{"id": "river", "bin": b, "answers": [river], "correct_rate": rate}],
             "provenance": {"verb": "grade", "template": {"name": "builtin:qwen3-nothink", "sha256": sha}}}
            for town, river, b, rate in [("Bonn", "Rhine", "known", 1.0), ("Ulm", "Danube", "known", 1.0),
                                         ("Tábor", "Lužnice", "unstable", 0.0)]]
    path = tmp_path / "graded.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return path


def pair_file(tmp_path: Path) -> Path:
    def fills(towns: list[str]) -> list[dict]:  # type: ignore[type-arg]
        return [{"value": t, "question": "river"} for t in towns]
    rows = [{"a": "Which river flows through Bonn, and which through Ulm?",
             "b": "Which river flows through Bonn, and which through Tábor?",
             "fills": {"a": fills(["Bonn", "Ulm"]), "b": fills(["Bonn", "Tábor"])}},
            {"a": "No fills here.", "b": "None at all."}]
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return path


def test_behaviour_check_asks_the_subject_and_skips_pairs_without_fills(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    # The subject answers Bonn and Ulm, and (wrongly for the design) also knows Tábor in context.
    subject = FakeAdapter(lambda prompt, t, s: "Rhine\nDanube" if "Ulm" in prompt else "Rhine\nLužnice")
    monkeypatch.setattr(cli, "make_adapter", lambda *a, **k: subject)
    code = cli.main(["check", str(pair_file(tmp_path)), "--checks", "behaviour", "--graded", str(graded_file(tmp_path)),
                     "--subject-model", "fake", "--template", "qwen3-nothink", "--require", "behaviour"])
    out, err = capsys.readouterr()
    flags = [json.loads(line) for line in out.splitlines()]
    assert flags[0]["verdict"] == "fail" and "not-known clause 'Tábor' answered 1.00" in flags[0]["verdicts"][0]["reason"]
    assert flags[1]["skipped"] == {"behaviour": "pair has no fills"}
    assert code == 1 and "required check behaviour was skipped on 1 pairs" in err
    seeds = {c["seed"] for c in subject.calls if c["seed"] is not None}
    assert min(seeds) >= 10_000  # never the grade's seeds


def test_blind_judge_reports_accuracy_and_fails_a_leaky_set(monkeypatch: pytest.MonkeyPatch) -> None:
    pairs = [PairView.model_validate({"a": f"plain text {i}", "b": f"INTENT text {i}"}) for i in range(20)]
    # A judge that always finds the side containing INTENT: a surface leak.
    judge = FakeAdapter(lambda prompt, t, s: "1" if "1: INTENT" in prompt else "2")
    monkeypatch.setattr(model, "make_adapter", lambda *a, **k: judge)
    check = model.BlindJudge()
    verdicts = [check(p, None, target="states an intention") for p in pairs]
    assert all(v is not None and v.severity == "warn" for v in verdicts)
    lines, failed = check.summary()
    assert failed and "20/20" in lines[0] and "LEAK" in lines[0]
    # A judge at chance does not fail the set.
    chance = model.BlindJudge()
    monkeypatch.setattr(model, "make_adapter", lambda *a, **k: FakeAdapter(lambda p, t, s: "1"))
    for p in pairs:
        chance(p, None, target="states an intention")
    assert chance.summary()[1] is False


def test_behaviour_check_refuses_a_different_template(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    graded = tmp_path / "graded.jsonl"
    graded.write_text(json.dumps({"entity": "Bonn", "questions": [{"id": "river", "bin": "known", "answers": ["Rhine"]}],
                                  "provenance": {"verb": "grade", "template": {"name": "t", "sha256": "0" * 64}}}) + "\n")
    code = cli.main(["check", str(pair_file(tmp_path)), "--checks", "behaviour", "--graded", str(graded),
                     "--subject-model", "fake", "--template", "qwen3-nothink"])
    assert code == 2 and "grade and probe with the same string" in capsys.readouterr().err
