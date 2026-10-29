"""The JSONL contract: data records are read from a path or stdin, written to stdout,
and fields a verb doesn't know pass through untouched."""

import io
import json

import pytest

from contrastset import cli
from contrastset.adapters import FakeAdapter

EXTRA = {"my_label": 3, "my_note": {"nested": ["kept", "as is"], "ü": None}}


def stdin(monkeypatch: pytest.MonkeyPatch, *records: dict) -> None:  # type: ignore[type-arg]
    monkeypatch.setattr("sys.stdin", io.StringIO("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)))


def town(name: str, **question_extra: object) -> dict:  # type: ignore[type-arg]
    return {"entity": name, "type": "town", **EXTRA,
            "questions": [{"id": "country", "q": f"Which country is {name} in?", "answers": ["Germany"],
                           **question_extra}]}


def grade_stdin(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *records: dict) -> list[dict]:  # type: ignore[type-arg]
    monkeypatch.setattr(cli, "make_adapter", lambda *a, **k: FakeAdapter({"Düren": ["Germany"]}))
    stdin(monkeypatch, *records)
    assert cli.main(["grade", "-", "--model", "fake", "--template", "qwen3-nothink", "-n", "2"]) == 0
    captured = capsys.readouterr()
    assert "passing through fields Entity doesn't use: my_label, my_note" in captured.err
    return [json.loads(line) for line in captured.out.splitlines()]


def test_grade_passes_unknown_entity_fields_through(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    [out] = grade_stdin(monkeypatch, capsys, town("Düren"))
    assert {k: out[k] for k in EXTRA} == EXTRA


@pytest.mark.xfail(strict=True, reason="grade builds each QuestionGrade afresh and drops unknown question "
                                       "fields: github.com/modemuser/contrastset/issues/1")
def test_grade_passes_unknown_question_fields_through(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    [out] = grade_stdin(monkeypatch, capsys, town("Düren", my_q_label="x"))
    assert out["questions"][0]["my_q_label"] == "x"


def test_priors_and_candidates_pass_graded_fields_through(tmp_path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    graded = grade_stdin(monkeypatch, capsys, town("Düren"))
    graded[0]["questions"][0]["my_q_label"] = "x"  # added downstream by another tool
    path = tmp_path / "graded.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in graded))

    assert cli.main(["grade", "--priors", str(path)]) == 0
    [marked] = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert {k: marked[k] for k in EXTRA} == EXTRA and marked["questions"][0]["my_q_label"] == "x"

    assert cli.main(["generate", "--candidates", "--graded", str(path)]) == 0
    [candidate] = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert {k: candidate[k] for k in EXTRA} == EXTRA and candidate["my_q_label"] == "x"


def test_check_reads_any_pair_lines_from_stdin(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    stdin(monkeypatch, {"a": "The lamp is here.", "b": "The sofa is here.", **EXTRA})
    assert cli.main(["check", "-", "--checks", "length,slot_diff"]) == 0
    captured = capsys.readouterr()
    [flag] = [json.loads(line) for line in captured.out.splitlines()]
    assert flag["record"] == "contrastset.check/1" and flag["verdict"] == "ok"
    assert "passing through fields PairView doesn't use: my_label, my_note" in captured.err
