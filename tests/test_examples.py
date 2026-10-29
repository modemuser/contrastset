"""The example ladder, run against its committed files.

Every verb is run on committed inputs, never on another verb's live output, and its
result is compared with the committed output: byte for byte where the run is
deterministic (SOURCE_DATE_EPOCH fixes the timestamps). Exceptions, each named:
pairs made by an unseeded generator (Haiku) are compared without provenance, after
regenerating them from their own frames; graded files are verified by re-scoring
their stored samples, since a test can't rerun the subject model. Model-calling
checks are not run here (see test_model_checks.py).
"""

import io
import json
import shutil
from pathlib import Path

import pytest

from contrastset import cli, grade, jsonl
from contrastset.records import GradedEntity

EXAMPLES = Path(__file__).parent.parent / "examples"
EPOCH = "1793145600"  # 2026-10-28T00:00:00Z, the value the committed files were made with


@pytest.fixture(autouse=True)
def fixed_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", EPOCH)


def run(argv: list[str], tmp_path: Path, name: str) -> tuple[int, bytes]:
    out = tmp_path / name
    code = cli.main([*argv, "-o", str(out)])
    return code, out.read_bytes()


def committed(example: str, name: str) -> bytes:
    return (EXAMPLES / example / name).read_bytes()


def assert_rescores_identically(example: str, template: str) -> list[GradedEntity]:
    path = EXAMPLES / example / "graded.jsonl"
    phrases = grade.Template.load(str(EXAMPLES / example / template)).refusal_phrases
    assert phrases is not None
    records = list(jsonl.read(path, GradedEntity))
    buf = io.StringIO()
    for record in records:
        jsonl.write(grade.rescore(record, phrases), buf)
    assert buf.getvalue().encode() == path.read_bytes()
    return records


# --- 01 · lexical control: no model, no graded file ---------------------------------


def test_01_generate_and_check_reproduce_byte_for_byte(tmp_path: Path) -> None:
    ex = EXAMPLES / "01-lexical-control"
    code, pairs = run(["generate", str(ex / "spec.yaml")], tmp_path, "pairs.jsonl")
    assert code == 0 and pairs == committed("01-lexical-control", "pairs.jsonl")
    code, report = run(["check", str(ex / "pairs.jsonl"), "--spec", str(ex / "spec.yaml")], tmp_path, "check.jsonl")
    assert code == 0 and report == committed("01-lexical-control", "check.jsonl")


# --- 02 · capitals: grade + select + frame -------------------------------------------


def test_02_grade_rescores_and_is_a_sane_smoke_test() -> None:
    records = assert_rescores_identically("02-capitals", "template.txt")
    # A 14B model knows nearly all capitals: a non-trivial unstable count would mean
    # the matcher is broken, not the model.
    unstable = [r.entity for r in records if r.questions[0].bin.value == "unstable"]
    assert len(unstable) <= 2, unstable


def test_02_generate_and_check_reproduce_byte_for_byte(tmp_path: Path) -> None:
    ex = EXAMPLES / "02-capitals"
    code, pairs = run(["generate", str(ex / "spec.yaml"), "--graded", str(ex / "graded.jsonl")], tmp_path, "pairs.jsonl")
    assert code == 0 and pairs == committed("02-capitals", "pairs.jsonl")
    code, report = run(["check", str(ex / "pairs.jsonl"), "--spec", str(ex / "spec.yaml")], tmp_path, "check.jsonl")
    assert report == committed("02-capitals", "check.jsonl")


# --- 03 · two-clause rivers: sides, familiar, order, behavioural check ---------------


def test_03_grade_rescores_identically() -> None:
    assert_rescores_identically("03-two-clause-rivers", "template.txt")


def test_03_pairs_regenerate_from_their_own_frames(tmp_path: Path) -> None:
    """The committed pairs came from Haiku (unseeded; provenance records it). Given
    their frames back, generate must produce the same pairs."""
    ex = EXAMPLES / "03-two-clause-rivers"
    code, pairs = run(["generate", str(ex / "spec.yaml"), "--graded", str(ex / "graded.jsonl"),
                       "--frames-from", str(ex / "pairs.jsonl"), "--limit", "60"], tmp_path, "pairs.jsonl")
    assert code == 0

    def content(raw: bytes) -> list[dict]:  # type: ignore[type-arg]
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        return [{k: v for k, v in r.items() if k != "provenance"} for r in rows]

    assert content(pairs) == content(committed("03-two-clause-rivers", "pairs.jsonl"))
    original = json.loads(committed("03-two-clause-rivers", "pairs.jsonl").decode().splitlines()[0])["provenance"]
    assert original["model"]["model"] == "claude-haiku-5-5" and original["model"]["seed_supported"] is False


def test_03_surface_check_reproduces_byte_for_byte(tmp_path: Path) -> None:
    ex = EXAMPLES / "03-two-clause-rivers"
    code, report = run(["check", str(ex / "pairs.jsonl"), "--spec", str(ex / "spec.yaml"), "--checks",
                        "length,overlap,slot_diff,duplicates,example_similarity,example_reuse"], tmp_path, "check.jsonl")
    assert code == 0 and report == committed("03-two-clause-rivers", "check.jsonl")


def test_03_candidates_are_a_valid_slot_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--candidates output, filtered as jq would, feeds back in as a `file` source."""
    ex = EXAMPLES / "03-two-clause-rivers"
    code, raw = run(["generate", "--candidates", "--graded", str(ex / "graded.jsonl")], tmp_path, "cands.jsonl")
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    assert code == 0 and len(rows) == 29 * 3
    # jq -c 'select(.question == "river" and .correct_rate >= 0.875)'
    known = [r for r in rows if r["question"] == "river" and r["correct_rate"] >= 0.875]
    (tmp_path / "known.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in known) + "\n")
    spec = tmp_path / "spec.yaml"
    spec.write_text("""
contrast: which known town
frames: ["Which river flows through [[X1]]?"]
slots:
  select:
    a: {source: file, path: known.jsonl}
    b: {source: file, path: known.jsonl}
""")
    code, pairs = run(["generate", str(spec), "--limit", "3"], tmp_path, "pairs.jsonl")
    fills = [json.loads(line)["fills"]["a"][0] for line in pairs.decode().splitlines()]
    assert code == 0 and len(fills) == 3
    assert all(f["source"] == "file" and f["question"] == "river" and f["bin"] == "known" for f in fills)


# --- 04 · intent pairs: hand-written pairs, user check, blind judge ------------------


def test_04_surface_and_user_checks_reproduce_byte_for_byte(tmp_path: Path) -> None:
    ex = EXAMPLES / "04-intent-pairs"
    code, report = run(["check", str(ex / "pairs.jsonl"), "--spec", str(ex / "spec.yaml"), "--checks",
                        "length,overlap,slot_diff,intent_checks.no_intent_words"], tmp_path, "check.jsonl")
    assert code == 1  # wall-04 says "deliberately": the user check fails it, on purpose
    assert report == committed("04-intent-pairs", "check.jsonl")


def test_04_hand_written_pairs_keep_their_own_fields(tmp_path: Path) -> None:
    shutil.copy(EXAMPLES / "04-intent-pairs" / "pairs.jsonl", tmp_path / "p.jsonl")
    lines = [json.loads(line) for line in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert all("scenario" in line and "pair_id" not in line for line in lines)
    flags = [json.loads(line) for line in committed("04-intent-pairs", "check.jsonl").decode().splitlines()]
    assert len({f["pair_id"] for f in flags}) == len(lines)  # derived ids still join back
