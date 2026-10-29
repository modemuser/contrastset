"""Regrade tier (opt-in, slow, `pytest -m regrade`): run it when the grading path changes.

Regrades the 29 towns of examples/03-two-clause-rivers with HF Qwen3-8B, template B
(name only), from log-probabilities alone, and compares with the frozen file
tests/regrade/graded-29-B.jsonl. A log-prob mass is a lower bound: the true value is
in [p, p + pruned_max] (HFAdapter.prefix_mass drops paths below 1e-5), so two grades
may differ by at most the larger of their pruned masses. Bins and modal answers must
match exactly. About 27 minutes on an M-series Mac.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contrastset import cli

pytestmark = pytest.mark.regrade

ROOT = Path(__file__).parent.parent
FROZEN = Path(__file__).parent / "regrade" / "graded-29-B.jsonl"
EPS = 1e-6  # float noise on top of the pruned-mass bound


def masses_agree(new: dict, old: dict, where: str) -> None:  # type: ignore[type-arg]
    assert new["bin"] == old["bin"] and new["modal_answer"] == old["modal_answer"], where
    for p, bound in (("p_correct", "p_correct_pruned_max"), ("p_modal", "p_modal_pruned_max")):
        slack = max(new[bound] or 0.0, old[bound] or 0.0) + EPS
        assert abs(new[p] - old[p]) <= slack, f"{where} {p}: {new[p]} vs frozen {old[p]} (bound {slack:.2e})"


def test_regrade_matches_frozen_logprob_grade(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    out = tmp_path / "graded.jsonl"
    code = cli.main([
        "grade", str(ROOT / "examples/03-two-clause-rivers/entities.jsonl"), "--adapter", "hf", "--model", "Qwen/Qwen3-8B",
        "--template", str(ROOT / "experiments/templates/qwen3-name-only.txt"), "-n", "0", "--logprob",
        "--temperature", "0.7", "--top-k", "20", "--top-p", "0.95", "--gate", "country", "-o", str(out)])
    assert code == 0
    new = [json.loads(line) for line in out.read_text().splitlines()]
    old = [json.loads(line) for line in FROZEN.read_text().splitlines()]
    assert [r["entity"] for r in new] == [r["entity"] for r in old]
    for n, o in zip(new, old):
        assert n["bin"] == o["bin"], n["entity"]
        assert [q["id"] for q in n["questions"]] == [q["id"] for q in o["questions"]], n["entity"]
        for qn, qo in zip(n["questions"], o["questions"]):
            assert qn["bin"] == qo["bin"], f"{n['entity']}/{qn['id']}"
            for block in ("logprob", "logprob_sampler"):
                masses_agree(qn[block], qo[block], f"{n['entity']}/{qn['id']} {block}")
