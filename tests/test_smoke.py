"""Smoke tier (opt-in, `pytest -m smoke`): Marks & Tegmark step 0 end to end on Qwen3-8B.

Needs the hf extra, the Qwen/Qwen3-8B weights (downloaded on first use) and network
access for the statement sets. About 10 minutes on an M-series Mac. See
examples/00-marks-tegmark/README.md.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

EX = Path(__file__).parent.parent / "examples" / "00-marks-tegmark"
MIN_AUC, FROM_LAYER = 0.95, 20


def run(*args: str, stdin: str | None = None) -> str:
    done = subprocess.run([sys.executable, *args], input=stdin, capture_output=True, text=True, cwd=EX)
    sys.stderr.write(done.stderr[-2000:])
    if done.returncode not in (0, 1):  # check exits 1 on flagged pairs; the test reads the verdicts itself
        raise AssertionError(f"{args} exited {done.returncode}")
    return done.stdout


def test_marks_tegmark_step0(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    run("fetch_data.py")

    pairs = run("make_pairs.py", "data/cities.csv")
    report = run("-m", "contrastset.cli", "check", "-", "--checks", "slot_diff,duplicates", stdin=pairs)
    flags = [json.loads(line) for line in report.splitlines()]
    assert len(flags) == 748 and all(f["verdict"] == "ok" for f in flags), "pairs must differ in the country slot only"

    expected = [json.loads(line) for line in (EX / "expected-auc.jsonl").read_text().splitlines()]
    for dataset in ["cities", "sp_en_trans"]:
        rows = [json.loads(line) for line in run("auc.py", dataset).splitlines()]
        (tmp_path / f"auc-{dataset}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        late = {r["layer"]: r["heldout_auc"] for r in rows if r["layer"] >= FROM_LAYER}
        assert len(late) == 37 - FROM_LAYER
        low = {layer: a for layer, a in late.items() if a < MIN_AUC}
        assert not low, f"{dataset}: held-out AUC below {MIN_AUC} at layers {low}"
        ref = {r["layer"]: r["heldout_auc"] for r in expected if r["dataset"] == dataset}
        drift = max(abs(r["heldout_auc"] - ref[r["layer"]]) for r in rows if r["heldout_auc"] is not None)
        print(f"{dataset}: min AUC layers {FROM_LAYER}+ {min(late.values()):.4f}, "
              f"max |AUC - expected-auc.jsonl| {drift:.2e}", file=sys.stderr)
