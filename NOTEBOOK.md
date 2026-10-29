# Notebook

Chronological record of maintenance passes on contrastset: what was run, in what
order, what came out, including mistakes. Research results go to FINDINGS.md. Times
are CET, newest entries at the bottom.

## 2026-10-29 · Release pass to v0.1.0

Goal: contrastset public as v0.1.0, the Marks & Tegmark step 0 as an opt-in regression
test, no behaviour change. One commit per section.

**12:10 · State found.** No commits yet on `main`, no git remote. Fast tests: 70 passed,
1 skipped. PyPI name `contrastset` unclaimed.

**12:12 · Two gaps between the plan and the repos.**
1. *No 29-town template-B HF log-prob graded file existed.* `runs/parity-hf-logprob-29.jsonl`
   is template A; template B with HF log-prob existed only for 294 towns. Decision: make
   the baseline file before any change, with the exact flags of the 294-town B run on the
   29 towns of `examples/03-two-clause-rivers/entities.jsonl`. With `-n 0 --logprob` the
   grade draws no samples, so it is deterministic.
2. *There is no graded file for the Marks & Tegmark statements.* Step 0 in knowing-probe
   fed the labelled CSVs from geometry-of-truth straight to the activation reader; truth
   labels come from the dataset, nothing was graded or generated. So the smoke tier runs
   check → activations → per-layer AUC, without grade/generate, and the regrade tier
   regrades the 29-town template-B file from (1) instead.

**12:19 · Baseline timing probe** (2 towns, 2 pairs): grade ≈55 s/town incl. model
load, behaviour check ≈10 s/pair; full baseline ≈37 min. Commands:

    uv run contrastset grade examples/03-two-clause-rivers/entities.jsonl --adapter hf --model Qwen/Qwen3-8B \
      --template experiments/templates/qwen3-name-only.txt -n 0 --logprob --temperature 0.7 --top-k 20 --top-p 0.95 \
      --gate country -o graded-29-B.jsonl
    E=examples/03-two-clause-rivers
    uv run contrastset check $E/pairs.jsonl --spec $E/spec.yaml --checks behaviour --require behaviour \
      --graded $E/graded.jsonl --subject-model qwen3:14b --template $E/template.txt -o check-behaviour.jsonl

Hash rule: sha256 over the JSONL with `provenance.created` and `provenance.tool_version`
dropped (wall clock and version string), keys sorted. Raw file sha256 recorded alongside.

**12:24 · Snapshot commit `05d3bf8`.** The working tree as found (code, tests, examples
01–04, FINDINGS, PLAN, experiments, scripts, uv.lock), without `runs/`, so every release
section below is its own reviewable diff.

**12:25 · Hygiene scan.** No absolute paths, home dirs, machine names, tokens or keys in
tracked files. API keys are only read from the environment (`OPENAI_API_KEY`, the
Anthropic SDK's own lookup).

**12:26 · Example 00 data: not copied.** saprmarks/geometry-of-truth has no licence file,
so its CSVs are not redistributed. `fetch_data.py` downloads them at commit 5d1c630 and
checks sha256 (f2560a0c… cities, 9afa97c1… sp_en_trans). The frozen split is committed
as `split.json` (held-out cities / Spanish words, seed 20261029): 249 cities and 115
words, the same as knowing-probe's sidecars. `expected-auc.jsonl` = knowing-probe
`runs/step0/*-summary.json` per layer (commit 911d9e0 there). Cities' held-out AUC is
0.994–1.000 over layers 15–35 and 0.985 at layer 36.

**12:28 · `check` on the cities pairs** (748 pairs from `make_pairs.py`, checks
length,overlap,slot_diff,duplicates): slot_diff ok on all 748; length 38 warn, 9 fail
(country names up to five words longer); exit 1.

**12:29 · Pass-through test finds a gap.** `grade` keeps unknown fields on an entity line
but drops unknown fields inside `questions[]`: `grade_question` builds each
QuestionGrade afresh. `--priors` and `generate --candidates` keep both levels. Not fixed
(no behaviour change in this pass); the question-level test is `xfail(strict=True)`,
tracked as issue #1.

**12:31 · mypy on tests.** `src` passed strict; `tests` had 14 errors (Optional values
used unchecked, one annotation naming `torch.Tensor` through an importorskip module,
one untyped fixture, list invariance GradedEntity vs GradedView). Fixed with asserts,
annotations and one commented `type: ignore`; no test logic changed.

**12:36 · Release workflow.** `.github/workflows/publish.yml`: on a `v*` tag, build with
`uv build`, `twine check --strict`, check the tag equals the pyproject version, upload
with pypa/gh-action-pypi-publish through the GitHub environment `pypi` (trusted
publishing, no token stored). `.github/workflows/ci.yml`: fast tier only (ruff, mypy,
pytest) on Python 3.11 and 3.14.

**12:22–12:58 · Step 0 baseline, before any change to library code.** Grade 26 min 57 s,
behaviour check 8 min 16 s.

| file | normalised sha256 | raw sha256 |
|---|---|---|
| graded-29-B.jsonl | `f0da59f8e8b3c1a3c1a7275222400e159770edc71fc08029cc0f7def6f083d3e` | `a77d9d23…` |
| check-behaviour.jsonl | `7fe53294b46a37f4dd53e9e68f13d47c0fdac4b162a9b0c7875b89f4009ee76c` | `e5a372eb…` |

Environment: Python 3.14.7, numpy 2.5.4, torch 2.14.1, transformers 5.21.0.
Grade: question bins country 29 known; region 23 known / 6 unstable; river 8 known /
5 confidently_wrong / 16 unstable. Behaviour check: 52 ok, 8 fail, exit 1. The committed
`examples/03-two-clause-rivers/check-behaviour.jsonl` (2026-10-28) has 9 fail: pair
`cbd1f575701e0b5e` was fail there and is ok today, same model digest (Ollama 0.34.2; the
committed file predates recording the server version). So the committed example output
is not byte-reproducible today; it was not touched. Exit 1 is that example's normal state.
Determinism: the timing probe's first 2 towns and pairs match the full run's first lines
after normalisation.

**12:59–13:10 · Smoke before cleanup: PASS.** knowing-probe's own step 0 code
(`knowing_probe.acts`, `arrow`, `mt_probe.item_of`; contrastset at `05d3bf8`) on fresh
activations, all 37 layers. cities: min held-out AUC over layers 20–36 = 0.9854 (layer
36); sp_en_trans: 1.000. Max |AUC − knowing-probe `runs/step0/*-summary.json`| over
layers 1–36 = 0.0 for both. 0.27 s/statement, ~9 min of reading.

**13:11 · Section 1, repo hygiene (`95ba7da`).**
- `.gitignore`: activations (`*.npy`, `*.npz`), model caches, run outputs, notebook
  checkpoints, build output, ruff cache, `.DS_Store`, the example-00 data fetched from
  upstream.
- `LICENSE`: MIT, 2026 Thomas Misera.
- `pyproject.toml`: version 0.1.0, licence file, author, URLs, classifiers. Python floor
  3.11 (wheel installed and imported on 3.11.16 without torch; `--adapter hf` without
  the extra fails with "needs `pip install contrastset[hf]`"). Extras `hf` and
  `anthropic`; no `ollama` extra, because Ollama needs only httpx, which is core (the
  default adapter and the OpenAI adapter use it); `dev` is a uv dependency group.
- `__version__ = "0.1.0"`, surfaced by `contrastset --version`; `tests/test_version.py`
  ties it to the package metadata.
- The version bump changes `tool_version` in the committed model-free example outputs
  (01 pairs+check, 02 pairs+check, 03 check, 04 check), which the byte-for-byte example
  tests compare. Regenerated those six files with `SOURCE_DATE_EPOCH=1793145600`, after
  confirming the content is otherwise identical (dropping tool_version, and
  `inputs.pairs` where the pair file itself was regenerated). Model-made files (graded,
  check-behaviour, check-judge, the Haiku-made 03 pairs) keep their 0.1.0.dev0
  provenance: that is the version that made them.

**13:20 · Section 2, code cleanup, no behaviour change (`3c0cacc`).**
- Dead code: `generate.CANDIDATE_CORE` (defined, never used). No stale TODO/FIXME in
  `src/`; no experiment-only code paths (river/town names appear only in docstrings and
  comments as examples). `scripts/` are research scripts, left as run.
- `--help` against the code: `grade --template` and `entities` say "required unless
  --priors" (argparse can't require them, since `--priors` needs neither); `--language`
  shows its default; `--logprob` mentions the tokenization sum, the pruned-mass bound
  and `-n 0`; `generate --frames-from` no longer names a model; `check --checks` prints
  `checks.DEFAULT_CHECKS` instead of a copied list; `check --graded` and
  `--allow-template-mismatch` name the template-hash guard. Open: `check` has no
  `--limit`.
- Lint: ruff in the dev group, rules pinned to ruff's classic defaults (E4, E7, E9, F),
  so a newer ruff's wider default set does not change what "lint passes" means. One fix:
  `l` → `layer` in `HFAdapter.run` (E741). `scripts/` excluded. mypy over src and tests.
- `tests/test_passthrough.py`: the JSONL contract (stdin in, stdout out, unknown fields
  kept).

**13:24 · Section 3, Marks & Tegmark step 0 as `examples/00-marks-tegmark/` (`dab1434`).**
From knowing-probe (commit 911d9e0 there; its copy stays): `split.json`,
`expected-auc.jsonl` (its step 0 per-layer summaries, incl. shuffle control and
|cos(arrow, PC1)|), and `auc.py`, a trimmed copy of `mt_acts.py` + `mt_probe.py` +
`knowing_probe.acts/arrow`: same hook-based last-token reader, same split function,
same arrow/projection/AUC, no plots, JSONL out; it refuses to run if the recomputed
split differs from `split.json`. New: `fetch_data.py` and `make_pairs.py`. README cites
the paper, states the reproduced numbers and why the shuffled-label control does not
fall to chance (true/false is PC1 from layer ~9).

**13:30 · Section 4, test tiers (`4cbe3bc`).**
- fast (default): `addopts = -m 'not smoke and not regrade'`; 74 passed, 1 skipped,
  1 xfailed. On Python 3.11 without torch: 66 passed, 2 skipped (test_hf skips as a
  module), 1 xfailed.
- smoke (`-m smoke`): example 00 end to end: fetch, `check` the 748 cities pairs
  (`slot_diff,duplicates` ok on every pair), `auc.py` for both sets, assert held-out AUC
  ≥ 0.95 at layers 20–36, print drift from `expected-auc.jsonl`.
- regrade (`-m regrade`): regrades the 29 towns, template B, `-n 0 --logprob`, against
  `tests/regrade/graded-29-B.jsonl` = the step 0 baseline grade (raw sha256 a77d9d23…,
  made at `05d3bf8`). Bins and modal answers exact; `p_correct`/`p_modal` (raw and
  sampler) within the larger pruned mass + 1e-6. Comparator exercised offline: the
  frozen file against itself passes; one p_correct shifted by 0.5× its pruned-mass bound
  passes, by 2× fails; a changed modal answer fails.
- CI: fast tier only, `uv sync --locked` without extras. mypy marks the optional extras'
  modules (torch, transformers, numpy, huggingface_hub, anthropic)
  `ignore_missing_imports`, so it passes without them and still checks against them
  where installed.

**13:36 · Section 5, README (`48e84bd`).** Status line; what a contrast set is and why
matched pairs; the ladder 00–04; the JSONL contract incl. the question-level gap;
log-prob grading (tokenization sum, pruned-mass bound); the template-hash guard;
backends and the rule (Ollama when samples suffice, HF log-prob for anything feeding a
probe); Testing (measured times: fast ~9 s, smoke ~10 min, regrade ~27 min); Lessons
(tokenization undercount, refusal-instruction effect) linking to FINDINGS.md; licence.

**13:40 · Section 6, release workflow (`8306077`).** `src/` is byte-identical from
`3c0cacc` (section 2) on, so one behaviour check on this commit covers sections 2–6.

**13:50–14:37 · Last-digit drift in `p_refusal` across numpy builds.** Behaviour checks
on section 1 (`95ba7da`) and on `8306077`, run in fresh worktree environments, gave
grade hash `91535894…` instead of `f0da59f8…`; check-behaviour matched exactly. The
difference: 78 values, all `questions[].logprob.p_refusal` and `refusal_rate` (which is
set to it), in the last digit (max relative 6.3e-16). Every bin, modal answer,
p_correct, p_modal and pruned-mass bound bit-identical; the two worktree runs identical
to each other. Cause: the worktree environments came up on Python 3.11.16, where
`uv.lock` pins numpy 2.4.6 (2.5.4 from Python 3.12 on); the baseline ran on 3.14.7 with
numpy 2.5.4. p_refusal is the one mass scored as a padded batch
(`HFAdapter.score_many`). Confirmed with identical code: the pre-release `05d3bf8` on
the 3.11 environment reproduces the section-1 values exactly (first 5 towns) and
differs from its own 3.14 baseline in the same last digits. So graded files are
compared with floats within a relative tolerance of 1e-12, everything else exact
(README, Testing).

**14:31 · Final smoke on the release commit: PASS** (`pytest -m smoke`, 8 min 33 s):
748/748 cities pairs ok on slot_diff and duplicates; min held-out AUC layers 20–36 =
0.9854 (cities), 1.0000 (sp_en_trans); max |AUC − expected-auc.jsonl| = 0.0 for both.

**14:38 · PyPI 0.1.0** published from `b8f4829` (same `src/` and pyproject as
`8306077`): the wheel's package code is byte-identical to `src/` at the end of this
pass.

**14:45 · Review changes.** The `runs/` files FINDINGS.md and experiments/README.md cite
are committed (28 files, 13 MB; the rest of `runs/` stays ignored) and FINDINGS says so
at the top; issue #1 for the question-level pass-through gap, referenced from the xfail
and the README; README: Marks & Tegmark in the status line, FINDINGS and NOTEBOOK linked
at the top, CI badge, why graded files are compared within a tolerance;
`experiments/wikidata-towns-pass2.jsonl` documented as the unrun pass-2 candidate list;
PLAN.md rewritten as a design document with as-built notes (checked against the code:
the threshold sweep it asks for is not built). GitHub topics set.

**15:01–15:09 · Matched check on the release code: exact match.** `8306077` on Python
3.14.7, numpy 2.5.4, package list equal to the baseline environment's:

| file | normalised sha256 | baseline |
|---|---|---|
| graded-29-B.jsonl | `f0da59f8e8b3c1a3c1a7275222400e159770edc71fc08029cc0f7def6f083d3e` | same |
| check-behaviour.jsonl | `7fe53294b46a37f4dd53e9e68f13d47c0fdac4b162a9b0c7875b89f4009ee76c` | same |

Zero differing fields, floats included. **No behaviour change in this pass, confirmed
bit for bit.**

**15:10 · 0.1.0.post1.** PyPI keeps each version's description as uploaded, and 0.1.0's
predates the review changes; `0.1.0.post1` has the same code and the current README
(links absolute so they work on PyPI, install from PyPI). The six model-free example
outputs regenerated (only tool_version differs). ruff, mypy, fast tests (74 passed,
1 skipped, 1 xfailed) and `twine check --strict` pass; the wheel's package code equals
`src/` and its long description equals README.md.
