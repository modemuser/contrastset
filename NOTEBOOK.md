# Notebook

Chronological record of maintenance passes on contrastset: what was run, in what
order, what came out, including mistakes. Research results go to FINDINGS.md. Times
are CET, newest entries at the bottom.

## 2026-10-29 · Release pass to v0.1.0

Goal (Thomas's brief): contrastset public as v0.1.0, the Marks & Tegmark step 0 as an
opt-in regression test, no behaviour change. Commit per section.

**12:10 · State found.** No commits yet on `main` (so no history to audit), no git
remote. 70 fast tests pass + 1 skipped (the brief said 69). PyPI name `contrastset`
unclaimed (404): the 0.0.1 placeholder in `dist-placeholder/` was built but never
uploaded. No PyPI credentials on this machine (no `~/.pypirc`, no token env var).
`knowing-probe` has no remote either.

**12:12 · Two gaps between the brief and the repos.**
1. *No 29-town template-B HF log-prob graded file exists.* `runs/parity-hf-logprob-29.jsonl`
   is template A; template B with HF log-prob exists only for 294 towns. Decision: make
   the baseline file now, before any change, with the exact flags of the 294-town B run
   (`runs/hf-logprob-294-B.sh`) on the 29 towns of `examples/03-two-clause-rivers/entities.jsonl`.
   With `-n 0 --logprob` the grade draws no samples, so it is deterministic.
2. *There is no graded file for the Marks & Tegmark statements.* Step 0 in knowing-probe
   fed the labelled CSVs from geometry-of-truth straight to the activation reader; truth
   labels come from the dataset, nothing was graded or generated. So the smoke tier can
   run check → activations → per-layer AUC, but not grade/generate, and a regrade tier
   "against the frozen step 0 graded file" has nothing to compare with.
   Decision (for Thomas to ratify): the regrade tier regrades the 29-town template-B
   file from (1) with `--logprob` and compares with the frozen copy.

**12:19 · Baseline timing probe** (2 towns, 2 pairs): grade 130 s incl. model load
(≈55 s/town), behaviour check 29 s (≈10 s/pair). Full baseline ≈ 37 min.
Script: `baseline.sh <label> [limit]` (session scratchpad; commands below so it can be
redone without it):

    uv run contrastset grade examples/03-two-clause-rivers/entities.jsonl --adapter hf --model Qwen/Qwen3-8B \
      --template experiments/templates/qwen3-name-only.txt -n 0 --logprob --temperature 0.7 --top-k 20 --top-p 0.95 \
      --gate country -o graded-29-B.jsonl
    E=examples/03-two-clause-rivers
    uv run contrastset check $E/pairs.jsonl --spec $E/spec.yaml --checks behaviour --require behaviour \
      --graded $E/graded.jsonl --subject-model qwen3:14b --template $E/template.txt -o check-behaviour.jsonl

Hash rule: sha256 over the JSONL with `provenance.created` and `provenance.tool_version`
dropped (wall clock and version string, the only fields identical code may change),
keys sorted. Raw file sha256 recorded alongside.

**12:21 · Full baseline started** (background, Qwen3-8B on HF/MPS, then qwen3:14b on Ollama).

**12:24 · Snapshot commit `e30724a`.** The working tree as found (code, tests, examples
01–04, FINDINGS, PLAN, experiments, scripts, uv.lock), without `runs/` (19 MB of run
outputs, logs and chain scripts with absolute paths) and with the original
`.gitignore`, so every release section below is its own reviewable diff. The repo had
no commits, so there was no history to audit for blobs committed and later ignored.

**12:25 · Hygiene scan (read-only, during the baseline).** No absolute paths, home
dirs, machine names, tokens or keys in tracked files. The only absolute paths are in
`runs/*.sh` (a `cd` into the author's checkout), and `runs/` is ignored as a whole.
API keys are only ever read from the environment (`OPENAI_API_KEY`, the Anthropic SDK's
own lookup).

**12:26 · Example 00 data: not copied.** saprmarks/geometry-of-truth has no licence file
(GitHub API: `license: null`), so its CSVs are not redistributed. `fetch_data.py`
downloads them at commit 5d1c630 and checks sha256 (f2560a0c… cities, 9afa97c1…
sp_en_trans; fetch tested, both match). The frozen split is committed as `split.json`
(lists of held-out cities / Spanish words, seed 20261029): 249 cities and 115 words,
the same counts as knowing-probe's sidecars. `expected-auc.jsonl` = knowing-probe
`runs/step0/*-summary.json` per layer (commit 911d9e0 there). The brief's "1.00 from
layer 15 (cities)" is 0.994–1.000 over layers 15–35 and 0.985 at layer 36; the README
gives the exact figures.

**12:28 · `check` on the cities pairs** (748 pairs from `make_pairs.py`, checks
length,overlap,slot_diff,duplicates): slot_diff ok on all 748; length 38 warn, 9 fail
(country names up to five words longer); exit 1. A first README draft said "fails on
many of them"; corrected to the counts.

**12:29 · Pass-through test finds a gap.** `grade` keeps unknown fields on an entity line
but drops unknown fields inside `questions[]`: `grade_question` builds each
QuestionGrade afresh. `--priors` and `generate --candidates` keep both levels. Not fixed
(no behaviour change in this pass); the question-level test is `xfail(strict=True)` and
the README says so. For Thomas to decide.

**12:31 · mypy on tests.** `src` passed strict; `tests` had 14 errors (Optional values
used unchecked, one annotation naming `torch.Tensor` through an importorskip module,
one untyped fixture, list invariance GradedEntity vs GradedView). Fixed with asserts,
annotations and one commented `type: ignore`; no test logic changed.

**12:36 · PyPI via trusted publishing (Thomas created the PyPI account).** Added
`.github/workflows/publish.yml`: on a `v*` tag, build with `uv build`, `twine check
--strict`, check the tag equals the pyproject version, upload with
pypa/gh-action-pypi-publish through the GitHub environment `pypi` (OIDC, no token
stored). Thomas registers a pending publisher on PyPI: project contrastset, owner
modemuser, repo contrastset, workflow publish.yml, environment pypi. Consequence:
pushing the v0.1.0 tag *is* the PyPI release, so the tag is pushed only after both
smoke runs pass and the baseline hashes match. Also added `.github/workflows/ci.yml`:
fast tier only (ruff, mypy, pytest) on Python 3.11 and 3.14.

**12:37 · Pending publisher registered on PyPI** by Thomas with the values above.

**12:22–12:58 · Step 0 baseline, before any change to library code** (`baseline.sh before`).
Grade 26 min 57 s, behaviour check 8 min 16 s.

| file | normalised sha256 | raw sha256 |
|---|---|---|
| graded-29-B.jsonl | `f0da59f8e8b3c1a3c1a7275222400e159770edc71fc08029cc0f7def6f083d3e` | `a77d9d23…` |
| check-behaviour.jsonl | `7fe53294b46a37f4dd53e9e68f13d47c0fdac4b162a9b0c7875b89f4009ee76c` | `e5a372eb…` |

Grade: question bins country 29 known; region 23 known / 6 unstable; river 8 known /
5 confidently_wrong / 16 unstable. Behaviour check: 52 ok, 8 fail, exit 1. The committed
`examples/03-two-clause-rivers/check-behaviour.jsonl` (2026-10-28) has 9 fail: pair
`cbd1f575701e0b5e` was fail there and is ok today, same model digest (Ollama 0.34.2; the
committed file predates recording the server version). So the committed example output
is not byte-reproducible today; it was not touched. Exit 1 is that example's normal state.
Determinism: the 12:19 timing probe (first 2 towns, first 2 pairs) matches the full
run's first lines after normalisation (the check only in `inputs.pairs`, the probe read a
2-line file). Fast tests before: 70 passed, 1 skipped.

**12:59–13:10 · Smoke before cleanup: PASS.** No smoke tier existed yet, so this ran
knowing-probe's own step 0 code (`knowing_probe.acts`, `arrow`, `mt_probe.item_of`;
contrastset at `e30724a` via its editable path dependency) on fresh activations, all
37 layers, nothing written into knowing-probe (scratch script `smoke_before.py`).
cities: min held-out AUC over layers 20–36 = 0.9854 (layer 36); sp_en_trans: 1.000.
Max |AUC − knowing-probe `runs/step0/*-summary.json`| over layers 1–36 = 0.0 for both:
the MPS forward passes reproduce the 11:13 activations' AUCs exactly. 0.27 s/statement
steady, ~9 min of reading.

**13:11 · Section 1, repo hygiene.**
- `.gitignore`: activations (`*.npy`, `*.npz`), model caches, `runs/` (all run outputs),
  notebook checkpoints, build output, ruff cache, `.DS_Store`, and the example-00 data
  fetched from upstream.
- `LICENSE`: MIT, "2026 Thomas Misera" (the name on his other MIT repos; pyproject said
  only "Thomas").
- `pyproject.toml`: version 0.1.0, licence file, author, URLs (github.com/modemuser/
  contrastset), classifiers. Python floor stays 3.11 (wheel installed and imported on
  3.11.16 without torch; `--adapter hf` without the extra fails with "needs `pip install
  contrastset[hf]`"). Extras stay `hf` and `anthropic`; no `ollama` extra, because Ollama
  needs only httpx, which is core (the default adapter and the OpenAI adapter use it);
  `dev` stays a uv dependency group. All three for Thomas to ratify.
- `__version__ = "0.1.0"`, surfaced by `contrastset --version` (existing); new
  `tests/test_version.py` ties it to the package metadata.
- **Mistake on the way:** the first fast-test run after the version bump hung: the smoke
  and regrade test files were already in `tests/`, and their marker deselection belongs
  to section 4, so pytest began loading Qwen3-8B. Killed after ~2 min; nothing written.
  Fast runs use `--deselect` on those two files until section 4.
- **Version bump broke 4 byte-for-byte example tests**: committed model-free outputs
  (01 pairs+check, 02 pairs+check, 03 check, 04 check) carry `tool_version
  0.1.0.dev0`. Regenerated those six files with `SOURCE_DATE_EPOCH=1793145600`
  (scratch `regen_examples.py`), which refuses to write unless the normalised content
  (dropping tool_version, and `inputs.pairs` where the pair file itself was regenerated)
  is identical: it is, for all six. Model-made files (graded, check-behaviour,
  check-judge, the Haiku-made 03 pairs) keep their 0.1.0.dev0 provenance: that is the
  version that made them.
- Fast tests: 74 passed, 1 skipped, 1 xfailed (pass-through test file not part of this
  commit: 71 passed in this commit's own tree).

**13:20 · Section 2, code cleanup (no behaviour change).**
- Dead code: `generate.CANDIDATE_CORE` (defined, never used). No stale TODO/FIXME in
  `src/`; no experiment-only code paths found (river/town names appear only in
  docstrings and comments as examples). `scripts/` are research scripts, left as run.
- `--help` against the code: `grade --template` and `entities` now say "required unless
  --priors" (argparse can't require them, since `--priors` needs neither; the check
  is in `_grade`); `--language` shows its default; `--logprob` mentions the
  tokenization sum, the pruned-mass bound and `-n 0`; `generate --frames-from` no longer
  says "Haiku-made"; `check --checks` prints `checks.DEFAULT_CHECKS` instead of a copied
  list; `check --graded`/`--allow-template-mismatch` name the template-hash guard.
  `--graded`, `--require`, `--logprob`, `--allow-template-mismatch` were all present.
  Not changed, reported: `check` has no `--limit` (PLAN.md asked for it on every verb).
- Lint: ruff added to the dev group; rules pinned to ruff's classic defaults
  (E4, E7, E9, F) after `uv lock` pulled ruff 0.17.3, whose wider default set flagged
  refactors (context managers, blind excepts), not mechanical fixes. One fix: `l` →
  `layer` in `HFAdapter.run` (E741). `scripts/` excluded. mypy scope set to src+tests.
- `tests/test_passthrough.py` (the JSONL contract) and the test typing fixes from 12:29–12:31.
- ruff, mypy (28 files), fast tests: 74 passed, 1 skipped, 1 xfailed.

**13:24 · Section 3, Marks & Tegmark step 0 as `examples/00-marks-tegmark/`.**
From knowing-probe (commit 911d9e0 there; nothing deleted there): `split.json` (frozen
split, seed 20261029, 249 cities / 115 Spanish words held out), `expected-auc.jsonl`
(its step 0 per-layer summaries, incl. shuffle control and |cos(arrow, PC1)|), and
`auc.py`, a trimmed copy of `mt_acts.py` + `mt_probe.py` + `knowing_probe.acts/arrow`:
same hook-based last-token reader, same split function, same arrow/projection/AUC, no
plots, JSONL out; it refuses to run if the recomputed split differs from `split.json`.
New: `fetch_data.py` (pinned commit + sha256, instead of copying the unlicensed CSVs)
and `make_pairs.py` (cities as a pair file for `contrastset check`). README cites the
paper, states the reproduced numbers and why the shuffled-label control does not fall
to chance (true/false is PC1 from layer ~9).
Not moved because it does not exist: a graded file for these statements (see 12:12).

**13:30 · Section 4, test tiers.**
- fast (default): `addopts = -m 'not smoke and not regrade'`; 74 passed, 1 skipped
  (69 + 1 in the brief's count was 70 + 1 skipped here), 1 xfailed. Also on Python
  3.11 without torch: 66 passed, 2 skipped (test_hf skips as a module), 1 xfailed.
- smoke (`-m smoke`, `tests/test_smoke.py`): example 00 end to end: fetch, `check` the
  748 cities pairs (`slot_diff,duplicates` must be ok on every pair), `auc.py` for both
  sets, assert held-out AUC ≥ 0.95 at layers 20–36, print drift from
  `expected-auc.jsonl`. No grade/generate step: there is no graded file for these
  statements (12:12).
- regrade (`-m regrade`, `tests/test_regrade.py`): substitute for the brief's "regrade
  the step 0 statements" (nothing to regrade there): regrades the 29 towns, template B,
  `-n 0 --logprob`, against `tests/regrade/graded-29-B.jsonl` = the 12:49 baseline
  grade (raw sha256 a77d9d23…, made at e30724a before any change). Bins and modal
  answers exact; `p_correct`/`p_modal` (raw and sampler) within the larger pruned mass
  + 1e-6. For Thomas to ratify.
- CI (`.github/workflows/ci.yml`): fast tier only, ruff + mypy + pytest on 3.11 and
  3.14, `uv sync --locked` without extras. mypy marks the optional extras' modules
  (torch, transformers, numpy, huggingface_hub, anthropic) `ignore_missing_imports`,
  so it passes without them and still checks against them where installed.

**13:36 · Section 5, README.** Status line (v0.1.0, one experiment, API may change);
what a contrast set is and why matched pairs; install (git tag URL; PyPI once the tag
workflow has run); the ladder 00–04 (each verb with an example); the JSONL contract
incl. the question-level pass-through gap; log-prob grading (tokenization sum,
pruned-mass bound); the template-hash guard; backends and the rule (Ollama when samples
suffice, HF log-prob for anything feeding a probe); Testing (three tiers, measured
times: fast ~9 s, smoke ~10 min, regrade ~27 min); Lessons (tokenization undercount,
refusal-instruction effect) linking to FINDINGS.md here (knowing-probe has no public
URL, so it is named, not linked); licence. Claims checked against data while
writing: "fails on many" (example 00 length check) and "45 of 49 towns" (needed "of 49
towns an earlier grade had refused") were corrected; "`--limit` on every verb" was
false (`check` has none) and was removed.

**13:40 · Section 6 prep: release workflow committed** (`.github/workflows/publish.yml`,
see 12:36). Check plan: `src/` is byte-identical from `fe5ce77` (section 2) to this
commit (`git diff fe5ce77 HEAD -- src` empty), so one behaviour check on this commit
covers sections 2–6; the separately queued section-2 run was cancelled before it
started, and the run goes to a worktree at this commit instead. Then the final smoke,
also in a worktree at this commit.

**13:45 · Regrade comparator exercised offline** (no model): the frozen file against
itself passes; one p_correct shifted by 0.5× its question's pruned-mass bound passes,
by 2× fails (Kraków/region, bound 4.48e-05); a changed modal answer fails. The tier
itself (27 min of HF grading) was not run end to end in this pass: the behaviour
checks run the identical grade command and compare hashes, which is stricter.

**13:50 · STOP: section 1 grade hash differs from the baseline.** s1 (worktree at
`9fe330e`): graded-29-B normalised `91535894…` vs baseline `f0da59f8…`; check-behaviour
`7fe53294…` identical. What changed: 78 values, all `questions[].logprob.p_refusal` and
`questions[].refusal_rate` (which is set to it): last-digit differences (e.g.
4.917343709552636e-10 vs 4.917343709552635e-10). Every bin, modal answer, p_correct,
p_modal (raw and sampler) and every pruned-mass bound is bit-identical. Section 1's only
`src/` change is the `__version__` line. p_refusal is the one mass scored as a padded
batch (`HFAdapter.score_many`, refusal phrases); p_correct/p_modal go through
`prefix_mass`. Working hypothesis: run-to-run float nondeterminism of the batched MPS
forward pass, not a behaviour change. Not fixed forward; release halted. To test it:
the unchanged pre-release code (`e30724a`, worktree) regrades the first 5 towns
(4 of which differ) after the queued release-commit check and smoke.

**14:20 · Cause of the mismatch: my check environment, not the code.** The release-commit
run (`final`, worktree at `d0a3562`) is identical to s1 (0 differences) and differs from
the baseline at the same 39 p_refusal positions: systematic, so not run-to-run noise
(hypothesis of 13:50 dropped). Both worktree venvs came up on **Python 3.11.16 with
numpy 2.4.6**; the baseline ran in the main `.venv`, **Python 3.14.7, numpy 2.5.4**.
`uv.lock` pins numpy by interpreter (2.4.6 for < 3.12, 2.5.4 for ≥ 3.12), so `uv sync`
in a fresh worktree, which picked 3.11, got a different numpy. My mistake: I checked
torch/transformers versions when creating the worktrees, not the interpreter.
Matched check: worktree `wt-final314` at `d0a3562` with `uv sync --locked --all-extras
--python 3.14.7`; its package list equals the baseline venv's (apart from contrastset's
path and ruff). Queued: (1) `e30724a` (pre-release code) on 3.11, first 5 towns: should
reproduce s1's values, not the baseline's; (2) the full baseline script on
`wt-final314`: should match the baseline hashes exactly.

**14:31 · Final smoke on the release commit: PASS** (`pytest -m smoke` in the `d0a3562`
worktree, Python 3.11 env, 8 min 33 s): 748/748 cities pairs ok on slot_diff and
duplicates; min held-out AUC layers 20–36 = 0.9854 (cities), 1.0000 (sp_en_trans);
max |AUC − expected-auc.jsonl| = 0.0 for both.

**14:37 · Drift confirmed with identical code.** `e30724a` (pre-release, unchanged) on
the Python 3.11 / numpy 2.4.6 env, first 5 towns: identical to s1 (0 differences), and
14 last-digit p_refusal/refusal_rate differences (max relative 4.3e-16) from the
baseline, which is the same code on Python 3.14.7 / numpy 2.5.4. So the interpreter and
numpy build move the last digit of the batched refusal mass; the code changes do not.

**14:38 · Comparison rule changed (Thomas's decision, conditional on the confirmation
above):** graded files are compared with floats within a relative tolerance of 1e-12,
everything else exact (provenance.created and tool_version ignored as before)
(`compare_tol.py`, scratch). Results against the baseline: s1 and the release commit
(`final`): 78 float fields differ, max relative 6.3e-16, 0 beyond tolerance, 0 non-float
differences: **pass**. Behaviour-check hashes: exact match at every step
(`7fe53294…`). With tolerance 0 the same script flags exactly those 78 fields.
Still running for the record: the full baseline on the release commit in a Python
3.14.7 env with the baseline's package list (expected to match exactly).

**14:39 · Release.** Tag `v0.1.0` on the commit that adds this entry (it differs from
the checked `d0a3562` by NOTEBOOK.md only); create github.com/modemuser/contrastset
(public), push main and the tag; the tag runs the publish workflow (PyPI, trusted
publishing).
