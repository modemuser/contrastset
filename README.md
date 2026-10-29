# contrastset

[![ci](https://github.com/modemuser/contrastset/actions/workflows/ci.yml/badge.svg)](https://github.com/modemuser/contrastset/actions/workflows/ci.yml)

> **Status: v0.1.** Used for one experiment so far (an "I don't know" readout in
> Qwen3-8B), including a reproduction of Marks & Tegmark on Qwen3-8B as a built-in
> probe-stage check. The file formats and the CLI may still change.

Two files to read alongside the code: [FINDINGS.md](https://github.com/modemuser/contrastset/blob/main/FINDINGS.md), what building the
tool measured, each result with its data in `runs/` and how far it can be trusted,
including claims that turned out wrong and were corrected; and
[NOTEBOOK.md](https://github.com/modemuser/contrastset/blob/main/NOTEBOOK.md), the chronological log of the release pass: every run,
check and decision, mistakes included.

Build and check **contrast sets**, matched pairs of texts that differ in exactly one
controlled variable, for interpretability experiments on local LLMs (large language
models). A probe (a small classifier trained on a model's internal activations)
finds whatever separates the two sides of your data. If the sides also differ in
length, wording, or how famous the names are, a probe trained on those pairs learns
those surface cues instead of the variable you care about, and its accuracy looks
like a result. Matched pairs remove the alternatives by construction: the two sides
share everything but the contrast, so whatever separates them is the contrast.
contrastset makes such pairs cheap to produce and checks that they differ in one
thing only.

Three commands, each useful on its own, sharing one file format: JSONL (JSON Lines,
one JSON object per line), readable by eye and with `jq`. Data goes to stdout,
progress and reports to stderr, so every command pipes.

| command    | reads                               | writes                    |
|------------|-------------------------------------|---------------------------|
| `grade`    | entities with questions and answers | what the subject model knows, with every raw sample |
| `generate` | a spec (+ a graded file, if used)   | pairs                     |
| `check`    | any pair file (`a` and `b` per line) | per-pair verdicts, exit code for CI |

Two models are involved and never confused: the **subject model** is the small
local model being studied (graded behaviourally, by its answers, never by asking it
whether it knows); the **generator model** writes sentence frames and never sees
what the subject knows.

```bash
pip install contrastset          # core: grade/generate/check over HTTP (Ollama, OpenAI-compatible servers)
pip install "contrastset[hf]"    # + in-process transformers: log-prob grading, the example-00 probe
contrastset --help
```

Python 3.11+. Extras: `hf` (torch, transformers), `anthropic` (a generator model
through the Anthropic API). The core needs neither: Ollama and any
OpenAI-compatible server are reached over HTTP. For development: `uv sync`, then
`uv run pytest`.

## The ladder

Each example adds one concept. Every example folder holds its inputs and its
committed outputs, so you can read the result of each step without running it.
Commands are run from inside the example folder. The committed files were made
with `SOURCE_DATE_EPOCH=1793145600` (timestamps fixed), so rerunning a
deterministic step reproduces them byte for byte.

### 00 · Marks & Tegmark: does your probe stage work at all?

Not a contrastset feature but a check on the stage after it: a difference-of-means
probe on Marks and Tegmark's true/false statements, on Qwen3-8B, with a frozen split
and the expected per-layer AUC. If your probe code can't reproduce this, a null
result on your own pairs means nothing. See
[`examples/00-marks-tegmark/README.md`](https://github.com/modemuser/contrastset/blob/main/examples/00-marks-tegmark/README.md); it is
also the opt-in smoke test (see Testing).

### 01 · lexical control: spec, inline slots, hand-written frame, `check`

No model, no graded file, about a second.

```bash
cd examples/01-lexical-control
contrastset generate spec.yaml > pairs.jsonl
contrastset check pairs.jsonl --spec spec.yaml > check.jsonl
```

A **spec** says what the contrast is. Here, two inline word lists fill one
placeholder, `[[X1]]`, in hand-written **frames**:

```yaml
frames:
  - The [[X1]] is next to the window.
slots:
  select:
    a: {source: inline, values: [table, lamp, chair]}
    b: {source: inline, values: [shelf, sofa, desk]}
```

Each pair fills the same frame twice, once with a value for side a and once for
side b, so the sides differ only in the slot, by construction. A probe that tells
these pairs apart is reading word identity: run it as a control next to every real
contrast set. `check` runs the checks the spec lists (here `length`, `overlap`) and
exits non-zero if any pair fails.

### 02 · capitals: `grade`, then select slots from what the model knows

```bash
cd examples/02-capitals
contrastset grade entities.jsonl --model qwen3:14b --template template.txt > graded.jsonl
contrastset generate spec.yaml --graded graded.jsonl > pairs.jsonl
contrastset check pairs.jsonl --spec spec.yaml > check.jsonl
```

**Entities** are what you ask about, with reference answers (any alias counts):

```json
{"entity": "Burundi", "type": "country", "questions": [{"id": "capital", "q": "What is the capital of Burundi?", "answers": ["Gitega"]}]}
```

`grade` asks every question 8 times at temperature 0.7 plus once greedily (always
the most likely token), matches the answers against the references and sorts each
(entity, question) into a **bin**: `known`, `confidently_wrong` (the same wrong
answer again and again), `unstable` (scattered: guessing) or `refused`. Every raw
sample is kept.

The **template** is the exact text the model sees, chat format included; the
adapters send it as-is. It is always your choice (`--template` is required and has
no default) and carries the refusal phrases written for its own wording:

```
---
refusal_phrases: ["i don't know", "i do not know", "not sure", ...]
---
<|im_start|>user
{{question}} Answer with just the city name. If you do not know, say "I don't know".<|im_end|>
...
```

A **graded slot** selects candidates from the graded file. The unit is an (entity,
question) pair, and selectors filter a flattened view of it. To see that view,
print it, since it is itself a valid slot file:

```bash
contrastset generate --candidates --graded graded.jsonl | jq -c 'select(.bin == "known")' | head -2
```

Fields: `value` (= `entity`), `question`, `bin`, `correct_rate`, `consistency`,
`refusal_rate`, `n_correct`, `n`, `type`, `entity_bin`, `q.<id>` (the bin of the
entity's other questions), `answer.<id>`, `meta.<key>`. Filter in the spec with
`where` (`{bin: known}`, `{correct_rate: {max: 0.125}}`), or with jq and feed the
result back as `{source: file, path: ...}`: selectors are a convenience, not a
requirement.

This example doubles as a sanity check of the matcher: a 14B model knows nearly
every capital, so many `unstable` answers would mean the matcher, not the model, is
broken. qwen3:14b knows 34 of 37; the three it doesn't are the classic confusions
(Burundi → "Bujumbura", the capital until 2019; Sri Lanka → "Colombo"; Côte
d'Ivoire split between Abidjan and Yamoussoukro). With only three unknown
countries, every b side recurs about 17 times, and `check` says so: `duplicates`
warns on all 102 pairs. Read the warnings as information about your data, not
noise.

### 03 · two-clause rivers: sides, familiar entities, clause order, factors, behaviour

```bash
cd examples/03-two-clause-rivers
contrastset grade entities.jsonl --model qwen3:14b --template template.txt > graded.jsonl
contrastset generate spec.yaml --graded graded.jsonl --adapter anthropic --model claude-haiku-5-5 --limit 60 > pairs.jsonl
contrastset check pairs.jsonl --spec spec.yaml > check.jsonl          # add the subject flags below for `behaviour`
contrastset check pairs.jsonl --spec spec.yaml --checks behaviour --require behaviour \
    --graded graded.jsonl --subject-model qwen3:14b --template template.txt > check-behaviour.jsonl
```

Two-clause questions: side a asks about two towns whose river the model knows,
side b swaps one for a town it doesn't. New concepts:

- **`sides`** say which slots go where: `a: [shared, known]`, `b: [shared, unknown]`.
  A slot on both sides keeps its value.
- **`slot_defaults: {familiar: true}`**: every town must have its other questions
  (country, region) known, so the entity is familiar and only the river fact varies.
- **`order: both`** also emits every pair with the clauses reversed (`rotations`
  cycles through every position).
- **`factors`** control a variable instead of leaving it to chance: here whether the
  neighbouring town is in the same country, balanced and recorded per pair.
- **A generator model writes the frames** from 3 to 5 example pairs, entity-free, so
  it never sees the contrasted towns. It is unseeded; to rebuild the same pairs,
  reuse their frames: `generate ... --frames-from pairs.jsonl`.
- **`behaviour`** asks the subject model both sides and scores each clause: is the
  known clause still answered and the unknown one still not, inside the composite?
  It needs `--graded`, a subject model and a template; where any is missing it is
  *skipped*, which is not a pass: `--require behaviour` fails the run on a skip.

### 04 · intent pairs: hand-written pairs, your own check, a blind judge

```bash
cd examples/04-intent-pairs
contrastset check pairs.jsonl --spec spec.yaml --checks length,overlap,slot_diff,intent_checks.no_intent_words > check.jsonl
contrastset check pairs.jsonl --spec spec.yaml --checks blind_judge > check-judge.jsonl   # calls Haiku
```

No `generate`: the pairs are written by hand (the four in `pairs.jsonl` are stubs),
and the spec has no slots, only a contrast and checks. `check` needs only `a` and
`b`; other fields (`scenario`) pass through.

- **Your own check** is any function `check(pair, spec, /, **options) -> Verdict |
  None`, referenced by dotted path (`intent_checks.no_intent_words`, a module next to
  the spec). This one fails pairs that carry the contrast in a marker word
  ("deliberately"), which `wall-04` does on purpose.
- **Thresholds are per contrast**: these pairs differ in a clause, so the spec loosens
  `overlap` and `slot_diff`.
- **`blind_judge`** shows both sides to a judge model in random order and asks which
  one is the target. Over the set it tests the judge's accuracy against chance. For
  a contrast the subject's knowledge defines (example 03) a judge has only surface
  form to go on, so beating chance means a leak. For a contrast readable from meaning,
  like intent, a judge *should* succeed: there it is a manipulation check, not a
  leak test.

## The JSONL contract

Every JSONL input (entities, a graded file, pairs) is a path or stdin (`-`); a spec
is a YAML or JSON file. Every verb writes JSONL to stdout (or `-o FILE`), one line per item as soon as it is done, so you can watch partial
results and stop early (`grade` and `generate` take `--limit N` for a cheap first
look). What each
command **requires** from its input; every other field is passed through unchanged
and listed once on stderr, so a typo in an optional field is visible without
breaking files written by other tools:

| input | required | used if present |
|---|---|---|
| entity (`grade`) | `entity`, `type`, `questions[].q`, `questions[].answers` | `questions[].id`, `meta` |
| graded (`generate --graded`) | `entity`, `questions[].id`, `questions[].bin` | `type`, `meta`, `bin`, `questions[].{q, answers, correct_rate, ...}` |
| pair (`check`) | `a`, `b` | `pair_id` (else sha256 of a and b), `frame`, `fills`, `factors` |

One known gap in v0.1.0: `grade` keeps unknown fields on an entity line but not
inside its `questions[]` entries
([modemuser/contrastset#1](https://github.com/modemuser/contrastset/issues/1); the test
for it is marked as an expected failure).

Specs are hand-written configuration and stay strict: an unknown key is an error.
The schemas are in `src/contrastset/records.py` (`GradedView`, `PairView` are the
read contracts). Every output line carries provenance: model, template hash,
sampling settings (including the server's defaults), seed, input hashes, time.

## Log-prob grading (`grade --logprob`)

Sampling 8 answers measures what the model *says*, with sampling noise: a question
it answers correctly 73 % of the time comes out 8/8 about once in twelve tries.
With `--adapter hf`, `--logprob` also reads the untempered probabilities: the mass
of completions that start with a correct answer, with the greedy answer, and with a
refusal phrase. These are binned with the same thresholds and stored beside the
sampled numbers (`questions[].logprob`; `logprob_sampler` holds the same under the
sampling distribution). With `-n 0` the grade uses the probabilities alone: one
greedy decode plus teacher-forced scoring, no samples, deterministic.

**Marginalised over tokenizations.** The probability of a text is the sum over
every token sequence that spells it, and models often write rare names in a
segmentation the tokenizer would not choose. Scoring the canonical segmentation
alone undercounted "Rhineland-Palatinate" on Qwen3-8B by 26× (0.031 vs 0.823).
contrastset expands, step by step, every token consistent with the rest of the
answer (a byte trie over the vocabulary, no top-k) and sums the finished paths.

**The pruned-mass bound.** The one approximation: a partial path whose probability
falls below 1e-5 is dropped, and the dropped mass is recorded per question
(`p_correct_pruned_max`, `p_modal_pruned_max`). The true value lies in
[p, p + pruned_max]. Pruning limits and the bound go into provenance. Refusal
phrases are still scored in their canonical segmentation (everyday English, where
that is the model's own), a known limitation.

## One template for grading and probing

The template a model is probed or checked with must be the exact string it was
graded with: bins graded under one wording don't transfer to activations or answers
collected under another (see Lessons: one line in the prompt can create a whole
bin). Every graded line records the template's sha256. `check --checks behaviour`
compares the template it is given with the one in the graded file's provenance and
refuses a mismatch; `--allow-template-mismatch` overrides, and the override is
recorded in the check's provenance. Probe code downstream should do the same
(`grade.require_same_template`).

## Backends

| adapter | reaches | use |
|---|---|---|
| `ollama` (default) | Ollama's native API in raw mode: your template is sent as-is | subject or generator |
| `openai`, `openai-chat` | any OpenAI-compatible server (vLLM, llama.cpp server, online APIs) | subject or generator |
| `anthropic` | the Anthropic API (`[anthropic]` extra) | generator only |
| `hf` | transformers in-process (`[hf]` extra): samples, log-probabilities, activations | subject |
| `fake` | canned answers, for tests | tests |

**The rule: Ollama when samples suffice, HF log-prob for anything that feeds a
probe.** Ollama is much faster for sample-level questions (does the model answer
this? does a composite prompt still work?). A graded file that defines the cells of
a probe should come from the HF adapter with `--logprob`, on the same weights the
probe loads: a Q4 Ollama model and the bf16 HF model are different models, and
log-prob masses carry no sampling noise.

**Sampling order.** Where contrastset samples itself (the `hf` adapter), it uses one
order: top-k, then top-p, both on the untempered probabilities, then temperature,
then the draw. That is llama.cpp's order, so HF and Ollama grades sample alike;
transformers' own `generate()` tempers first, and then top-p cuts a sharpened
distribution and the samples come out less varied. Where a server samples
(`ollama`), provenance records the server's settings, its overrides and its
version. Every graded line records all sampler parameters, and `--top-k`/`--top-p`
set them explicitly.

## Testing

| tier | command | needs | time | when |
|---|---|---|---|---|
| fast (default) | `uv run pytest` | nothing: no model, no network | ~10 s | always; the only tier CI runs |
| smoke | `uv run --extra hf pytest -m smoke` | `[hf]`, Qwen/Qwen3-8B weights, network for the statement sets | ~10 min (M-series Mac) | before a release, after touching anything a probe depends on |
| regrade | `uv run --extra hf pytest -m regrade` | `[hf]`, Qwen/Qwen3-8B weights | ~27 min (M-series Mac) | when the grading path changes |

- **fast** reruns every example in the ladder against its committed files (rescoring
  stored samples, regenerating pairs from stored frames, rechecking), plus unit
  tests with a fake adapter.
- **smoke** runs example 00 end to end: fetch the statement sets, `check` the
  `cities` pairs (every pair must differ in the country slot only), read activations
  at every layer, fit the per-layer arrow on the frozen split, and assert held-out
  AUC ≥ 0.95 at every layer from 20 to the last.
- **regrade** regrades the 29 towns of example 03 with HF Qwen3-8B, template B
  (`experiments/templates/qwen3-name-only.txt`), `-n 0 --logprob`, and compares
  with the frozen `tests/regrade/graded-29-B.jsonl`: bins and modal answers must
  match exactly, masses within the pruned-mass bound.

Graded files are compared within a tolerance, not bit for bit: the refusal mass
(`p_refusal`) differs in the last digit across numpy builds (Python 3.11 with numpy
2.4.6 vs 3.14 with 2.5.4, same code; NOTEBOOK.md, 2026-10-29).

## Lessons

Two things building this tool taught, with the data in [FINDINGS.md](https://github.com/modemuser/contrastset/blob/main/FINDINGS.md):

- **Tokenization undercount.** Log-prob grading over the tokenizer's own
  segmentation undercounts multi-token answers by up to 26×, because the model
  writes rare names in other segmentations. Sum over all of them (above).
- **The refusal instruction makes the refused bin.** Of 49 towns an earlier grade
  had refused, Qwen3-8B still refused the river for 45 with "If you do not know, say
  'I don't know'" in the template, and for none without it. The refused bin was the instruction working, not a
  state of the model, and the instruction also hid answers the model gives reliably
  without it (3 of the 45). Grade with the template you probe with, and be wary of
  bins a single prompt line can create.

The downstream experiment (difference-of-means probes on Qwen3-8B, where example 00
comes from) is kept in a separate repository, [knowing-probe](https://github.com/modemuser/knowing-probe), which pins this
version.

## Not in scope

No probing, plotting or model internals in the package: output is a dataset file;
probe with TransformerLens, nnsight or whatever you already use. (Example 00 has a
small probe script, outside the package, as a check on that stage.) The design
brief is [PLAN.md](https://github.com/modemuser/contrastset/blob/main/PLAN.md); maintenance passes are logged in
[NOTEBOOK.md](https://github.com/modemuser/contrastset/blob/main/NOTEBOOK.md).

## Licence

MIT, see [LICENSE](https://github.com/modemuser/contrastset/blob/main/LICENSE).
