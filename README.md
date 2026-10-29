# contrastset

Build and check **contrast sets**, matched pairs of texts that differ in exactly one
controlled variable, for interpretability experiments on local LLMs (large language
models). A probe (a small classifier trained on a model's internal activations)
trained on sloppy pairs learns surface cues instead of the variable you care
about. contrastset makes it cheap to produce pairs that differ in one thing, and to
check that they do.

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
uv sync                      # Python 3.11+; extras: --extra anthropic, --extra hf
uv run pytest                # runs every example below against its committed files
```

## The ladder

Each example adds one concept. Every example folder holds its inputs and its
committed outputs, so you can read the result of each step without running it.
Commands are run from inside the example folder. The committed files were made
with `SOURCE_DATE_EPOCH=1793145600` (timestamps fixed), so rerunning a
deterministic step reproduces them byte for byte.

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
adapters send it as-is. It is always your choice (`--template` has no default) and
carries the refusal phrases written for its own wording:

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

This example is also a smoke test: a 14B model knows nearly every capital, so many
`unstable` answers would mean the matcher, not the model, is broken.
qwen3:14b knows 34 of 37; the three it doesn't are the classic confusions
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

## File contracts

What each command **requires** from its input; everything else is passed through
unchanged (and listed once on stderr, so typos in optional fields are visible):

| input | required | used if present |
|---|---|---|
| entity (`grade`) | `entity`, `type`, `questions[].q`, `questions[].answers` | `questions[].id`, `meta` |
| graded (`generate --graded`) | `entity`, `questions[].id`, `questions[].bin` | `type`, `meta`, `bin`, `questions[].{q, answers, correct_rate, ...}` |
| pair (`check`) | `a`, `b` | `pair_id` (else sha256 of a and b), `frame`, `fills`, `factors` |

**One template for grading and probing.** The template a model is probed or
checked with must be the exact string it was graded with: bins graded under one
wording don't transfer to activations or answers collected under another. `check
--checks behaviour` and the probe scripts compare the template's hash with the one
in the graded file's provenance and refuse a mismatch (`--allow-template-mismatch`
overrides, and is recorded).

**Sampling order.** Where contrastset samples itself (the `hf` adapter), it uses one
order: top-k, then top-p, both on the untempered probabilities, then temperature,
then the draw. That is llama.cpp's order, so HF and Ollama grades sample alike;
transformers' own `generate()` tempers first, and then top-p cuts a sharpened
distribution and the samples come out less varied. Where a server samples
(`ollama`), provenance records the server's settings, its overrides and its
version. Every graded line records all sampler parameters, and `--top-k`/`--top-p`
set them explicitly.

**`grade --logprob`** (hf only) also judges each question from the untempered
probabilities: the mass of completions starting with a correct answer, with the
greedy answer and with a refusal phrase, binned with the same thresholds and stored
beside the sampled numbers (`questions[].logprob`). The mass of an answer is summed
over every token segmentation that spells it: models often write rare names in a
segmentation the tokenizer would not choose, and scoring one segmentation can
undercount by more than an order of magnitude (26× for "Rhineland-Palatinate" on
Qwen3-8B; FINDINGS.md). Pruning limits and the bound on pruned mass are recorded.

Specs are hand-written configuration and stay strict: an unknown key is an error.
The schemas are in `src/contrastset/records.py` (`GradedView`, `PairView` are the
read contracts). Every output line carries provenance: model, template hash,
sampling settings (including the server's defaults), seed, input hashes, time.

## Not in scope

No probing, plotting or model internals: output is a dataset file; probe with
TransformerLens, nnsight or whatever you already use. Results that came out of
building the tool are in `FINDINGS.md`; the design brief is `PLAN.md`.
