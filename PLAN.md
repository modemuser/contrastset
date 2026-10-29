# Prompt: contrast-set library (grade / generate / check)

Oct 28, 2026 · @Thomas

## Context and goal

I want to build a small, standalone Python library and CLI for constructing **contrast sets** (matched stimulus pairs) for mechanistic-interpretability experiments on local LLMs. I will publish it on my GitHub. The first consumer is my own experiment: finding and validating an "I don't know" / uncertainty readout in a local Qwen3 (and later Llama, Gemma), using matched pairs that differ only in whether the model actually knows a fact.

The core problem it solves: careful dataset construction is what blocks these experiments. A probe trained on sloppy pairs learns surface cues (fictional-sounding names, odd phrasing) instead of the variable I care about. The library should make it cheap to produce pairs that differ in exactly one controlled variable, and to verify that they do.

Two models are involved, and they must never be confused:

- **Subject model**: the small local model being studied (e.g. Qwen3 via a local inference server such as Ollama / vLLM / llama.cpp, or HuggingFace transformers directly). Its knowledge is graded behaviourally.
- **Generator model**: a capable model (an online API or a larger local one) that writes the matched sentences under constraints. It must never be allowed to guess what the subject model knows.

I consider myself a noob on ML jargon: define every abbreviation once when first used in code comments and docs.

## Design constraints

Unix philosophy, strictly. Three small commands that each do one thing, share one file format, and are each useful on their own. Someone must be able to use `grade` with their own hand-written pairs, or `generate` with a different grading rule, without touching the rest.

- **The interface is the file format.** Every verb reads and/or writes JSONL (one JSON object per line). Humans must be able to inspect it with their eyes and `jq`. No binary caches, no hidden state between verbs.
- **No probing, no plotting, no model internals.** Output is a dataset file; people probe with TransformerLens / nnsight / whatever they already use. Resist every temptation to grow a framework.
- **Model-agnostic adapters.** Subject and generator are reached through a tiny interface (`complete(prompt, n, temperature) -> list[str]`), with adapters for an OpenAI-compatible HTTP endpoint (covers Ollama, vLLM, llama.cpp server, most online APIs) and for HuggingFace transformers. Add nothing else.
  - As built: `ollama` (native API in raw mode: Ollama's OpenAI-compatible endpoint wraps prompts in its own chat template, ignores `n` and lets Qwen3 think silently), `openai` (raw completions, or chat for generators), `anthropic` (official SDK, generator only; no temperature or seed since 2026-10-21), `hf` (in-process transformers; also `score(prompt, continuation)` for log-probabilities, which activation and probability experiments need), and `fake` for tests.
- **Reproducible by construction.** Every output line carries provenance: which model, which prompt template version, sampling settings, seed, timestamp. A run is a pure function of its inputs plus the models.
- **Python 3.11+, `pyproject.toml`, `uv`-friendly, typed, `pytest`.** Dependencies kept to a minimum (pydantic for the record schemas, httpx, typer or argparse for the CLI). Tests must not need a real model: use a fake adapter.
- **Cheap probe before full run.** Every verb must support `--limit N` and stream progress (one line per item as it completes) so I can watch partial results and stop early. No silent batch that reports only at the end.

## The three verbs and the file formats

Name: \*\*contrastset\*\* — folder, Python package, CLI command and GitHub repo are all \`contrastset\`, no "lib" suffix, no hyphens. Reserve the PyPI name with a placeholder release before writing code. CLI shape: `contrastset grade …`, `contrastset generate …`, `contrastset check …`.

### `grade` — census of what the subject model knows

Input: an entity list (JSONL: `{"entity": "Düren", "type": "town", "questions": [{"q": "Which country is Düren in?", "answers": ["Germany"]}]}`), plus a subject-model adapter. Output: one line per entity with a graded knowledge score and the raw evidence.

It must grade **behaviourally**, never by asking the model whether it knows something — the model's self-report is the variable under study and using it would be circular. Self-report may be collected as a separate, clearly labelled field for later comparison, never used for the score.

Output record (per entity):

- `entity`, `type`, provenance block
- `samples`: every raw completion, with the question, temperature and seed it came from
- `correct_rate`: fraction of samples judged correct
- `consistency`: agreement between samples regardless of correctness (same answer repeated = stable belief, scattered = guessing)
- `refusal_rate`: fraction of samples that decline or hedge
- `bin`: one of `known` (correct + consistent), `confidently_wrong` (consistent + incorrect), `unstable` (inconsistent), `refused`. Thresholds configurable, defaults documented.
- `self_report` (optional): what the model says when asked directly, kept separate

### `generate` — surface-matched pairs from a spec

Input: a **spec file** (YAML or JSON) and the graded entity file. Output: JSONL of pairs. The spec has at most four fields:

- `contrast`: the single variable allowed to differ, in plain words (e.g. "whether the subject model knows the entity")
- `examples`: three to five hand-written example pairs — preferred over a slot template, since a capable generator generalises better from examples than from a grammar
- `slots`: where entities come from — a graded file plus a selector, e.g. `{"a": {"bin": "known"}, "b": {"bin": "refused"}}`
- `constraints` (optional): forbidden words, length bounds, register, language

The generator prompt (what to look for, how to match surface form) is baked into the library once, versioned, and not something the user writes. Pair record: `pair_id`, `a`, `b`, the entity used in each, the contrast, the spec hash, generator provenance.

### `check` — leak detector for any pair file

Input: a pair JSONL (mine or anyone's). Output: a report (stdout table plus JSONL of per-pair flags) on whether pairs really differ only in the contrast. Cheap surface statistics, no models needed for v1:

- length in tokens and characters, per side, and the distribution of the difference
- vocabulary overlap between `a` and `b` (should be high apart from the slot)
- a position-aligned diff: does the difference sit only in the entity slot?
- near-duplicate detection across pairs
- similarity of every pair to the spec's `examples` — the generator copies example quirks, and that is a leak too
- a one-line verdict per pair plus an aggregate, with exit code non-zero above a threshold so it can gate a pipeline

## MVP scope and order of work

Build the ugly working version first, shaped by the real experiment, and only then polish the public interface. Do not design APIs for imagined users. Order:

1. **Record schemas and adapters.** Pydantic models for the three record types; the `complete()` interface; the OpenAI-compatible HTTP adapter; a fake adapter for tests. Point the HTTP adapter at my local Ollama/vLLM and prove a round trip.
2. **`grade`, minimal.** Behavioural grading with exact/normalised string match against the `answers` list, N samples at a fixed temperature, the four bins, streaming output. This is the hard verb — see the next section — but the first cut is deliberately dumb.
3. **A seed entity list.** A script (not part of the library) that pulls a few hundred candidate entities of one type from Wikidata — small towns with a country, minor historical figures with a birth year — mixing obscure and well-known ones so the graded file ends up with populated `known` and `refused` bins that look alike on the surface.
   - **Aliases are part of the answer list.** For every reference answer, pull the Wikidata label plus `skos:altLabel` aliases in English *and* in the entity's local language(s), plus the native-language label (e.g. Wupper → also "Wipper", its name in the upper reaches; Nouvelle-Aquitaine → also "New Aquitaine"). The string matcher only knows the names it is given, and the hand-written probe showed missing aliases grading correct answers as wrong — the matcher's weakness would otherwise masquerade as model ignorance. Keep the Wikidata QIDs (Q-number ids) of entity and answer in `meta` so aliases can be refreshed later.
   - Drop answers that occur as a whole word in the question or entity name (e.g. river Steyr for the town Steyr): containment matching would count any echo of the question as correct.
   - **"Country known" is the first filter**: `contrastset grade --gate country` asks the country question first and grades the rest only if it is known. Towns whose country the model doesn't know are unfamiliar entities, not useful for "fact unknown, entity familiar" pairs. Built: `scripts/wikidata_seed.py` (10 countries, sitelink-stratified, labels + aliases in English and the local language, echo filter on words and 5+ letter stems).
   - Ask the region question with the country and that country's own division name ("Which voivodeship of Poland is Kraków in?"). The bare "Which state or region is X in?" was answered with the country.
4. **`generate`, minimal.** Built as two stages: the generator writes entity-free frames (`[[X1]]`, `[[X2]]` placeholders, prompt `frames/v2`), the library fills both sides, so a and b differ only in the swapped slot by construction. Graded slots are (entity, question) pairs, selected by bin or by `correct_rate` range, and by default only from familiar entities (every other question known). Original wording:
    Spec loader, one versioned generator prompt, one generator adapter (online API), pairs written as JSONL. Two-clause sentences only for now: one known clause, one unknown, in both orders.
5. **`check`, minimal.** Length, overlap, slot-aligned diff, similarity to examples, exit code.
   - **Plus a subject-model behaviour check** (the composite probe, `scripts/probe_composite.py`, turned into a check through the check protocol): ask the subject model both sides of each pair and score every clause separately. A pair passes only if the known clauses are still answered and the unknown clause is still not. Surface statistics can't tell whether single-question knowledge survives inside a composite question; this can. Must sample with seeds disjoint from the ones the grade used (no double dipping).
6. **Run the real pipeline end to end** on Qwen3: grade → generate → check, with `--limit 20` first, then a few hundred. Fix whatever breaks. Only now write the README and tidy the CLI.

Definition of done for the MVP: `check` passes on a generated set of a few hundred known/unknown pairs, and the raw files are readable enough that I can hand them to a probing script in a separate repo. Everything after this is subsequent work.

## Grader design — the hard part

The grader quietly determines every downstream result, so its design choices must be explicit, configurable, and recorded in the output. The generator is a prompt in a loop; spend the design time here. Decisions to make, with my current defaults:

- **What "knows" means operationally.** Default: at least K of N samples match a reference answer AND the matching samples agree with each other. `known` requires both correctness and consistency; `confidently_wrong` requires consistency without correctness and is the most valuable bin, because it is where "knows" and "says it knows" come apart.
- **Sampling.** Default N = 8 samples per question at temperature 0.7, plus one greedy sample. Consistency is measured over the sampled answers; the greedy answer is reported separately. Record every sample.
- **Answer matching for free text.** v1: normalised exact match (case, whitespace, punctuation, articles) against the `answers` list, plus containment. Known weakness: paraphrases and partial answers. Add a `--judge` option later (a generator-side model judging equivalence) but never as the default, and never let the judge see the entity's bin.
- **Refusal detection.** v1: a small configurable phrase list ("I don't know", "I'm not sure", "no information"), language-aware. Report the matched phrase. This is crude and must be labelled as such in the output.
- **Prompt format.** The question must be asked the way the model will see it in the downstream experiment: same chat template, same system prompt or none. The grader takes a prompt template file and records its hash. Grading with one format and probing with another is a known source of mismatch.
- **Several questions per entity.** Knowledge is relational: a town can be "known" for its country and "unknown" for its river. Grade per question, aggregate per entity with the aggregation rule recorded (default: minimum over questions, i.e. known only if all are known).
- **Threshold sensitivity.** Ship a tiny report that shows how bin membership shifts as K and the consistency threshold move. If the bins are fragile, I need to know before building pairs on them.

Things the grader must not do: ask the model "do you know X" for scoring; use the generator model to decide what the subject model knows; collapse the raw samples into a single number without keeping them.

## Subsequent work (after the MVP ships)

In rough priority order; each is its own small PR, none blocks the MVP:

1. **Composition specs.** Extend `generate` beyond two clauses: questions built from several controlled known/unknown clauses, with the clause structure recorded per pair, so I can test whether the uncertainty readout behaves like a sum, product or minimum of clause-level values.
2. **Multiple generators and hold-out.** Generate the same spec with two different generator models and tag each pair with its source, so a probe that only works on one generator's style is caught. `check` gets a cross-generator comparison.
3. **Judge-based answer matching** as an opt-in for `grade`, with the judge's verdicts stored beside the string-match verdicts, never replacing them.
4. **Model-based leak checks in `check`.** Optional: a small classifier or a generator-side model asked to guess which side of each pair is the "unknown" one from surface form alone. If it can, the set leaks.
5. **Subject-model self-report comparison.** A report comparing graded knowledge against what the model claims when asked directly — a result in its own right (calibration of self-knowledge).
6. **Cross-model entity sets.** Grade the same entity list against Qwen3, Llama and Gemma and emit the intersection and differences, so later cross-model experiments (transferring directions via affine maps between residual streams) use entities whose status is known in every model.
7. **Alignment with existing work.** A built-in recipe reproducing the known/unknown entity setup from the 2024 entity-recognition interpretability papers, so my pairs can be compared against theirs directly.
8. **Packaging.** PyPI release, docs, a `cookbook/` with the two-clause and composition specs as worked examples.

## How to work with me in this session

- Start by proposing the package layout, the three record schemas and the adapter interface as a short plan. Push back on anything above that you think is wrong or over-specified; I prefer disagreement to agreement. Then build step 1 of the MVP order.
- Ask before any step that is expensive to redo (schema shape, CLI shape). Don't ask about things with a sensible default; pick it and say so.
- Keep each step small and runnable. After each step, show me the command to run it and what output to expect. Run the tests.
- Use a fake adapter for all tests. When you need my real local model, tell me which endpoint and model name to set and I'll provide them.
- Define every abbreviation once in code comments and the README. Assume I know software engineering well and ML jargon poorly.
- When something in the design turns out to be wrong in practice, say so and propose the change rather than quietly working around it.
