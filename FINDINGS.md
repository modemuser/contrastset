# Findings

Results that came out of building the tool, kept with the data that produced them.
Each entry says what was measured, on what, and how far it can be trusted.
The `runs/` files cited below are committed (graded files, pair files, probe outputs and
the logs quoted); other run outputs are not. Scripts are in `scripts/`, inputs in
`experiments/` and `examples/`.

## 2026-10-28 · An unknown neighbour costs a known clause nothing a known neighbour doesn't

**Question.** In a two-clause question, one clause the model can answer and one it
can't: does the unknown clause degrade the answer to the known one?

**Setup.** Subject: qwen3:14b (Q4_K_M, via Ollama raw mode, template `qwen3-nothink`).
60 pairs from `examples/03-two-clause-rivers/spec.yaml`: side a asks the river of two
towns graded known, side b swaps one of them for a town graded unstable; the other
("shared") town, the frame and the clause positions are identical across the two
sides. Each side asked with 1 greedy + 4 sampled completions (T = 0.7). Each clause
scored separately by string match against its reference rivers.
Data: `runs/pairs-two-clause-v1.jsonl`, `runs/composite-two-clause-v1.jsonl`,
probe script `scripts/probe_composite.py`.

**Result.** The clean comparison is the shared clause within a pair: same town, same
frame, same position, same prompt; only its neighbour changes from known to unknown.

> KU − KK = **+0.05** correct answers out of 5 (sd 1.35, n = 60 pairs);
> 43 pairs identical, 9 better, 8 worse.

Pooled over all known clauses the picture is the same:

| known clause                         | correct          |
|--------------------------------------|------------------|
| next to another known clause (KK)    | 517/600 = 0.862  |
| next to an unknown clause (KU)       | 260/300 = 0.867  |
| KU, unknown clause second            | 132/150 = 0.880  |
| KU, unknown clause first             | 128/150 = 0.853  |

**Reading.** An unknown neighbour costs the known clause nothing that a known
neighbour doesn't. This leans against hypothesis (c), that uncertainty is attention
mass leaving the unknown clause: if attention were being pulled towards or away from
the unknown clause, the known neighbour should pay more in KU than in KK.

**Weak evidence, because:**
- It measures accuracy, the argmax answer. Attention could shift and lower the
  model's confidence in the known answer without changing which answer wins. Next:
  the probability of the correct known-clause answer, KK vs KU; needs logits, so the
  HF (Hugging Face) transformers adapter, not Ollama.
- Small: 60 pairs over 13 known and 10 unstable towns, 5 completions per side.
- The "unknown" clause is `unstable`, which includes half-known towns (single-
  question correct rate 0.5–0.62); rerun on strictly selected pairs.
- One model, one fact type (rivers of European towns), one prompt template.

**Aside, confounded: known clauses do worse in composites than alone.** Against the
same towns' single-question rate (24 fresh samples) a known clause drops from about
0.98 to 0.86–0.87, equally in KK and KU. This cannot be read as a cost of composition:
the single question and the composite are different prompts (one question versus two
under an instruction written for one, "Answer with just the name"), so "two clauses"
and "different prompt" are mixed. What the errors do show is where the model slides
under load: towards a default big river (Monschau → "Rhein", Mértola → "Tagus", Ptuj →
"Sava"), the same pattern as Cochem → "Rhine" in single-question grading.

**Anomaly: the neighbour sometimes helps a weakly known town, in the second
clause.** Tábor's river is 0/24 asked alone but 8/30 in composites, and all 8
successes have Tábor in the second clause (second 8/15, first 0/15; successes with
partners Tolmin and Wetzlar, none with Uppsala). Düren shows the same direction
(first 3/10, second 9/10). Over all unknown clauses: second 41/150 = 0.27, first
30/150 = 0.20. The towns that are truly unknown alone (Kežmarok, Písek, Ronda,
Bad Münstereifel, Wels) stay at 0/190 in either position, so the help reaches only
towns with some latent knowledge. The geographic-leak explanation (a Czech or nearby
partner priming Czech rivers) is untested: none of Tábor's partners here were Czech.
Same-country versus different-country partners is now a controlled factor in the
spec (`slots.factors`), so the next run separates the two.

*Not a format artefact.* Tábor's first-position answers are well-formed, confident
wrong rivers: "Vltava" 9 times (the region's default big river; also its modal
answer alone, 0/24), "Svitava", "Erbach", plus 4 refusals all from one frame ("Does
the river through ... differ"). In second position: "Lužnice" 8 times. Düren alike:
first "Rhein"/"Rhine"/"Erft", second "Rur" 9/10. The matcher isn't failing on form;
the model reaches for the default river first and the specific one later. The
partner matters too: next to Uppsala, Tábor is wrong in both positions.

*Reading (Thomas).* A threshold being nudged, not knowledge created: a fact near the
decision boundary is pushed over by the state the model is in after producing an
answer. That is what a graded per-clause confidence looks like from outside; a
binary known/unknown would not move with position. Encouraging for the readout
hypothesis, pending two checks:
- format: done, above, ruled out;
- recency (second position is also nearest to where the model starts writing):
  `experiments/three-clause-position.yaml` rotates the swapped clause through
  first, middle and last with the same towns and frame. Middle like last: having
  answered helps. Middle like first: recency. Pairs generated
  (`runs/pairs-three-clause-v1.jsonl`, frames/v3); subject run pending.

*Consequence for selection.* Half-known facts are a resource: excluded from clean
contrast sets, selectable as a correct_rate middle band. But Tábor is not in the
middle band (0/24 alone), so a sampling rate understates near-boundary knowledge:
the probability of the correct answer (HF adapter) is the better selector for it.

## 2026-10-28 · 8 samples are enough at the extremes, not in the middle

**Question.** Are the strict thresholds (known ≥ 7/8, unknown ≤ 1/8) real, or do
borderline towns just regress to the mean when re-sampled?

**Setup.** The 29 probe towns re-graded with 24 samples at seed 1000, so that none
of the 24 overlaps the original 8 (seed 0; sample i uses seed + i).
Data: `runs/probe-v2-qwen3-14b.jsonl` (n = 8), `runs/probe-v2-n24-qwen3-14b.jsonl` (n = 24).

**Result** (river question; shift = rate at n = 24 minus rate at n = 8):

| n = 8 band  | towns | mean shift | mean abs shift |
|-------------|-------|------------|----------------|
| ≤ 1/8       | 11    | +0.02      | 0.03           |
| middle      | 7     | +0.10      | 0.18           |
| ≥ 7/8       | 11    | +0.02      | 0.02           |

The extremes are stable; the middle is noise at n = 8 (Düren 0.38 → 0.75, Coimbra
0.62 → 0.88, Bernkastel-Kues 0.50 → 0.29). Country bins: 0 of 29 change; region 2.

More samples do not predict composite behaviour better (correlation of single-
question rate with composite rate over 23 towns: n = 8 r = 0.91, n = 24 r = 0.88), and
strict selection on the 24-sample rates gives 21/28 valid pairs against 24/34 on the
8-sample rates. So the remaining ~25 % invalid pairs are not selection noise: they are
known towns that slip inside composites (the confounded aside above) plus a few
weakly known towns answered in context (Tábor 0/24 alone, 8/30 in composites).

**Consequence.** Screen with n = 8 and the strict thresholds; do not select from the
middle band. The decisive filter is the behavioural check on the pairs themselves
(MVP step 5), not a larger single-question sample.

## 2026-10-28 · Methods note: every grade so far sampled with top_k = 20, top_p = 0.95

Ollama applies the Modelfile defaults for qwen3:14b (temperature 0.6, top_k 20,
top_p 0.95) unless a request overrides them. The grades passed only temperature
(0.7), so all samples above came from temperature 0.7 *with* top-k 20 and nucleus
(top-p) 0.95 truncation, while provenance recorded only the temperature. The
results stand; the record was incomplete. Fixed: the Ollama adapter now records the
server's sampling defaults in `provenance.model.server_sampling_defaults`, and
top_k / top_p can be set explicitly. The HF adapter does not apply a model's
generation_config defaults silently (pure temperature sampling unless set) and
records the defaults it ignored. Comparing grades across adapters requires
matching these settings explicitly.

## 2026-10-28 · Experiment 1 (probe): no second-clause advantage for half-known towns

**Question.** Are half-known facts retrieved more often in the second clause (the
Tábor/Düren anomaly above)?

**Setup.** Wikidata towns graded on qwen3:14b (n = 8, gate country), river
correct_rate 0.2–0.7, `familiar` on (country and region known): 14 towns, not the
20 asked for (the band has 24 without `familiar`). Each paired with one strictly
known town (≥ 7/8), side a (known, middle), side b (middle, known), the 8 frames/v4
frames of example 03. 1 greedy + 4 sampled completions per side, seeds from 10000.
Control (a): three clauses, the middle-band town rotated through first/middle/last,
each position twice per town. Data: `runs/pairs-position-*-v1.jsonl`,
`runs/composite-position-*-v1.jsonl`, `scripts/analyse_position.py`.

**Result.**

| position | two clauses | three clauses |
|---|---|---|
| first | 42/70 = 0.60 | 78/140 = 0.56 |
| middle | – | 64/140 = 0.46 |
| second / last | 38/70 = 0.54 | 76/140 = 0.54 |

Two clauses: 6 towns better second, 6 better first, 2 equal (sign test p = 1.0).
Known partners unaffected (62/70 vs 61/70). Per-town swings are large (Imst 5/5
first, 2/5 second) at 5 completions, so per-town numbers are noise-level.

**Control (b), first-position answers of ten towns:** 35 correct, 15 wrong rivers,
0 hedged, 0 malformed, 0 empty. Not a format artefact. Two matcher artefacts found
while reading them: "Lot-et-Garonne" (a département) counts as correct for Floirac
because it contains "Garonne"; "Innsbruck" (a town) given as Kufstein's river.

**Reading.** The second-clause advantage seen for Tábor (8/15 vs 0/15) and Düren
does not generalise to the middle band: no effect in two clauses, and in three no
"later is better" (middle is lowest, if anything). The earlier anomaly was two towns
and one partner each; treat it as partner-specific until shown otherwise. Not
scaled, per the plan (scale only if the effect holds).

## 2026-10-28 · Parity Ollama vs HF (Qwen3-8B): same answers, different sampler

**Setup.** The 29 probe towns graded twice with the same weights family and
precision class: Ollama `qwen3:8b-fp16` (llama.cpp, F16) and HF `Qwen/Qwen3-8B`
(transformers on MPS, bf16). Same template file (example 03), n = 8, T = 0.7,
top_k 20, top_p 0.95 set explicitly on both, seed 0. Data:
`runs/parity-ollama-8b-fp16.jsonl`, `runs/parity-hf-qwen3-8b.jsonl`,
`scripts/compare_grades.py`.

| question | strict bins agree | greedy answer agrees | correct_rate r | mean distinct answers / 8 samples (Ollama, HF) |
|---|---|---|---|---|
| country | 28/29 = 97% | 29/29 | 1.00 | 1.10, 1.10 |
| region | 26/29 = 90% | 29/29 | 0.97 | 2.03, 1.48 |
| river | 23/29 = 79% | 28/29 (the 29th: "San river" vs "San") | 0.97 | 3.03, 2.21 |

**Reading.** No template or tokenisation mismatch: the same prompt gives the same
greedy answer and the same correct rate. The river disagreements are all one way,
unstable (Ollama) -> confidently_wrong (HF): HF's samples are less diverse, so the
modal wrong answer crosses the 0.75 consistency threshold (Ronda 0.38 -> 0.88,
Písek 0.62 -> 1.00). Most likely cause: sampler order. transformers applies the
temperature before top-k/top-p, so top_p = 0.95 cuts a sharpened distribution and
keeps fewer candidates; llama.cpp truncates the untempered distribution first and
applies the temperature last. Same settings, different procedure.

**Consequence.** The unstable / confidently_wrong boundary depends on the backend's
sampler, not only on the model. Bins must not be compared across backends until the
sampling procedure is matched. Proposed: the HF adapter applies top-k / top-p to the
untempered logits and the temperature after them (llama.cpp order), then re-run.

**Update, after the re-run: the sampler-order explanation was wrong.** With the HF
adapter switched to the contrastset order (truncate untempered, then temperature),
689 of 696 HF samples are identical to before: at top_k = 20 the kept set barely
changes. Parity: country 97 %, region 90 %, river 83 % (24/29). What is established
(`scripts/diag_backends.py`, runs/diag-*.json):

- same prompt: identical prompt token counts on both backends (38/38, 39/39, 42/42);
- same distribution: top-20 next tokens overlap 19-20/20, probabilities within 0.03
  (fp16 vs bf16 numerics);
- Ollama samples exactly as the contrastset order predicts: 200 first-token draws
  (Steyr) and 200 mid-word draws (Ronda after "Guadal", total variation 0.017);
- inside transformers' generate(), exactly the intended warpers apply (checked on
  a model with extreme generation_config defaults, which are ignored);
- torch.multinomial on MPS is unbiased (20 000 draws, total variation 0.003).

Not the repetition penalty (Thomas's hypothesis: Ollama's default 1.1 would push
mass to tail tokens mid-word). The Modelfile sets repeat_penalty 1, and measured:
mean distinct answers per 40 draws over five river prompts, explicit 1.0: 7.20,
unset (as in the grades): 7.20, explicit 1.1: 7.80. The penalty was off in the
grades; turned on, it does raise diversity, so the mechanism is real but not active.

Yet HF's whole answers are less diverse than Ollama's, systematically: per town,
Ollama more diverse in 16 vs HF in 2 (river, sign test p = 0.001), 9 vs 0 (region,
p = 0.004). Ollama's extra answers are long-tail multi-token names ("Guadalhambrilla",
"Mžeřížka"). Not yet tested: HF's empirical draws from the 8B model itself.

**Resolved: shared seeds, not the backends.** 200 full answers per town on each
backend (`scripts/diag_positions.py`, Ronda and Písek) agree at every token position
(total variation 0.02-0.09 at tokens 1-4; "Guadalhorce" 145 vs 145, "Vltava" 173 vs
169). The parity gap came from the grade's seeding: every question used seeds
seed + 0..7, so all 87 questions on a backend shared the same 8 random streams.
Ollama's streams 0-7 happen to be an unusually diverse block (Ronda: 5 distinct
answers vs a mean of 2.84 over the 25 blocks of 8 seeds; Písek 4 vs 2.12); HF's
are ordinary. One lucky block moved every bin in the file the same way, and the
per-town sign test (p = 0.001) counted that one draw of luck 29 times. Fixed: each
question's seeds now start at seed + sha256(entity, question id) mod 1e8 (the
behaviour check likewise per pair and side). Earlier sampled grades are internally
correlated in this way; their per-question rates are still valid estimates, but
differences between two such files are not independent across questions.

**sampled vs logprob bins (HF, same file):** country 97 %, region 66 %, river 83 %.
Two causes. (1) A bug, fixed after this run: answer forms were made prefix-free as
strings, so "South Bohemian Region" was dropped because "South Bohemia" is its
string prefix, though not its token prefix; Tábor's p_correct came out 0.09 while
its greedy answer, a correct form, had 0.84. Now compared on token ids. (2) Expected:
logprob masses are untempered (T = 1) while samples are drawn at T = 0.7 with
truncation; for multi-token answers ("Rhineland-Palatinate") per-token uncertainty
compounds at T = 1, so p_correct sits below the sampled rate (region means 0.65 vs
0.81). Single-token answers (countries) agree 97 %.

## 2026-10-28 · Log-prob grading must sum over token segmentations

> **Log-prob grading over a single tokenization undercounts multi-token answers by
> up to 26×** (Koblenz, "Rhineland-Palatinate": 0.031 canonical vs 0.823 summed over
> segmentations, Qwen3-8B). Checkable with `HFAdapter.prefix_mass` vs `score`.

**Observed.** First HF `--logprob` grade of the 29 towns: sampled and log-prob bins
disagreed most on long names, and log-prob masses at T = 1 and at the sampling
distribution (T = 0.7, top-k/top-p) had the same means (region 0.71 vs 0.72), both
below the sampled rate (0.81). So the gap was **not** the temperature, as claimed in
the parity entry above ("expected, not a bug"): that explanation was wrong.

**Cause.** Koblenz, region: greedy answer "Rhineland-Palatinate" (7 of 8 samples).
The model generates it as `Rh·in·eland·-P·al·at·inate` with probability 0.799; the
tokenizer re-segments the same string as `R·hin·eland·...`, which has probability
0.031. The probability of a TEXT is the sum over every token sequence that spells
it, and models often write rare names in a non-canonical segmentation. Scoring the
canonical segmentation alone missed 96 % of the mass here; for one-segmentation
answers ("Alentejo", 0.981 both ways) nothing changes.

**Fix.** `HFAdapter.prefix_mass` expands, step by step, EVERY token consistent with
the rest of the target text (byte trie over the vocabulary, no top-k), summing
finished paths (a token may overshoot the end, so "Mur" includes "Mura..."). The
only approximation is mass pruning: a partial path is dropped below probability
1e-5, and the dropped mass is reported as an upper bound (`p_correct_pruned_max`,
`p_modal_pruned_max`; the true value is in [p, p + bound]). A first version also cut
candidates to the top 200 tokens and the beam to 16 paths; a beam width biases
against strings with many spellings, so it was replaced before any grade relied on
it. Checked against brute-force enumeration on a toy vocabulary with overlapping
tokens, unpruned and fully pruned (bound = exactly the dropped mass). Koblenz now
0.823 at T = 1 and 0.931 under the sampling distribution, matching 7/8 sampled.
Refusal phrases are still scored in the canonical segmentation (everyday English,
where the model's segmentation is the canonical one); a known limitation.

Also fixed in the same run: answer forms were made prefix-free as strings before
scoring, dropping "South Bohemian Region" behind "South Bohemia"; with marginal
masses, string prefixes now nest exactly.

## 2026-10-28 · Log-prob vs sampled grades, recomputed with segmentation sums

**Setup.** 29 probe towns, HF Qwen3-8B, `grade -n 0 --logprob` (masses summed over
token segmentations; pruning bound max 3.5e-5, mean 2.5e-6 over all questions),
against the sampled n = 8 grades (HF and Ollama; old shared-seed scheme, so their
noise is correlated across questions). Data: `runs/parity-hf-logprob-29.jsonl`,
`scripts/compare_logprob.py`.

| question | vs HF samples: bins (T=0.7) | r | slope | mean p(T=0.7) vs sampled |
|---|---|---|---|---|
| country | 28/29 | 1.00 | 0.98 | 0.99 vs 0.99 |
| region | 25/29 | 0.96 | 0.99 | 0.81 vs 0.81 |
| river | 26/29 | 0.97 | 0.97 | 0.33 vs 0.38 |

(T = 1: bins 28 / 24 / 25 of 29, slopes 0.78-1.03.) The relationship is one-to-one
(slope ≈ 1, r ≥ 0.96): what disagreement remains is outliers, not a slope. Region,
0.71 vs 0.81 before the segmentation fix, now matches (0.81 vs 0.81).

**The outliers have three causes.**
1. Spellings without diacritics ("Soca River", "Swietokrzyskie"): the sampled
   matcher folds accents, the log-prob forms were exact bytes. Fixed after this run:
   forms now include accent-folded variants.
2. Answers that do not START with the name ("River Fyris", "Yes, the river that
   flows through Uppsala is the **Fyrisån**"): by definition p_correct is a prefix
   mass, and the sampled matcher matches anywhere. Uppsala river 0.00 vs 0.38.
   Documented, not fixable by enumeration.
3. Sampling noise at n = 8 (Ptuj 0.24 vs 4/8).

**Side finding: the template invites self-report.** "If you do not know, say 'I
don't know'" makes the model open with "Yes, I know. The river that flows through
Uppsala is ..." (HF greedy answer for Uppsala). Statements about its own knowledge
then sit in the very outputs whose internals the probe reads.

## 2026-10-28 · The refusal invitation creates the refused cell, and hides knowledge

**Question.** Does the refused cell survive without "If you do not know, say 'I
don't know'" in the template? (It also makes the model open with "Yes, I know",
a self-report at the positions the probe reads.)

**Setup.** Ollama qwen3:8b-fp16, n = 8, T = 0.7, top_k 20, top_p 0.95, per-question
seeds identical for both templates (paired). Template A: examples/03 (with the
invitation). Template B: `experiments/templates/qwen3-name-only.txt` ("Answer with
the name only.", no invitation). Same refusal-phrase list in both. Two pools: the
29 probe towns, and the 49 Wikidata towns the 8B grade had refused on river or
region. Data: `runs/template-{A,B}-{29,refused}-ollama.jsonl`,
`scripts/compare_templates.py`.

**29 probe towns.** A: 2 refused bins, 16/696 samples with a refusal phrase, 6
self-report openings, 2 correct answers not starting with the name. B: all zero.

**Refused pool (49 towns), river:**

| | A | B |
|---|---|---|
| refused bins | 45 | 0 |
| samples with a refusal phrase | 341 / 392 | 0 / 392 |
| mean correct_rate | 0.01 | 0.13 |

A's 45 refused rivers under B: unstable 29, confidently_wrong 11, known 5 (e.g.
Albalat de la Ribera: "Júcar" 6/8). Region: refused 3 -> 0, correct rate unchanged
(0.50 / 0.51).

**Reading.** The refused cell was not a model state; it was the instruction
working: its membership is set by a line in the prompt, so it cannot be what a
probe is about. The refusal instruction also suppresses answers on towns where the
model has partial knowledge, including some it answers reliably without it (5 of
A's 45 refused rivers looked known under B at n = 8). The 294-town log-prob grade
(next entry) puts a number on it: of A's 45 refused rivers, 3 are known under B
(p_correct at T = 1: Malgrate 0.90, Monza 0.84, Kufstein 0.81), 2 are half-known
(Albalat de la Ribera 0.54, Carcaixent 0.56, both "Júcar"), 4 more sit at
0.16–0.24, and 36 are at or below 1/8. Carcaixent's 8/8 was luck: at the sampler
distribution (T = 0.7) its p_correct is 0.73, and 0.73^8 is about 0.08. So: 3 of 45
refused rivers (about 1 in 15) are ones the model answers reliably without the
invitation, and 9 of 45 have p_correct above 1/8. Without the invitation,
unknown facts show up as guessing (unstable) or consistent wrong answers, part of
them default guesses (Seehausen -> Weser 6/8), and no self-report.

**Decision (Thomas).** Template B for grading and probing. The first probe's three
cells are known, unstable (correct_rate <= 1/8) and prior-guess, all defined by what
the model does with no escape hatch.

**What B gives up.** With no refusal anywhere, the "about to say I don't know" axis
cannot be tested under B. Template A stays, with its graded files
(`runs/wikidata-qwen3-8b-fp16.jsonl`, Ollama, 294 towns; `runs/template-A-*`): if a
knowing direction turns up under B, a second experiment under A on the same towns
tests whether refusing is a separate direction or the same one. The template-hash
rule keeps the two from being mixed.

## 2026-10-29 · 294 towns, template B, log-prob grade: the three cells

**Setup.** HF Qwen/Qwen3-8B bf16 (revision b968826), template B
(`experiments/templates/qwen3-name-only.txt`), `-n 0 --logprob`, sampler T = 0.7,
top_k 20, top_p 0.95, gate country; then `--priors --prior-source logprob
--prior-group answer.country --prior-min 4`. 3 h 46 min on the M1 Pro (21:44–01:30). Data:
`runs/wikidata-qwen3-8b-hf-logprob-B{,-priors}.jsonl`, logs `runs/*-hf-logprob-B.log`,
`runs/priors-hf-B.log`. Bins and correct_rate below are at T = 1 (the model's own
distribution); the sampler distribution is given where it differs.

**Gate.** 252 of 294 towns pass the country question; 42 stop there.

**Question bins (252 gated towns):**

| | known | confidently_wrong | unstable | refused |
|---|---|---|---|---|
| region | 89 | 34 | 129 | 0 |
| river | 48 | 55 | 149 | 0 |

correct_rate bands (<= 1/8 | middle | >= 7/8): region 108 | 65 | 79, river 163 | 52 | 37.

**No refusals at all.** No greedy answer contains a refusal phrase, and the largest
refusal mass on any question is 0.045. Under B the refused cell is empty, as the A/B
run predicted.

**Prior guesses.** 88 (entity, question) slots where the wrong modal answer is one
shared by at least 4 towns of the same country: rivers Vltava x15, "Västergötland
river" x9, Váh x8, Ebro x6, Tagus x6, Dordogne x5; regions South Bohemian x9,
Västmanland x7, Lublin x5, Banská Bystrica x5, Emilia-Romagna x5, Vila Real x4,
Carinthia x4. Of the 88, 60 bin as unstable and 28 as confidently_wrong (at T = 0.7:
46 / 42, because sharpening the distribution turns a leading default into a
consistent one).

**The three cells, made disjoint** (prior-guess takes precedence over unstable):

| | known | unstable, p_correct <= 1/8, not a prior guess | prior-guess |
|---|---|---|---|
| region | 89 | 55 | 39 |
| river | 48 | 84 | 49 |
| total | 137 | 139 | 88 |

At the sampler distribution (T = 0.7): known 99 + 59, unstable 47 + 62, prior-guess
39 + 49. The cells are large enough for a first probe at either temperature; T = 1
is the one to report, since it is the model's distribution rather than ours.

**Pruning bound.** The largest dropped mass on any p_correct is 1.4e-4 (T = 1) and
2.7e-8 (T = 0.7); on p_modal 3.5e-5. No bin or band can change because of pruning.

**Not checked.** Prior guesses are defined by country and answer string only. A
shared wrong answer can still be partly right (the Vltava catchment, a region one
level up), so before the probe, the 13 prior answers should be read once by hand.

## 2026-10-29 · The country-shared prior flag catches country defaults, not basin knowledge

Follow-up to "Not checked" above, done offline in `../knowing-probe`
(`scripts/river_cells.py`, FINDINGS there). Wrong modal river answers were checked
against Wikidata P403 mouth chains (does the gold river flow into the answer, any
number of hops?) and against a basin check (is the same answer also the modal answer
for same-country towns outside that basin?). Of the river slots carrying `prior_guess`
(familiar, p_correct ≤ 1/8), 12 name a river downstream of the truth, and all 12 fail
the basin check. Vltava is the modal answer for 16 Czech towns, 10 of them outside its
basin (six on the Elbe). Váh: 6 of 9 outside. Dordogne: 4 of 6. So `--priors` with
`answer.country` grouping finds country defaults, some of which sit downstream of the
town by coincidence. Basin-level answers that pass the check (19, e.g. Leine for an
Ilme town, Neckar for a Jagst town) are all unflagged: each is shared by too few towns
to reach `--prior-min 4`. A river-aware prior rule needs the mouth chain, not the
country.
