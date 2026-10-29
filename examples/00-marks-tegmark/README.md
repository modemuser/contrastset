# 00 · Marks & Tegmark: does the probe stage work at all?

A sanity check on public data, before you trust a probe on your own contrast set.
If a difference-of-means probe can't separate plain true and false statements, a
null result on your own pairs tells you nothing.

Reproduces the first result of Samuel Marks and Max Tegmark, *The Geometry of
Truth: Emergent Linear Structure in Large Language Model Representations of
True/False Datasets* (2023, [arXiv:2310.06824](https://arxiv.org/abs/2310.06824)),
on Qwen/Qwen3-8B (bf16, revision `b968826`): true and false statements separate
along one direction in the residual stream from the early-middle layers on.

Two of their statement sets, from
[saprmarks/geometry-of-truth](https://github.com/saprmarks/geometry-of-truth) at
commit `5d1c630`:

- `cities`: 1496 statements, "The city of Krasnodar is in Russia." and the same
  city with a wrong country; 748 cities, one true and one false statement each.
- `sp_en_trans`: 354 statements, "The Spanish word 'uno' means 'one'.", true or
  false; 344 Spanish words.

The CSVs are not committed (the upstream repository has no licence file);
`fetch_data.py` downloads them at the pinned commit and checks their sha256.

## Result

Held-out AUC (area under the ROC curve: the chance that a random true statement
projects higher than a random false one; 0.5 is chance) of the arrow
mean(true) − mean(false), fitted on training statements, per layer. Layer 0 is
the embedding output, layer i the output of block i (36 blocks).

| layer | cities | sp_en_trans |
|---|---|---|
| 1–3 | 0.49–0.52 | 0.49–0.58 |
| 4 | 0.87 | 0.71 |
| 8 | 0.94 | 0.99 |
| 12 | 0.96 | 1.00 |
| 15 | 1.00 | 1.00 |
| 20–35 | 0.99–1.00 | 1.00 |
| 36 | 0.985 | 1.00 |

Held-out AUC is ≥ 0.94 by layer 8 and reaches 1.00 (rounded) from layer 15 for
`cities` and 13 for `sp_en_trans`. Layer 0 gives a zero arrow: every statement
ends in the same "." token, so its embedding is the same for every item, which
confirms that the read position is the last token. All 37 layers are in
`expected-auc.jsonl`.

**The shuffled-label control does not fall to chance, and that is expected.** With
the training labels shuffled, the held-out AUCs pile up near 0 *and* near 1 in
the middle layers instead of sitting around 0.5. From about layer 9 on, the
true/false direction is the first principal component (PC1, the direction of
greatest variance) of the activations: |cos(arrow, PC1)| reaches 1.00. A shuffled
arrow is the difference between the means of two random halves, which points
along the high-variance directions, so it lands on PC1 with a random sign.
The share of shuffled AUCs above 0.5 stays near one half at every layer, so
there is no leak. Read the control by that sign fraction and by the permutation
p-value, not by a band around 0.5 (`auc.py --shuffles 200` reports both).

## Files

| file | what |
|---|---|
| `fetch_data.py` | downloads `data/cities.csv`, `data/sp_en_trans.csv` (pinned commit, sha256 checked) |
| `split.json` | the frozen split: a third of the cities (249) and Spanish words (115) held out, seed 20261029; both statements about an item stay on the same side |
| `auc.py` | activations at the last token, every layer, then the per-layer AUC on the frozen split, as JSONL |
| `make_pairs.py` | `cities` as a contrastset pair file (a = true, b = false, same city) for `contrastset check` |
| `expected-auc.jsonl` | the reference run's per-layer numbers, incl. the shuffle control and |cos(arrow, PC1)| |

## Run it

```bash
pip install 'contrastset[hf]'          # or: uv sync --extra hf
python fetch_data.py
python make_pairs.py data/cities.csv | contrastset check - --checks length,overlap,slot_diff,duplicates
python auc.py cities > auc-cities.jsonl              # ~7 min on an M-series Mac (MPS), 0.28 s/statement
python auc.py sp_en_trans > auc-sp_en_trans.jsonl    # ~2 min
```

`check` confirms that all 748 pairs differ in one slot, the country, and nothing
else (`slot_diff` ok on every pair). `length` flags 47 of them (38 warn, 9 fail)
because some country names are up to five words longer than others, so `check`
exits 1 here. Read a checker's verdicts before acting on its exit code: here the
flags describe the dataset, not a bug.

The opt-in smoke test (`pytest -m smoke`, see the main README) runs these steps
and asserts held-out AUC ≥ 0.95 at every layer from 20 to the last.

The code is a trimmed copy of step 0 in the knowing-probe experiment, the first
consumer of contrastset. That experiment keeps its own copy and results.
