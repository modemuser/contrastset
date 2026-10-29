# Experiments

Research specs and inputs, not examples. Results are in `../FINDINGS.md`.

## Templates (grade and probe with the same string; `check` and the probe scripts enforce it)

| template | use | graded files |
|---|---|---|
| `templates/qwen3-name-only.txt` (B) | first probe: knowing, no escape hatch | `runs/wikidata-qwen3-8b-hf-logprob-B.jsonl` (HF Qwen3-8B, log-prob, 294 towns) |
| `../examples/03-two-clause-rivers/template.txt` (A) | kept for the later refusal-direction experiment | `runs/wikidata-qwen3-8b-fp16.jsonl` (Ollama 8B-fp16, sampled, 294 towns; old shared-seed scheme), `runs/wikidata-qwen3-14b.jsonl` (Ollama 14B) |

A invites "I don't know" and so creates the refused cell (and some self-report);
B asks for the name only. See FINDINGS, "The refusal invitation creates the refused cell".

## Files
- `wikidata-towns.jsonl`: 294 towns from `scripts/wikidata_seed.py`.
- `position-*.yaml`, `three-clause-position.yaml`: Experiment 1 (position effect).
