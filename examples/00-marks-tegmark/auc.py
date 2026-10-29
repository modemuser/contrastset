"""Per-layer held-out AUC of a difference-of-means probe on true vs false statements.

Usage: python auc.py cities|sp_en_trans [--limit N] [--shuffles N] > auc-cities.jsonl
Needs `pip install contrastset[hf]` and data/<dataset>.csv (fetch_data.py).

What it does, per dataset:
1. Feed each statement to Qwen/Qwen3-8B as plain text (no chat template, no special
   tokens) and read the residual stream at its last token (the full stop) at every
   layer. Residual stream: the running vector each transformer block reads from and
   adds to. Layer 0 is the embedding output, layer i the output of block i, read with
   forward hooks, so the last layer is also taken before the model's final norm.
2. Split by item (city or Spanish word) with the frozen split in split.json: a third
   of the items held out, both statements about an item on the same side.
3. Arrow = mean(true) - mean(false) over the training statements, per layer (Marks
   and Tegmark's mass-mean probe without the covariance correction). Project every
   statement onto it, rescaled so the training false mean sits at 0 and the training
   true mean at 1; threshold 0.5.
4. AUC (area under the ROC curve): the chance that a random true held-out statement
   projects higher than a random false one; 0.5 is chance, 1.0 perfect separation.

Writes one JSON line per layer to stdout, progress to stderr. --shuffles N adds the
label-shuffle control: arrows refitted on training labels permuted N times.

This is a copy of step 0 in knowing-probe (scripts/mt_acts.py, scripts/mt_probe.py and
the src/knowing_probe modules they use), trimmed to what the AUC needs: no plots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from contrastset.adapters.hf import HFAdapter

HERE = Path(__file__).parent
MODEL, REVISION = "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"


def item_of(dataset: str, row: dict[str, str]) -> str:
    if dataset == "cities":
        return row["city"]
    m = re.match(r"The Spanish word '(.+?)' means", row["statement"])
    if not m:
        raise ValueError(f"cannot find the Spanish word in {row['statement']!r}")
    return m.group(1)


def split_groups(groups: Sequence[Hashable], held_out_frac: float, seed: int) -> np.ndarray:
    """Boolean mask, True = held out. Whole groups (cities, words) go one way."""
    uniq = sorted(set(groups), key=str)
    rng = random.Random(seed)
    rng.shuffle(uniq)
    held = set(uniq[: round(len(uniq) * held_out_frac)])
    return np.array([g in held for g in groups])


def read_last_token(adapter: HFAdapter, texts: Sequence[str], label: str) -> tuple[np.ndarray, list[str]]:
    """[n_texts, n_layers, d_model] float32 residuals at each text's last token, and those tokens."""
    torch = adapter._torch
    blocks = adapter.model.model.layers
    captured: dict[int, Any] = {}

    def hook(i: int) -> Any:
        def fn(_module: Any, _inp: Any, out: Any) -> None:
            h = out[0] if isinstance(out, tuple) else out
            captured[i] = h[0, -1].float().cpu()
        return fn

    def embed_hook(_module: Any, _inp: Any, out: Any) -> None:
        captured[0] = out[0, -1].float().cpu()

    handles = [adapter.model.model.embed_tokens.register_forward_hook(embed_hook)]
    handles += [b.register_forward_hook(hook(i + 1)) for i, b in enumerate(blocks)]
    n_layers = len(blocks) + 1
    out = np.zeros((len(texts), n_layers, adapter.model.config.hidden_size), dtype=np.float32)
    last_tokens: list[str] = []
    t0 = time.time()
    try:
        with torch.no_grad():
            for k, text in enumerate(texts):
                ids = adapter._ids(text)
                captured.clear()
                adapter.model.model(ids)  # the backbone only: no LM head, nothing generated
                out[k] = np.stack([captured[layer].numpy() for layer in range(n_layers)])
                last_tokens.append(adapter.tokenizer.convert_ids_to_tokens([int(ids[0, -1])])[0])
                if (k + 1) % 25 == 0 or k + 1 == len(texts):
                    el = time.time() - t0
                    print(f"{label} {k + 1}/{len(texts)}  {el / (k + 1):.2f} s/item  "
                          f"eta {el / (k + 1) * (len(texts) - k - 1):.0f} s", file=sys.stderr, flush=True)
    finally:
        for h in handles:
            h.remove()
    return out, last_tokens


def fit(acts: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arrows [layers, d], and per layer the projections of the negative and positive training means."""
    mu_pos = acts[pos].mean(axis=0)
    mu_neg = acts[neg].mean(axis=0)
    d = mu_pos - mu_neg
    return d, np.einsum("ld,ld->l", mu_neg, d), np.einsum("ld,ld->l", mu_pos, d)


def project(acts: np.ndarray, arrow: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """[n, layers] positions on the arrow: 0 = training negative mean, 1 = training positive mean."""
    d, lo, hi = arrow
    raw = np.einsum("nld,ld->nl", acts, d)
    span = hi - lo
    # a zero arrow (layer 0: every statement ends in the same "." token) projects to NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(span > 0, (raw - lo) / np.where(span > 0, span, 1), np.nan)


def auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float | None:
    if len(pos_scores) == 0 or len(neg_scores) == 0 or np.isnan(pos_scores).any() or np.isnan(neg_scores).any():
        return None
    diff = pos_scores[:, None] - neg_scores[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def threshold_acc(pos_scores: np.ndarray, neg_scores: np.ndarray, t: float = 0.5) -> float | None:
    if np.isnan(pos_scores).any() or np.isnan(neg_scores).any():
        return None
    return float(((pos_scores > t).sum() + (neg_scores <= t).sum()) / (len(pos_scores) + len(neg_scores)))


def shuffled(pos: np.ndarray, neg: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The same training items with their pos/neg labels permuted among themselves."""
    idx = np.flatnonzero(pos | neg)
    labels = pos[idx].copy()
    np.random.default_rng(seed).shuffle(labels)
    p, n = np.zeros_like(pos), np.zeros_like(neg)
    p[idx[labels]] = True
    n[idx[~labels]] = True
    return p, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", choices=["cities", "sp_en_trans"])
    ap.add_argument("--limit", type=int, help="first N statements only (timing probe; the split check is skipped)")
    ap.add_argument("--shuffles", type=int, default=0, help="label-shuffle control: N refits (default: 0)")
    args = ap.parse_args()
    path = HERE / "data" / f"{args.dataset}.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))[: args.limit]
    frozen = json.loads((HERE / "split.json").read_text())[args.dataset]
    items = [item_of(args.dataset, r) for r in rows]
    held = split_groups(items, frozen["held_out_frac"], frozen["seed"])
    if args.limit is None and sorted({i for i, h in zip(items, held) if h}) != frozen["held_out"]:
        raise SystemExit("recomputed split differs from split.json")
    label = np.array([r["label"] == "1" for r in rows])

    adapter = HFAdapter(MODEL, revision=REVISION)
    acts, last = read_last_token(adapter, [r["statement"] for r in rows], args.dataset)
    train_pos, train_neg, test_pos, test_neg = label & ~held, ~label & ~held, label & held, ~label & held
    fitted = fit(acts, train_pos, train_neg)
    scores = project(acts, fitted)
    sh = np.full((args.shuffles, acts.shape[1]), np.nan)
    for k in range(args.shuffles):
        ss = project(acts, fit(acts, *shuffled(train_pos, train_neg, seed=k)))
        sh[k] = [np.nan if (a := auc(ss[test_pos, li], ss[test_neg, li])) is None else a for li in range(acts.shape[1])]
    common = {"dataset": args.dataset, "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "n_statements": len(rows), "held_out_statements": int(held.sum()), "split_seed": frozen["seed"],
              "last_tokens": sorted(set(last)), "model": MODEL, "revision": REVISION}
    for li in range(acts.shape[1]):
        s = scores[:, li]
        row: dict[str, Any] = {"layer": li, "heldout_auc": auc(s[test_pos], s[test_neg]),
                               "heldout_acc_at_midpoint": threshold_acc(s[test_pos], s[test_neg]),
                               "train_auc": auc(s[train_pos], s[train_neg])}
        if args.shuffles:
            row["shuffled_heldout_auc_p5_p50_p95"] = [float(x) for x in np.nanpercentile(sh[:, li], [5, 50, 95])]
            row["shuffled_frac_auc_above_half"] = float(np.mean(sh[:, li] > 0.5))
        print(json.dumps({**row, **common}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
