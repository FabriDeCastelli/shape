"""Score one released checkpoint on one corpus. The only evaluation entry point.

    python -m shape.evaluate --ckpt runs/final/flow/sup/Pems08W864A/seed0 \
        --target Pems08W864A --fold test --out results/final/flow.jsonl

The checkpoint's own ``train.json`` supplies the architecture, so nothing about
the model is restated on the command line and a run can never be scored as a
configuration it was not trained in. The metrics are the ones the corpus's card
names, alongside the trivial predictor on the same fold.

Whether the checkpoint has seen the target is *recorded*, not assumed and not
used to skip: a supervised row and a zero-shot row are produced by this same
command and differ only in ``seen_target``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from shape.data import ShapeDataset
from shape.model import Shape, fusion_slot_masses
from shape.pretrain import BalancedBatchSampler, MultiCorpus, collate_pack, trivial_baseline
from shape.train import configure_backends, metrics_for
from shape.zeroshot_traffic import build_model


def find_checkpoint(path: str) -> str:
    """Accept either a checkpoint or the directory holding one."""
    if os.path.isfile(path):
        return path
    hits = sorted(glob.glob(os.path.join(path, "*.ckpt")))
    if not hits:
        raise FileNotFoundError(f"no .ckpt under {path!r}")
    if len(hits) > 1:
        raise ValueError(f"{path!r} holds {len(hits)} checkpoints; name one: {hits}")
    return hits[0]


def load_meta(ckpt: str) -> Dict:
    meta = os.path.join(os.path.dirname(ckpt), "train.json")
    if not os.path.exists(meta):
        raise FileNotFoundError(
            f"{meta!r} is missing, so the architecture of {ckpt!r} is unknown. "
            "Every run written by shape.train or shape.pretrain records one; a "
            "checkpoint without it predates that and cannot be scored safely.")
    m = json.load(open(meta))
    m.setdefault("checkpoint", ckpt)
    m["checkpoint"] = ckpt                       # the file we were actually given
    return m


def assert_architecture(model: Shape, meta: Dict) -> None:
    """The model that was rebuilt is the model the checkpoint recorded."""
    want = [] if meta["relations"] == "none" else meta["relations"].split("+")
    assert model.rel_names == want, f"rebuilt {model.rel_names}, recorded {want}"
    assert model.rel_fusion.proj.in_features == (len(want) + 1) * model.hidden_dim
    for r in want:
        blk = model.rel_blocks[r]
        assert (blk.gates is not None) == bool(meta.get("rel_gate", True)), r
        assert (blk.depth_pool is not None) == bool(meta.get("layer_agg", True)), r
    # A relation whose branch was dead all run left its slot at exact zero.
    slots = fusion_slot_masses(model)
    dead = [r for r in want if slots[r] == 0.0]
    assert not dead, f"relation(s) {dead} never left their zero init in this checkpoint"


@torch.no_grad()
def score(model: Shape, target: str, device: str, batch_size: int, fold: str) -> Dict:
    ds = MultiCorpus([target], root=None).use_fold(fold)
    dl = DataLoader(ds, batch_sampler=BalancedBatchSampler(
        ds, {target: batch_size}, None, False, torch.Generator().manual_seed(0)),
        collate_fn=collate_pack, num_workers=2)
    card = ds.sets[target].card
    preds, ys, masks = [], [], []
    for b in dl:
        pred = model(b["x"].to(device), edge_index=b.get("edge_index"),
                     edge_weight=b.get("edge_weight"),
                     u=b["u"].to(device), u_mask=b["u_mask"].to(device), card=card,
                     num_nodes=b.get("num_nodes"),
                     deg=None if b["deg"] is None else b["deg"].to(device))
        preds.append(pred.float().cpu()); ys.append(b["y"])
        masks.append(None if b["mask"] is None else b["mask"])
    m = None if masks[0] is None else torch.cat(masks)
    return metrics_for(card, torch.cat(preds), torch.cat(ys), m) | {"windows": len(ds)}


# Steps per cycle, by sampling period. A seasonal copy is the baseline a long
# context makes available, and on a periodic series it is far stronger than
# persistence: on MoBiNS at W=336 the weekly copy scores 0.140 MAE against
# persistence's 1.078. Reporting it is what keeps a long-context result honest.
CYCLES = {"5min": {"daily": 288, "weekly": 2016},
          "15min": {"daily": 96, "weekly": 672},
          "1h": {"daily": 24, "weekly": 168},
          "1d": {"weekly": 7}}


def seasonal_periods(card) -> Dict[str, int]:
    """Cycles this corpus's window is long enough to copy from."""
    return {n: p for n, p in CYCLES.get(card.freq or "", {}).items() if p <= card.W}


def trivial(target: str, fold: str) -> Dict:
    """The predictors the model has to beat: persistence, a seasonal copy, the prior."""
    ds = MultiCorpus([target], root=None).use_fold(fold)
    card = ds.sets[target].card
    if card.task != "forecast":
        return {"trivial_loss": trivial_baseline(ShapeDataset(target), "train")}
    W, H, ch = card.W, card.H, card.channel_groups["real"][0]
    cycles = seasonal_periods(card)
    persist, seasonal, ys, masks = [], {n: [] for n in cycles}, [], []
    for i in range(len(ds)):
        it = ds[i]
        x = it["x"][:, :, ch]                                        # [N, W]
        persist.append(x[:, -1:].expand(-1, H).unsqueeze(0))
        for n, p in cycles.items():
            # The same phase one cycle back. A horizon longer than the cycle
            # would run past the window, so it is tiled.
            src = torch.cat([x[:, W - p:]] * (H // p + 1), dim=1)[:, :H]
            seasonal[n].append(src.unsqueeze(0))
        ys.append(it["y"].unsqueeze(0))
        masks.append(None if it["mask"] is None else it["mask"].unsqueeze(0))
    y = torch.cat(ys)
    m = None if masks[0] is None else torch.cat(masks)
    out = {f"persistence_{k}": v for k, v in
           metrics_for(card, torch.cat(persist), y, m).items()}
    # The strongest seasonal copy is the one the model actually has to beat.
    best = None
    for n, preds in seasonal.items():
        r = metrics_for(card, torch.cat(preds), y, m)
        out |= {f"seasonal_{n}_{k}": v for k, v in r.items()}
        if best is None or r["MAE"] < best[1]:
            best = (n, r["MAE"], r["RMSE"])
    if best:
        out |= {"seasonal_best": best[0], "seasonal_MAE": best[1],
                "seasonal_RMSE": best[2]}
    return out


# Different batch shapes select different GEMM kernels, and over four
# propagation layers that moves the reported MAE by ~0.3% (13.8486 at batch 64
# against 13.8843 at batch 16 on PeMS08, fp32, same weights, same windows, same
# mask). Nothing is wrong -- floating point addition is not associative -- but a
# table cell must not depend on it, so every result is produced at this batch
# size and the row records it.
EVAL_BATCH = 64


def evaluate(ckpt: str, target: str, fold: str, device: str, batch_size: int) -> Dict:
    meta = load_meta(find_checkpoint(ckpt))
    model = build_model(meta, device)
    assert_architecture(model, meta)
    row = score(model, target, device, batch_size, fold)
    card = ShapeDataset(target).card
    return row | trivial(target, fold) | {
        "target": target, "fold": fold, "metric": card.metric, "W": card.W, "H": card.H,
        # Recorded, never used to skip: this is what separates the supervised row
        # of a table from the zero-shot one.
        "seen_target": target in meta["corpora"],
        "protocol": meta.get("protocol", "loo" if len(meta["corpora"]) > 1 else "sup"),
        "trained_on": meta["corpora"], "k": len(meta["corpora"]),
        "seed": meta.get("seed"), "relations": meta["relations"],
        "rel_gate": meta.get("rel_gate"), "layer_agg": meta.get("layer_agg", True),
        "patch_len": meta.get("patch_len"), "attn_depth": meta.get("attn_depth"),
        "hidden": meta.get("hidden"), "layers": meta.get("layers"),
        "fusion_slots": fusion_slot_masses(model),
        "checkpoint": meta["checkpoint"], "eval_batch_size": batch_size,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="a .ckpt file, or the directory holding one")
    ap.add_argument("--target", required=True, help="corpus to score")
    ap.add_argument("--fold", default="test", choices=("train", "val", "test"))
    ap.add_argument("--out", default=None, help="append the row as JSON lines")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=EVAL_BATCH,
                    help="part of the reported number; see EVAL_BATCH")
    ap.add_argument("--precision", default="fp32",
                    help="fp32 by default: evaluation is cheap and bf16 adds "
                         "another 0.004 MAE of kernel noise on top of the batch effect")
    a = ap.parse_args()
    configure_backends(a.precision)

    row = evaluate(a.ckpt, a.target, a.fold, a.device, a.batch_size)
    seen = "supervised" if row["seen_target"] else "zero-shot"
    head = "  ".join(f"{k} {row[k]:.4f}" for k in row
                     if k in ("MAE", "RMSE", "MAPE", "AUC", "MSE"))
    print(f"{row['target']:18s} {seen:10s} k={row['k']:<3d} {a.fold:5s} {head}", flush=True)
    for name, keys in (("persistence", ("persistence_MAE", "persistence_RMSE")),
                       (f"seasonal/{row.get('seasonal_best')}",
                        ("seasonal_MAE", "seasonal_RMSE"))):
        vals = "  ".join(f"{k.split('_', 1)[1]} {row[k]:.4f}" for k in keys if k in row)
        if vals:
            print(f"{'':18s} {name:16s}       {vals}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"  -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
