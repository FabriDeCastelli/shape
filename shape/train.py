"""Supervised single-dataset training for SHAPE, evaluated on the card's metric.

One model per dataset, no MoE and no shared pretraining: this is the baseline the
later components have to beat, so it must be measurable on exactly the splits and
metrics the TIDES paper reports.

Two split kinds, from the card. ``temporal`` selects which time samples belong to
each fold. ``node`` is transductive -- every fold sees every snapshot, and the
fold decides which *nodes* contribute to the loss and the score.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional

import torch

from shape.data import DatasetCard, ShapeDataset
from shape.model import Shape, loss_for


def stack(ds: ShapeDataset, idx: List[int], device: str):
    """A batch of samples as ``[B,N,W,C]`` plus targets and validity."""
    items = [ds[i] for i in idx]
    x = torch.stack([it["x"] for it in items]).to(device)
    y = torch.stack([it["y"] for it in items]).to(device)
    m = None
    if items[0]["mask"] is not None:
        m = torch.stack([it["mask"] for it in items]).to(device)
    return x, y, m


def metrics_for(card: DatasetCard, pred: torch.Tensor, y: torch.Tensor,
                mask: Optional[torch.Tensor]) -> Dict[str, float]:
    """Exactly the numbers the paper reports for this dataset's metric.

    Forecasting targets are globally z-scored for training, so both sides go
    back to raw units first -- an MAE in normalised units is not comparable to
    the paper's 14.09.
    """
    if card.task == "forecast" and card.global_std:
        pred = pred * card.global_std + card.global_mean
        y = y * card.global_std + card.global_mean
    if card.metric == "mse":
        return {"MSE": float(((pred - y.view_as(pred)) ** 2).mean())}
    if card.metric == "masked_mae":
        p, t = pred.reshape(-1), y.view_as(pred).reshape(-1)
        keep = t != 0 if mask is None else (mask.reshape(-1) & (t != 0))
        p, t = p[keep], t[keep]
        err = (p - t).abs()
        return {"MAE": float(err.mean()), "RMSE": float(err.pow(2).mean().sqrt()),
                "MAPE": float((err / t.abs()).mean() * 100)}
    if card.metric == "accuracy":
        from sklearn.metrics import f1_score
        pl = pred.reshape(-1, card.num_classes).argmax(-1).cpu().numpy()
        tl = y.reshape(-1).long().cpu().numpy()
        return {"MicroF1": float(f1_score(tl, pl, average="micro") * 100),
                "MacroF1": float(f1_score(tl, pl, average="macro") * 100)}
    if card.metric == "rocauc":
        from sklearn.metrics import roc_auc_score
        t = y.reshape(-1).long().cpu().numpy()
        p = pred.reshape(-1).float().cpu().numpy()
        # A fold with one class present has no defined AUC.
        return {"AUC": float(roc_auc_score(t, p)) if len(set(t.tolist())) > 1 else float("nan")}
    raise ValueError(card.metric)


def node_fold(ds: ShapeDataset, fold: str, device: str) -> Optional[torch.Tensor]:
    """Which nodes count for this fold, or None when the split is temporal."""
    if ds.card.split_kind != "node":
        return None
    return ds.splits[fold].to(device).bool()


def run_epoch(model, ds, card, sample_idx, node_mask, opt, device, batch_size, train: bool):
    model.train(train)
    total, seen, preds, ys, masks = 0.0, 0, [], [], []
    for s in range(0, len(sample_idx), batch_size):
        idx = sample_idx[s:s + batch_size]
        x, y, m = stack(ds, idx, device)
        b_u = ds.u[idx].to(device) if ds.u is not None else None
        b_um = ds.u_mask[idx].to(device) if ds.u_mask is not None else None
        with torch.set_grad_enabled(train):
            pred = model(x, u=b_u, u_mask=b_um, card=card)
            if node_mask is not None:              # transductive: score only this fold's nodes
                pred, y = pred[:, node_mask], y[:, node_mask]
            loss = loss_for(card, pred, y, m)
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        else:
            preds.append(pred.detach().cpu())
            ys.append(y.detach().cpu())
            masks.append(None if m is None else m.detach().cpu())
        total += float(loss) * len(idx)
        seen += len(idx)
    if train:
        return total / max(seen, 1), None
    cat_m = None if masks[0] is None else torch.cat(masks)
    return total / max(seen, 1), metrics_for(card, torch.cat(preds), torch.cat(ys), cat_m)


def train_one(name: str, seed: int, epochs: int, batch_size: int, lr: float,
              patience: int, device: str, hidden: int, layers: int,
              use_covariates: bool = False, weight_decay: float = 0.0) -> Dict:
    torch.manual_seed(seed)
    ds = ShapeDataset(name)
    card = ds.card
    model = Shape([card], hidden_dim=hidden, num_layers=layers,
                     use_covariates=use_covariates).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    if card.split_kind == "temporal":
        folds = {k: ds.splits[k].tolist() for k in ("train", "val", "test")}
        nodes = {k: None for k in folds}
    else:
        every = list(range(len(ds)))
        folds = {k: every for k in ("train", "val", "test")}
        nodes = {k: node_fold(ds, k, device) for k in folds}

    best, best_state, bad, t0 = math.inf, None, 0, time.time()
    for ep in range(epochs):
        run_epoch(model, ds, card, folds["train"], nodes["train"], opt, device, batch_size, True)
        vloss, _ = run_epoch(model, ds, card, folds["val"], nodes["val"], opt, device, batch_size, False)
        if vloss < best - 1e-6:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    _, test = run_epoch(model, ds, card, folds["test"], nodes["test"], opt, device, batch_size, False)
    return {"dataset": name, "seed": seed, "metric": card.metric, "epochs": ep + 1,
            "val_loss": best, "seconds": round(time.time() - t0, 1), **test}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--covariates", action="store_true")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--out", default=None, help="append results as JSON lines")
    a = ap.parse_args()

    for name in a.datasets:
        rows = []
        for seed in a.seeds:
            r = train_one(name, seed, a.epochs, a.batch_size, a.lr, a.patience,
                          a.device, a.hidden, a.layers, a.covariates, a.weight_decay)
            rows.append(r)
            if a.out:
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
        keys = [k for k in rows[0] if k not in ("dataset", "seed", "metric", "epochs", "val_loss", "seconds")]
        summary = "  ".join(
            f"{k} {sum(r[k] for r in rows) / len(rows):.4f}"
            f"±{(sum((r[k] - sum(q[k] for q in rows) / len(rows)) ** 2 for r in rows) / len(rows)) ** 0.5:.4f}"
            for k in keys)
        print(f"{name:16s} {summary}   ({len(rows)} seeds, "
              f"{sum(r['epochs'] for r in rows) // len(rows)} ep avg, "
              f"{sum(r['seconds'] for r in rows):.0f}s)", flush=True)
