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
import contextlib
import json
import logging
import math
import os
import tempfile
import time
import warnings
from typing import Dict, List, Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Subset

from shape.data import DatasetCard, ShapeDataset, densify_union
from shape.model import Shape, fusion_slot_masses, loss_for
from shape.tracking import finish as wandb_finish, wandb_logger

# One summary line per dataset is the whole output contract; Lightning's INFO
# banner would put an accelerator report between every seed, and its
# worker-count nag is noise when a corpus is small enough not to need them.
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", ".*does not have many workers.*")

# Lightning maps these to torch.autocast(device, dtype=..., cache_enabled=False)
# and refuses a GradScaler for bf16 -- bf16 carries fp32's 8-bit exponent, so
# gradients do not underflow the way fp16's [-14,15] range makes them
# (Kalamkar et al. 2019; Micikevicius et al. 2017 sec. 3.2). No scaler anywhere.
PRECISION = {"bf16": "bf16-mixed", "fp32": "32-true"}


def configure_backends(precision: str) -> None:
    """Matmul precision policy. Must run before any CUDA work.

    A bf16 tensor-core GEMM accumulates into fp32, but cuBLAS may split one
    reduction across thread blocks and combine the partials in bf16. Forcing that
    combine to fp32 is what makes the dot product fp32-accumulated in the sense
    Micikevicius et al. 2017 sec. 3 requires, so it is set rather than assumed.

    TF32 follows the flag instead of being enabled globally. In fp32 mode this
    run is the accuracy reference the bf16 path gets compared against, and TF32
    would quietly make that reference 10-bit-mantissa -- the comparison would
    then understate any bf16 regression. In bf16 mode autocast already routes
    every large matmul to bf16, so TF32 only reaches the few fp32 matmuls that
    escape it, where the speed is free. (No convolutions exist in this model, so
    the cudnn flag is set only to keep the two consistent.)
    """
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    use_tf32 = precision == "bf16"
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32


def collate(items: List[Dict]) -> Dict:
    """A batch of samples as ``[B,N,W,C]`` plus targets, covariates and validity."""
    out = {"y": torch.stack([it["y"] for it in items]),          # [B, N, H] or [B]
           "u": torch.stack([it["u"] for it in items]),          # [B, W, 8]
           "u_mask": torch.stack([it["u_mask"] for it in items]),  # [B, W, 8]
           "pe": None if items[0].get("pe") is None
                 else torch.stack([it["pe"] for it in items]),   # [B, N, 16]  (dense path)
           "mask": None,
           "num_nodes": None,
           "deg": None if items[0]["deg"] is None else torch.stack([it["deg"] for it in items]),
           # Ragged: snapshots in a window have different edge counts, so the
           # topology stays a list and is densified only if a relation wants it.
           "edge_index": None if items[0].get("edge_index") is None
                         else [it["edge_index"] for it in items],
           "edge_weight": None if items[0].get("edge_weight") is None
                          else [it["edge_weight"] for it in items]}
    # A probe-only corpus yields only its active nodes, which differ per sample.
    compact = densify_union(items)
    if compact is not None:
        out["x"], out["node_ids"], out["num_nodes"], out["pe"] = compact   # [B, K, W, C]
    else:
        out["x"] = torch.stack([it["x"] for it in items])       # [B, N, W, C]
    if items[0]["mask"] is not None:
        out["mask"] = torch.stack([it["mask"] for it in items])
    return out


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
        y = y.view_as(pred)

        def score(p, t, m):
            keep = (t != 0) if m is None else (m & (t != 0))
            p, t = p[keep], t[keep]
            e = (p - t).abs()
            return (float(e.mean()), float(e.pow(2).mean().sqrt()),
                    float((e / t.abs()).mean() * 100))

        mae, rmse, mape = score(pred.reshape(-1), y.reshape(-1),
                                None if mask is None else mask.reshape(-1))
        out = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
        # TIDES' Table 2 averages over horizons; its Table 12 (Metr-LA, PeMS-Bay)
        # reports step 12 alone, as does most of the traffic literature. Both are
        # emitted so a comparison names which one it uses.
        for h in (3, 6, 12):
            if pred.shape[-1] < h:
                continue
            mh = None if mask is None else mask[..., h - 1].reshape(-1)
            a, r, p_ = score(pred[..., h - 1].reshape(-1), y[..., h - 1].reshape(-1), mh)
            out[f"MAE@{h}"], out[f"RMSE@{h}"], out[f"MAPE@{h}"] = a, r, p_
        return out
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


class ShapeTask(L.LightningModule):
    """The model, its loss, and the card's metric over a whole fold.

    ``node_masks`` is per-fold and non-None only for a transductive split, where
    it says which nodes count towards the loss and the score.
    """

    def __init__(self, card: DatasetCard, node_masks: Dict[str, Optional[torch.Tensor]],
                 lr: float, weight_decay: float, hidden: int, layers: int,
                 use_covariates: bool, eval_chunk: int = 0, fuse: str = "mean",
                 relations: str = "gram+topo",
                 patch_len: int = 12,
                 attn_depth: int = 2, rel_gate: bool = True, layer_agg: bool = True,
                 covariate_readout: bool = False):
        super().__init__()
        self.card = card
        self.node_masks = node_masks
        self.lr, self.weight_decay = lr, weight_decay
        self.model = Shape([card], hidden_dim=hidden, num_layers=layers,
                           use_covariates=use_covariates, eval_chunk=eval_chunk, fuse=fuse,
                           relations=relations, patch_len=patch_len,
                           attn_depth=attn_depth, rel_gate=rel_gate,
                           layer_agg=layer_agg, covariate_readout=covariate_readout)
        self.test_out: List[Dict] = []

    def step(self, batch, fold: str):
        pred = self.model(batch["x"], edge_index=batch.get("edge_index"),
                          edge_weight=batch.get("edge_weight"),
                          u=batch["u"], u_mask=batch["u_mask"], card=self.card,
                          num_nodes=batch.get("num_nodes"), deg=batch.get("deg"))
        y = batch["y"]
        nodes = self.node_masks[fold]
        if nodes is not None:                 # transductive: score only this fold's nodes
            pred, y = pred[:, nodes], y[:, nodes]
        return pred, y, loss_for(self.card, pred, y, batch["mask"])

    def training_step(self, batch, _):
        _, _, loss = self.step(batch, "train")
        self.log("train_loss", loss, batch_size=batch["x"].shape[0])
        return loss

    def validation_step(self, batch, _):
        _, _, loss = self.step(batch, "val")
        self.log("val_loss", loss, batch_size=batch["x"].shape[0])

    def test_step(self, batch, _):
        pred, y, _ = self.step(batch, "test")
        # The metrics are set-level -- F1, AUC, and a masked MAE over non-zero
        # targets -- so the fold is concatenated and scored once, never averaged
        # per batch.
        self.test_out.append({"pred": pred.float().cpu(), "y": y.cpu(),
                              "mask": None if batch["mask"] is None else batch["mask"].cpu()})

    def on_test_epoch_end(self):
        cat = lambda k: torch.cat([o[k] for o in self.test_out])
        mask = None if self.test_out[0]["mask"] is None else cat("mask")
        self.metrics = metrics_for(self.card, cat("pred"), cat("y"), mask)
        self.test_out.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def loaders(ds: ShapeDataset, batch_size: int, eval_batch_size: int = 0,
            shuffle: bool = False, seed: int = 0, num_workers: int = 0):
    """One loader per fold, plus the node mask each fold scores on.

    A temporal split gives each fold its own time samples. A node split is
    transductive: every fold walks every sample and differs only in its mask.
    """
    if ds.card.split_kind == "temporal":
        folds = {k: ds.splits[k].tolist() for k in ("train", "val", "test")}
        nodes = {k: None for k in folds}
    else:
        every = list(range(len(ds)))
        folds = {k: every for k in ("train", "val", "test")}
        nodes = {k: ds.splits[k].bool() for k in folds}
    # A DataLoader iterator draws a worker base seed from the global RNG every
    # epoch. With num_workers=0 that seed is never used, but the draw would still
    # shift the stream the model's dropout runs on. A private generator keeps the
    # global stream reserved for the model.
    # The node budget caps the graph while training but not while evaluating, so
    # a batch that fits in training can still OOM on the full eligible set.
    bs = {k: (batch_size if k == "train" else (eval_batch_size or batch_size)) for k in folds}
    # Only the train fold may be shuffled: val/test metrics are set-level and are
    # concatenated before scoring, so their order is irrelevant either way.
    # Windowing happens in __getitem__, so with the default num_workers=0 every
    # batch is cut synchronously in the main process while the device idles. The
    # sampler and its generator decide batch composition, so workers change only
    # who does the cutting: results are unaffected. persistent_workers matters
    # here because epochs are short and respawning would dominate them.
    kw = {}
    if num_workers:
        kw = dict(num_workers=num_workers, persistent_workers=True,
                  prefetch_factor=4, pin_memory=True)
    dl = {k: DataLoader(Subset(ds, idx), batch_size=bs[k], shuffle=(shuffle and k == "train"),
                        collate_fn=collate, generator=torch.Generator().manual_seed(seed), **kw)
          for k, idx in folds.items()}
    return dl, nodes


def train_one(name: str, seed: int, epochs: int, batch_size: int, lr: float,
              patience: int, device: str, hidden: int, layers: int,
              use_covariates: bool = False, weight_decay: float = 0.0,
              precision: str = "bf16", eval_chunk: int = 0,
              eval_batch_size: int = 0, fuse: str = "mean",
              shuffle: bool = False, ckpt_dir: Optional[str] = None,
              limit_train_batches: int = 0, relations: str = "gram+topo",
              patch_len: int = 12,
              attn_depth: int = 2, rel_gate: bool = True, layer_agg: bool = True,
              covariate_readout: bool = False, num_workers: int = 0) -> Dict:
    configure_backends(precision)
    L.seed_everything(seed, workers=True, verbose=False)
    ds = ShapeDataset(name)
    card = ds.card
    dl, nodes = loaders(ds, batch_size, eval_batch_size, shuffle, seed, num_workers)
    task = ShapeTask(card, nodes, lr, weight_decay, hidden, layers, use_covariates,
                     eval_chunk, fuse, relations, patch_len, attn_depth, rel_gate, layer_agg, covariate_readout)

    accelerator, devices = ("cpu", 1) if device == "cpu" else ("gpu", [int(device.split(":")[1])])
    if device != "cpu" and torch.cuda.is_available():
        # reset_peak_memory_stats needs an initialised context on that device
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    # The best epoch is scored and then thrown away unless a directory is named.
    # Phases that freeze a trained backbone need the weights to outlive the run.
    keep = ckpt_dir is not None
    if keep:
        os.makedirs(ckpt_dir, exist_ok=True)
    with (contextlib.nullcontext(ckpt_dir) if keep else tempfile.TemporaryDirectory()) as d:
        best = ModelCheckpoint(dirpath=d, monitor="val_loss", mode="min", save_top_k=1,
                               filename=f"{name}-seed{seed}", enable_version_counter=False)
        wb = wandb_logger(f"{name}-seed{seed}", group=f"sup-{name}", config=dict(
            protocol="sup", corpora=[name], seed=seed, hidden=hidden, layers=layers,
            lr=lr, batch_size=batch_size, relations=relations, rel_gate=rel_gate,
            patch_len=patch_len, attn_depth=attn_depth, W=card.W, H=card.H,
            precision=precision))
        trainer = L.Trainer(
            max_epochs=epochs, accelerator=accelerator, devices=devices,
            precision=PRECISION[precision],
            # Caps the per-epoch training budget so a single-dataset run can be
            # given exactly the exposure a joint run gives one corpus.
            limit_train_batches=limit_train_batches or 1.0,
            gradient_clip_val=5.0, num_sanity_val_steps=0,
            callbacks=[EarlyStopping("val_loss", min_delta=1e-6, patience=patience, mode="min"), best],
            logger=wb or False, enable_progress_bar=False, enable_model_summary=False,
        )
        trainer.fit(task, dl["train"], dl["val"])
        # Score the best epoch, not the last one the stopper happened to reach.
        trainer.test(task, dl["test"], ckpt_path=best.best_model_path, verbose=False)
        val_loss = float(best.best_model_score) if best.best_model_score is not None else math.inf
    wandb_finish()

    row = {"dataset": name, "seed": seed, "metric": card.metric,
            "epochs": trainer.current_epoch,
            "hidden": hidden, "layers": layers, "lr": lr, "weight_decay": weight_decay,
            "covariates": use_covariates, "revin": task.model.revin, "batch_size": batch_size,
            "precision": precision, "fuse": fuse, "shuffle": shuffle,
            "limit_train_batches": limit_train_batches,
            "relations": relations, "patch_len": patch_len,
            "attn_depth": attn_depth, "rel_gate": rel_gate, "layer_agg": layer_agg,
            "covariate_readout": covariate_readout,
            "checkpoint": best.best_model_path if keep else None,
           "corpora": [name], "protocol": "sup", "W": card.W, "H": card.H,
           # A relation whose branch stayed dead leaves its slot at exact zero.
           "fusion_slots": fusion_slot_masses(task.model),
            "val_loss": val_loss, "seconds": round(time.time() - t0, 1),
            "peak_mem_mb": _peak_mem_mb(device), **task.metrics}
    for r in task.model.rel_names:
        assert row["fusion_slots"][r] > 0, (
            f"relation {r!r} never left its zero init: its branch was dead all run")
    # The checkpoint has to carry its own config, or nothing downstream can
    # rebuild the model that produced it.
    if keep:
        json.dump(row, open(os.path.join(ckpt_dir, "train.json"), "w"), indent=2)
    return row


def _peak_mem_mb(device: str) -> float:
    """Peak allocated CUDA memory for this run, 0.0 on CPU."""
    if device == "cpu" or not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated(device) / 2 ** 20, 1)


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
    ap.add_argument("--attn-depth", type=int, default=2,
                    help="causal self-attention layers over the patch sequence; "
                         "0 = the order-free pooled encoder. Needs --patch-len")
    ap.add_argument("--no-rel-gate", dest="rel_gate", action="store_false",
                    help="disable eq:relation-gate (ablation only; on by default)")
    ap.add_argument("--rel-gate", dest="rel_gate", action="store_true", default=True,
                    help="gate the aggregated message per node and channel")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="dataloader workers; windowing happens in __getitem__, so 0 "
                         "cuts every batch in the main process while the device waits")
    ap.add_argument("--covariate-readout", action="store_true",
                    help="FiLM the final representation on the calendar covariates, "
                         "after propagation rather than before it")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--precision", choices=list(PRECISION), default="bf16",
                    help="bf16 mixed precision (default) or a strict fp32 reference run")
    ap.add_argument("--eval-chunk", type=int, default=0,
                    help="split the eval Gram into blocks of this many nodes "
                         "(block-diagonal approximation; needed for ArXiv)")
    ap.add_argument("--eval-batch-size", type=int, default=0,
                    help="batch size for val/test; defaults to --batch-size")
    ap.add_argument("--fuse", choices=["mean", "attn"], default="mean",
                    help="how per-source node embeddings combine: unweighted "
                         "mean (original) or a learned query over source tokens")
    ap.add_argument("--patch-len", type=int, default=12,
                    help="patch the input window instead of a flat Linear(W,.); 0 = off")
    ap.add_argument("--no-layer-agg", dest="layer_agg", action="store_false",
                    help="read H^(L) only instead of eq:pi (ablation arm)")
    ap.add_argument("--relations", default="gram+topo",
                    help="default is \\MODEL{} as defined in the paper: the learned\n"
                         "Gram relation plus the observed topology. Ablation values:\n"
                         "gram | topo | ident | none.")
    ap.add_argument("--limit-train-batches", type=int, default=0,
                    help="train on at most this many batches per epoch (0 = all)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="keep the best-epoch checkpoint here instead of discarding it")
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle the train fold (TIDES does; SHAPE historically did not)")
    ap.add_argument("--out", default=None, help="append results as JSON lines")
    a = ap.parse_args()

    for name in a.datasets:
        rows = []
        for seed in a.seeds:
            r = train_one(name, seed, a.epochs, a.batch_size, a.lr, a.patience,
                          a.device, a.hidden, a.layers, a.covariates, a.weight_decay,
                          a.precision, a.eval_chunk, a.eval_batch_size, a.fuse, a.shuffle,
                          a.ckpt_dir, a.limit_train_batches, a.relations, a.patch_len,
                          attn_depth=a.attn_depth, rel_gate=a.rel_gate,
                          layer_agg=a.layer_agg, covariate_readout=a.covariate_readout,
                          num_workers=a.num_workers)
            rows.append(r)
            if a.out:
                # A run that finished should never lose its row to a missing
                # directory: the write happens after training, so the cost of
                # the failure is the whole run.
                os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
        keys = [k for k in rows[0] if k not in ("dataset", "seed", "metric", "epochs",
                                               "val_loss", "seconds", "precision", "fuse",
                                               "shuffle", "checkpoint",
                                               "limit_train_batches", "relations",
                                               "scale")
                and isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool)]
        summary = "  ".join(
            f"{k} {sum(r[k] for r in rows) / len(rows):.4f}"
            f"±{(sum((r[k] - sum(q[k] for q in rows) / len(rows)) ** 2 for r in rows) / len(rows)) ** 0.5:.4f}"
            for k in keys)
        print(f"{name:16s} {summary}   ({len(rows)} seeds, "
              f"{sum(r['epochs'] for r in rows) // len(rows)} ep avg, "
              f"{sum(r['seconds'] for r in rows):.0f}s)", flush=True)
