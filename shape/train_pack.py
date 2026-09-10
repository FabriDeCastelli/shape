"""Joint multi-network training on a MiNT datapack (phase 3).

One shared SHAPE backbone over every network in the pack. All MiNT corpora are
graph-classify, so they share a single head and the model carries no per-network
parameters at all -- which is what makes a checkpoint transferable to an unseen
network with no new weights.

The checkpoint is selected on **macro validation ROC-AUC**: the metric averaged
per network, then across networks. MiNT selects the same way ("we compute the
average validation results across these datasets"). Pooled loss would not do --
val samples span 12 to 324 per network, so a pooled figure is chosen by the
largest network alone, and BCE loss is not monotone in AUC anyway.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from typing import Dict, List

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from shape.model import Shape, loss_for
from shape.packs import (HomogeneousBatchSampler, PackDataset, collate_pack,
                         pack_names)
from shape.train import PRECISION, configure_backends

logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", ".*does not have many workers.*")


class PackTask(L.LightningModule):
    """Shared backbone over a pack, scored by macro val AUC."""

    def __init__(self, pack: PackDataset, lr: float, weight_decay: float,
                 hidden: int, layers: int, ablate: str = "none", readout: str = "linear",
                 relations: str = "legacy", top_k: int = 20, dropout: float = 0.1,
                 window: int = 0, patch_len: int = 0, attn_depth: int = 0):
        super().__init__()
        self.pack = pack
        self.lr, self.weight_decay = lr, weight_decay
        # ``window`` keeps only the newest W snapshots of each sample. The label is
        # precomputed per snapshot, so shortening the input changes what the model
        # reads without touching what it predicts.
        self.window = window
        self.model = Shape(pack.cards, hidden_dim=hidden, num_layers=layers, ablate=ablate,
                           readout=readout, relations=relations, top_k=top_k,
                           dropout=dropout, patch_len=patch_len, attn_depth=attn_depth)
        self.cards = {c.name: c for c in pack.cards}
        self._val: Dict[str, List] = {}

    def _forward(self, batch):
        card = self.cards[f"MiNT{batch['network']}"]
        if self.window:
            batch = {**batch, "x": batch["x"][..., -self.window:, :]}
        pred = self.model(batch["x"], u=batch["u"], u_mask=batch["u_mask"], card=card,
                          num_nodes=batch.get("num_nodes"), deg=batch.get("deg"))
        return pred, loss_for(card, pred, batch["y"], batch["mask"])

    def training_step(self, batch, _):
        _, loss = self._forward(batch)
        self.log("train_loss", loss, batch_size=batch["x"].shape[0])
        return loss

    def validation_step(self, batch, _):
        pred, loss = self._forward(batch)
        self.log("val_loss", loss, batch_size=batch["x"].shape[0])
        # AUC is a ranking statistic over a whole fold, so it cannot be averaged
        # per batch -- the fold is accumulated and scored once per network.
        slot = self._val.setdefault(batch["network"], [[], []])
        slot[0].append(pred.detach().float().reshape(-1).cpu())
        slot[1].append(batch["y"].detach().reshape(-1).cpu())

    def on_validation_epoch_end(self):
        aucs = []
        for network, (preds, ys) in self._val.items():
            p = torch.cat(preds).numpy()
            t = torch.cat(ys).long().numpy()
            if len(set(t.tolist())) > 1:          # a one-class fold has no AUC
                aucs.append(roc_auc_score(t, p))
        self._val.clear()
        # No scorable network yet (sanity pass): log a floor so the monitor exists.
        self.log("val_auc", float(sum(aucs) / len(aucs)) if aucs else 0.0, prog_bar=False)

    def configure_optimizers(self):
        """Adam + cosine anneal, matching TIDES' MiNT_training.yaml.

        Same optimiser, lr, weight decay and schedule as the MiNT column this
        curve is compared against, so a difference is the model and not the
        training recipe.
        """
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs, eta_min=1e-5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


def loaders(networks, batch_size, seed, root=None, num_workers=6, eval_batch_size=8):
    """Fold loaders for the pack.

    ``__getitem__`` rebuilds ``P_t R`` for all W snapshots of a window, which is
    78% of its cost and the step's real bottleneck -- so loading is pushed onto
    worker processes to overlap it with the GPU. Workers change nothing
    numerically: the dataset is a pure function of its index.
    """
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for fold, shuffle in (("train", True), ("val", False)):
        ds = PackDataset(networks, root=root).use_fold(fold)
        # node_budget caps the Gram graph only while training, so evaluation runs
        # the full active set (~6.5k on the largest networks) and a [B,K,K] dense
        # graph at the training batch size will not fit. Evaluation therefore has
        # its own batch size. It is not a free knob: gram_nodes takes the
        # eligible set as any-active-across-the-batch, so this value changes the
        # graph. It is fixed across every pack, seed and the zero-shot run.
        bs = batch_size if fold == "train" else eval_batch_size
        kw = dict(pin_memory=True)          # pinned staging -> async H2D
        if num_workers:
            kw |= dict(num_workers=num_workers, persistent_workers=True, prefetch_factor=6)
        out[fold] = DataLoader(
            ds,
            batch_sampler=HomogeneousBatchSampler(ds, bs, shuffle, gen),
            collate_fn=collate_pack,
            # Private generator: a DataLoader iterator draws a worker base seed
            # from the global RNG each epoch, which would shift the model's
            # dropout stream. See shape/train.py:loaders.
            generator=torch.Generator().manual_seed(seed),
            **kw,
        )
    return out, out["train"].dataset


def train_pack(pack: str, seed: int, out_dir: str, epochs: int, batch_size: int,
               lr: float, patience: int, device: str, hidden: int, layers: int,
               weight_decay: float, precision: str, root: str | None = None,
               num_workers: int = 6, eval_batch_size: int = 8,
               ablate: str = "none", readout: str = "linear",
               overwrite: bool = False, relations: str = "legacy",
               top_k: int = 20, dropout: float = 0.1, window: int = 0,
               patch_len: int = 0, attn_depth: int = 0) -> Dict:
    configure_backends(precision)
    L.seed_everything(seed, workers=True, verbose=False)
    networks = pack_names(pack)
    # Numeric packs keep the zero-padded name the scaling runs used; a named
    # pack (a domain split, say) keeps its own name.
    stem = f"pack{int(pack):02d}" if pack.isdigit() else f"pack-{pack}"
    tag = (stem + ("" if ablate == "none" else f"-{ablate}")
           # The suffix names the directory layout the overnight run established
           # (informed unsuffixed, everything else suffixed), NOT the current
           # default. Tying it to the default would make a new linear run write
           # into the informed run's directory and silently overwrite it.
           + ("" if readout == "informed" else f"-{readout}")
           + ("" if relations == "legacy" else f"-{relations}")
           + ("" if top_k == 20 else f"-k{top_k}")
           + ("" if dropout == 0.1 else f"-do{dropout}")
           + ("" if window == 0 else f"-w{window}")
           + ("" if patch_len == 0 else f"-p{patch_len}a{attn_depth}")
           + ("" if hidden == 64 else f"-h{hidden}")
           + ("" if lr == 1e-3 else f"-lr{lr}"))
    run_dir = os.path.join(out_dir, tag, f"seed{seed}")
    done = os.path.join(run_dir, "train.json")
    if os.path.exists(done) and not overwrite:
        raise SystemExit(f"{done} exists -- refusing to overwrite a finished run. "
                         f"Pass --overwrite, or --out a different directory.")
    os.makedirs(run_dir, exist_ok=True)
    dl, train_ds = loaders(networks, batch_size, seed, root, num_workers, eval_batch_size)
    task = PackTask(train_ds, lr, weight_decay, hidden, layers, ablate, readout,
                    relations, top_k, dropout, window, patch_len, attn_depth)

    accelerator, devices = ("cpu", 1) if device == "cpu" else ("gpu", [int(device.split(":")[1])])
    if device != "cpu" and torch.cuda.is_available():
        # reset_peak_memory_stats needs an initialised context on that device
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    best = ModelCheckpoint(dirpath=run_dir, filename="best", monitor="val_auc",
                           mode="max", save_top_k=1, enable_version_counter=False)
    trainer = L.Trainer(
        max_epochs=epochs, accelerator=accelerator, devices=devices,
        precision=PRECISION[precision], gradient_clip_val=5.0, num_sanity_val_steps=0,
        callbacks=[EarlyStopping("val_auc", mode="max", patience=patience, min_delta=1e-4), best],
        logger=False, enable_progress_bar=False, enable_model_summary=False,
    )
    t0 = time.time()
    trainer.fit(task, dl["train"], dl["val"])

    row = {"pack": int(pack) if pack.isdigit() else pack, "seed": seed, "networks": networks, "top_k": top_k,
           "dropout": dropout, "window": window,
           "patch_len": patch_len, "attn_depth": attn_depth,
           "n_networks": len(networks), "epochs_run": trainer.current_epoch,
           "val_auc": float(best.best_model_score) if best.best_model_score is not None else float("nan"),
           "checkpoint": best.best_model_path,
           "hidden": hidden, "layers": layers, "lr": lr, "weight_decay": weight_decay,
           "batch_size": batch_size, "eval_batch_size": eval_batch_size, "ablate": ablate, "readout": readout,
           "relations": relations,
           "max_epochs": epochs, "precision": precision,
           "train_windows": len(train_ds),
           "seconds": round(time.time() - t0, 1),
           "peak_mem_mb": (round(torch.cuda.max_memory_allocated(device) / 2 ** 20, 1)
                           if device != "cpu" and torch.cuda.is_available() else 0.0)}
    with open(os.path.join(run_dir, "train.json"), "w") as fh:
        json.dump(row, fh, indent=2)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", nargs="+", default=["4", "8", "16"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="runs/mint-scaling")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--precision", choices=list(PRECISION), default="bf16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--root", default=None)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--ablate", choices=["none", "degrees_only", "no_graph"], default="none")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing into a run directory that already holds a finished run")
    ap.add_argument("--readout", choices=["informed", "linear"], default="informed")
    ap.add_argument("--top-k", type=int, default=20,
                    help="edges kept per row of the Gram operator")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patch-len", type=int, default=0,
                    help="patched encoder instead of Linear(W,.); required to vary --window")
    ap.add_argument("--attn-depth", type=int, default=0)
    ap.add_argument("--window", type=int, default=0,
                    help="keep only the newest W snapshots of the input; 0 = the card's W")
    ap.add_argument("--relations", default="legacy",
                    help="legacy | gram | gram+phys. MiNT has no static topology, so\n"
                         "gram is the only relation available here.")
    a = ap.parse_args()

    for pack in a.packs:
        for seed in a.seeds:
            r = train_pack(pack, seed, a.out, a.epochs, a.batch_size, a.lr, a.patience,
                           a.device, a.hidden, a.layers, a.weight_decay, a.precision, a.root,
                           a.num_workers, a.eval_batch_size, a.ablate, a.readout,
                           a.overwrite, a.relations, a.top_k, a.dropout, a.window,
                           a.patch_len, a.attn_depth)
            print(f"pack{r['pack']} seed{r['seed']}  val_auc {r['val_auc']:.4f}  "
                  f"{r['epochs_run']} ep  {r['seconds']:.0f}s  "
                  f"{r['peak_mem_mb']:.0f} MB  -> {r['checkpoint']}", flush=True)
