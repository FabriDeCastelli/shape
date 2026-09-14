"""Phase 02: one shared backbone pretrained over heterogeneous corpora.

Every corpus keeps its own task, metric, node count and channel count; the
backbone is shared and only the heads are per-task (or per-dataset, for node
classification, where the class count differs).

Two things make a mixed corpus trainable that a plain concatenation does not:

*Balanced exposure.* Corpora differ in length by three orders of magnitude
(ChickenPox 407 train windows, PeMS-Bay 36,474). Proportional sampling would make
this "PeMS with a rounding error", so each epoch draws the same number of batches
from every corpus.

*A comparable loss.* ``L = sum_d L_tau(d) / b_d`` with ``b_d`` the loss of the
trivial predictor on that corpus -- persistence for forecasting, the training
class prior for classification. Raw losses are not comparable across a masked MAE
on z-scored traffic and a 40-way cross-entropy, so an unnormalised sum is a
silent weighting by whichever corpus happens to have the largest numbers.
``b_d`` is measured on the train fold, never assumed.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from shape.data import DatasetCard, ShapeDataset
from shape.model import Shape, fusion_slot_masses, loss_for
from shape.tracking import (finish as wandb_finish, wandb_logger,
                            experiment_name, log_test)
from shape.packs import collate_pack
from shape.train import PRECISION, configure_backends, metrics_for

logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", ".*does not have many workers.*")

# Batch size per corpus. Node counts span 20 to 169,343 and the Gram graph is
# dense over whatever enters it, so one global batch size would either waste the
# small corpora or OOM on the large ones.
BATCH = {"DBLP10": 1, "ArXiv": 1, "PemsBay": 16, "Pems07": 16}
DEFAULT_BATCH = 32
# MiNT node counts run to 118k and the eval Gram is dense over the active set,
# so every network in the collection takes the pack runs' batch size, not just
# the four that were named individually when the map was written.
MINT_BATCH = 8


def batch_for(name: str) -> int:
    return BATCH.get(name, MINT_BATCH if name.startswith("MiNT") else DEFAULT_BATCH)


# Which reported domain a corpus belongs to. The cards cannot carry this: their
# schema is fixed and shared, and an extra key breaks every loader (phase 00 6.2).
SPEED = {"MetrLA", "PemsBay", "PeMSD7M", "PeMSD7L"}


def domain_of(card) -> str:
    """flow | speed | graph. Three domains, because the paper reports three."""
    if card.task != "forecast":
        return "graph"
    return "speed" if card.name in SPEED else "flow"


def domain_caps(cards, budget: int) -> Dict[str, int]:
    """Per-corpus batch caps that give every *domain* the same share of an epoch.

    Equal batches per corpus hands the epoch to whichever domain has the most
    corpora -- 16 MiNT networks against 3 speed corpora is a 5:1 weighting that
    has nothing to do with either domain's importance. Splitting the budget by
    domain first and then within it is the only setting in which "the cost of
    adding a domain" is a statement about interference rather than about
    sampling.
    """
    by: Dict[str, List[str]] = defaultdict(list)
    for c in cards:
        by[domain_of(c)].append(c.name)
    per_domain = budget // max(len(by), 1)
    return {n: max(1, per_domain // len(names)) for names in by.values() for n in names}


class MultiCorpus(Dataset):
    """Several corpora behind one index, each sample tagged with its corpus."""

    def __init__(self, names: Sequence[str], root: Optional[str] = None,
                 use_pe: bool = False):
        kw = {"use_pe": use_pe} if root is None else {"root": root, "use_pe": use_pe}
        self.names = list(names)
        self.sets: Dict[str, ShapeDataset] = {n: ShapeDataset(n, **kw) for n in self.names}
        self.cards: List[DatasetCard] = [self.sets[n].card for n in self.names]
        self.index: List[tuple] = []

    def use_fold(self, fold: str) -> "MultiCorpus":
        """Restrict to one fold of every corpus.

        A temporal split selects samples; a node split is transductive, so every
        fold walks every sample and the node mask decides what counts.
        """
        self.index = []
        for n in self.names:
            ds = self.sets[n]
            idx = (ds.splits[fold].tolist() if ds.card.split_kind == "temporal"
                   else list(range(len(ds))))
            self.index += [(n, int(i)) for i in idx]
        return self

    def node_mask(self, name: str, fold: str) -> Optional[torch.Tensor]:
        ds = self.sets[name]
        return None if ds.card.split_kind == "temporal" else ds.splits[fold].bool()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict:
        name, sample = self.index[i]
        item = self.sets[name][sample]
        item["network"] = name           # collate_pack's key for "which corpus"
        return item


class BalancedBatchSampler(torch.utils.data.Sampler):
    """Equal batches per corpus per epoch, homogeneous, interleaved.

    Homogeneous because N and C differ across corpora, so a mixed batch cannot be
    stacked. Equal because exposure proportional to length would hand the epoch to
    the longest corpus. ``cap`` batches are drawn from each corpus without
    replacement where possible; a corpus shorter than ``cap`` batches is cycled,
    which is what oversampling a small domain means. ``cap=None`` means every
    batch of every corpus, which is what validation and test want.
    """

    def __init__(self, dataset: MultiCorpus, batch: Dict[str, int],
                 cap: Optional[int] | Dict[str, int],
                 shuffle: bool, generator: Optional[torch.Generator] = None,
                 nets: Optional[int] = None):
        self.dataset, self.batch, self.cap = dataset, batch, cap
        self.shuffle, self.generator, self.nets = shuffle, generator, nets
        self._deck: List[str] = []
        self._by_corpus: Dict[str, List[int]] = defaultdict(list)
        for flat, (name, _) in enumerate(dataset.index):
            self._by_corpus[name].append(flat)

    def _names(self) -> List[str]:
        """The corpora this epoch draws from.

        ``nets`` bounds an epoch to a subset, so the cost of an epoch stops
        growing with the size of the pack. The subset is dealt from a shuffled
        deck that is only reshuffled once exhausted, rather than sampled
        independently each epoch: drawing independently leaves a corpus unseen
        with probability ``(1 - nets/k)**epochs``, which for 8-of-64 over 35
        epochs is about 1% per corpus, and a pre-training corpus that is never
        visited is the one outcome this must not have. Dealing guarantees every
        corpus appears once per ``ceil(k/nets)`` epochs.

        Only the shuffled (train) loader subsamples: validation must score the
        same windows every epoch or early stopping compares two different
        quantities.
        """
        names = list(self._by_corpus)
        if not self.shuffle or not self.nets or self.nets >= len(names):
            return names
        out: List[str] = []
        while len(out) < self.nets:
            if not self._deck:
                order = torch.randperm(len(names), generator=self.generator).tolist()
                # a refilled deck must not hand back a corpus this epoch already
                # holds, or that corpus silently takes a double share of it
                self._deck = [names[i] for i in order if names[i] not in out]
            out.append(self._deck.pop())
        return out

    def _epoch(self) -> List[List[int]]:
        out: List[List[int]] = []
        for name in self._names():
            flat = self._by_corpus[name]
            bs = self.batch.get(name, batch_for(name))
            if self.shuffle:
                perm = torch.randperm(len(flat), generator=self.generator).tolist()
                flat = [flat[p] for p in perm]
            batches = [flat[s:s + bs] for s in range(0, len(flat), bs)]
            cap = self.cap.get(name) if isinstance(self.cap, dict) else self.cap
            if cap is not None:
                while len(batches) < cap:           # short corpus: cycle it
                    batches += batches[:cap - len(batches)]
                batches = batches[:cap]
            out += batches
        if self.shuffle:
            order = torch.randperm(len(out), generator=self.generator).tolist()
            out = [out[o] for o in order]
        return out

    def __iter__(self):
        yield from self._epoch()

    def __len__(self) -> int:
        # Deterministic, and deliberately does NOT draw from the generator --
        # see phase_00 6.4(a), where HomogeneousBatchSampler.__len__ moved the
        # batch order every time the framework asked for a length.
        n_corpora = len(self._by_corpus)
        if self.shuffle and self.nets:
            n_corpora = min(n_corpora, self.nets)
        if isinstance(self.cap, dict):
            caps = sorted(self.cap.get(n, 0) for n in self._by_corpus)
            return sum(caps[-n_corpora:]) if n_corpora < len(caps) else sum(caps)
        if self.cap is not None:
            return self.cap * n_corpora
        return sum(-(-len(v) // self.batch.get(n, batch_for(n)))
                   for n, v in self._by_corpus.items())


def trivial_baseline(ds: ShapeDataset, fold: str = "train", limit: int = 512) -> float:
    """``b_d``: the loss of the trivial predictor on this corpus.

    Forecasting -> persistence, the last observed step held for all H. That is
    the standard reference for a traffic forecaster and it is scale-free in the
    same units the model's own loss uses.

    Classification -> the training class prior as a constant prediction, scored
    with the card's own loss. Not accuracy: the objective divides a loss by
    ``b_d``, so ``b_d`` has to be a loss.
    """
    card = ds.card
    idx = (ds.splits[fold].tolist() if card.split_kind == "temporal" else list(range(len(ds))))
    idx = idx[::max(1, len(idx) // limit)][:limit]
    mask_nodes = None if card.split_kind == "temporal" else ds.splits[fold].bool()

    if card.task == "forecast":
        total, n = 0.0, 0
        real = card.channel_groups.get("real")
        for i in idx:
            it = ds[i]
            x, y = it["x"], it["y"]
            last = x[:, -1, real[0]] if real else x[:, -1, 0]        # [N]
            pred = last.unsqueeze(-1).expand(-1, card.H)             # [N, H]
            m = it["mask"]
            total += float(loss_for(card, pred.unsqueeze(0), y.unsqueeze(0),
                                    None if m is None else m.unsqueeze(0)))
            n += 1
        return total / max(n, 1)

    ys = []
    for i in idx:
        y = ds[i]["y"]
        ys.append(y[mask_nodes] if (mask_nodes is not None and y.dim() > 0
                                    and y.shape[0] == card.num_nodes) else y)
    y = torch.cat([v.reshape(-1) for v in ys])
    if card.metric == "accuracy":
        prior = torch.bincount(y.long(), minlength=card.num_classes).float()
        prior = (prior / prior.sum()).clamp(min=1e-9)
        logits = prior.log().unsqueeze(0).expand(y.numel(), -1)
        return float(loss_for(card, logits, y, None))
    rate = float(y.float().mean().clamp(1e-6, 1 - 1e-6))
    logit = torch.logit(torch.tensor(rate)).expand(y.numel())
    return float(loss_for(card, logit, y, None))


def truncate_window(batch: Dict, k: int) -> Dict:
    """Keep only the newest ``k`` steps of the window.

    Sinusoidal positions are measured from the present, so the newest patch sits
    at distance 0 whatever ``k`` is: a truncated window is exactly the view the
    model gets when deployed against a shorter history. The calendar covariates
    are indexed by window position and have to be cut with the signal.
    """
    if k >= batch["x"].shape[-2]:
        return batch
    out = {**batch, "x": batch["x"][..., -k:, :]}
    for key in ("u", "u_mask"):
        if batch.get(key) is not None:
            out[key] = batch[key][..., -k:, :]
    return out


class PretrainTask(L.LightningModule):
    """One backbone, per-corpus baseline-normalised loss, macro validation score."""

    def __init__(self, corpus: MultiCorpus, base: Dict[str, float], lr: float,
                 weight_decay: float, hidden: int, layers: int, eval_chunk: int = 0,
                 freeze_trunk: bool = False, relations: str = "gram+topo",
                 patch_len: int = 0, use_covariates: bool = False,
                 rel_gate: bool = True, layer_agg: bool = True,
                 attn_depth: int = 2,
                 pe_readout: bool = False, window_scales=()):
        super().__init__()
        self.base = base
        self.window_scales = list(window_scales)
        self.lr_, self.wd_ = lr, weight_decay
        self.names = corpus.names
        self.cards = {c.name: c for c in corpus.cards}
        self.masks = {f: {n: corpus.node_mask(n, f) for n in corpus.names}
                      for f in ("train", "val", "test")}
        self.model = Shape(corpus.cards, hidden_dim=hidden, num_layers=layers,
                           eval_chunk=eval_chunk, relations=relations, patch_len=patch_len,
                           use_covariates=use_covariates, rel_gate=rel_gate,
                           layer_agg=layer_agg, attn_depth=attn_depth,
                           pe_readout=pe_readout)
        if freeze_trunk:
            for name, p in self.model.named_parameters():
                p.requires_grad = name.startswith("heads.")
        self._val: Dict[str, List[float]] = defaultdict(list)
        self._test: Dict[str, List[Dict]] = defaultdict(list)

    def _forward(self, batch, fold: str):
        name = batch["network"]
        card = self.cards[name]
        pred = self.model(batch["x"], pe=batch.get("pe"),
                          edge_index=batch.get("edge_index"),
                          edge_weight=batch.get("edge_weight"),
                          u=batch["u"], u_mask=batch["u_mask"], card=card,
                          num_nodes=batch.get("num_nodes"), deg=batch.get("deg"))
        y, mask = batch["y"], batch["mask"]
        nodes = self.masks[fold][name]
        if nodes is not None:                     # transductive: score this fold's nodes
            pred, y = pred[:, nodes], y[:, nodes]
            mask = None if mask is None else mask[:, nodes]
        return name, pred, y, mask, loss_for(card, pred, y, mask)

    def training_step(self, batch, _):
        if self.window_scales:
            i = int(torch.randint(len(self.window_scales), (1,)).item())
            batch = truncate_window(batch, self.window_scales[i])
        name, _, _, _, loss = self._forward(batch, "train")
        # The division is the whole point: a masked MAE on z-scored traffic and a
        # 40-way cross-entropy are not on one scale, and an unnormalised sum is a
        # silent weighting by whichever corpus has the larger numbers.
        scaled = loss / self.base[name]
        self.log("train_loss", scaled, batch_size=batch["x"].shape[0])
        return scaled

    def validation_step(self, batch, i):
        if self.window_scales:
            batch = truncate_window(batch, self.window_scales[i % len(self.window_scales)])
        name, _, _, _, loss = self._forward(batch, "val")
        self._val[name].append(float(loss) / self.base[name])

    def on_validation_epoch_end(self):
        # Macro over corpora, not pooled: val fold sizes span 51 to 10,421, so a
        # pooled figure would be chosen by PeMS-Bay alone.
        per = {n: sum(v) / len(v) for n, v in self._val.items() if v}
        self._val.clear()
        for n, v in per.items():
            self.log(f"val/{n}", v)
        self.log("val_macro", sum(per.values()) / len(per) if per else float("inf"))

    def test_step(self, batch, _):
        name, pred, y, mask, _ = self._forward(batch, "test")
        self._test[name].append({"pred": pred.float().cpu(), "y": y.cpu(),
                                 "mask": None if mask is None else mask.cpu()})

    def on_test_epoch_end(self):
        self.metrics = {}
        for name, outs in self._test.items():
            cat = lambda k: torch.cat([o[k] for o in outs])
            m = None if outs[0]["mask"] is None else cat("mask")
            self.metrics[name] = metrics_for(self.cards[name], cat("pred"), cat("y"), m)
        self._test.clear()

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.lr_, weight_decay=self.wd_)


def loaders(names, fold_batch, cap, seed, root=None, num_workers=2, use_pe=False,
            val_cap=None, nets=None):
    out = {}
    for fold, shuffle in (("train", True), ("val", False), ("test", False)):
        ds = MultiCorpus(names, root=root, use_pe=use_pe).use_fold(fold)
        kw = dict(pin_memory=True)
        if num_workers:
            kw |= dict(num_workers=num_workers, persistent_workers=True, prefetch_factor=4)
        # test is never capped: it is the number the paper reports.
        fold_cap = cap if fold == "train" else (val_cap if fold == "val" else None)
        out[fold] = DataLoader(
            ds,
            batch_sampler=BalancedBatchSampler(
                ds, fold_batch, fold_cap, shuffle,
                torch.Generator().manual_seed(seed), nets=nets),
            collate_fn=collate_pack,
            generator=torch.Generator().manual_seed(seed), **kw)
    return out, MultiCorpus(names, root=root, use_pe=use_pe).use_fold("train")


def measure_baselines(names, root=None, cache="runs/phase02/baselines.json") -> Dict[str, float]:
    """``b_d`` per corpus, measured once and reused so every run shares them."""
    have = json.load(open(cache)) if os.path.exists(cache) else {}
    todo = [n for n in names if n not in have]
    if todo:
        for n in todo:
            ds = ShapeDataset(n, **({} if root is None else {"root": root}))
            have[n] = trivial_baseline(ds)
            print(f"  b_d[{n:14}] = {have[n]:.6f}   ({ds.card.task}, {ds.card.metric})", flush=True)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(have, open(cache, "w"), indent=2, sort_keys=True)
    return {n: have[n] for n in names}


def pretrain(names, seed, out_dir, epochs, cap, lr, patience, device, hidden, layers,
             weight_decay, precision, root=None, num_workers=2, eval_chunk=0,
             freeze_trunk=False, init_from=None, tag="joint",
             relations="gram+topo", batch=0, patch_len=12,
             domain_balance=False, covariates=False, rel_gate=True, attn_depth=2,
             layer_agg=True, pe_readout=False, window_scales=(),
             val_cap=0, nets=0) -> Dict:
    configure_backends(precision)
    L.seed_everything(seed, workers=True, verbose=False)
    run_dir = os.path.join(out_dir, tag, f"seed{seed}")
    done = os.path.join(run_dir, "train.json")
    if os.path.exists(done):
        raise SystemExit(f"{done} exists -- refusing to overwrite a finished run.")
    os.makedirs(run_dir, exist_ok=True)

    base = measure_baselines(names, root)
    fold_batch = {n: batch or batch_for(n) for n in names}
    # cap 0 means "every batch of every corpus each epoch": per-epoch exposure per
    # corpus then equals what a single-corpus run gets, which is the only setting
    # in which a joint/single comparison is not a budget comparison.
    # A per-domain budget replaces the flat cap: see domain_caps. cap is then the
    # *total* batches per epoch rather than the batches per corpus.
    if domain_balance:
        probe = MultiCorpus(names, root=root)
        caps = domain_caps(probe.cards, cap)
        del probe
        print("  per-corpus caps: " + ", ".join(f"{n}={c}" for n, c in sorted(caps.items())),
              flush=True)
    else:
        caps = cap or None
    if nets and nets < len(names):
        print(f"  sampling {nets} of {len(names)} corpora per epoch", flush=True)
    if val_cap:
        print(f"  validation capped at {val_cap} batches per corpus", flush=True)
    dl, train_ds = loaders(names, fold_batch, caps, seed, root, num_workers,
                           use_pe=pe_readout, val_cap=val_cap or None, nets=nets or None)
    task = PretrainTask(train_ds, base, lr, weight_decay, hidden, layers, eval_chunk,
                        freeze_trunk, relations, patch_len, use_covariates=covariates, rel_gate=rel_gate,
                        layer_agg=layer_agg, attn_depth=attn_depth, pe_readout=pe_readout,
                        window_scales=window_scales)
    if init_from:
        state = torch.load(init_from, map_location="cpu", weights_only=False)["state_dict"]
        missing = task.load_state_dict(state, strict=False)
        print(f"  loaded {init_from}  (missing {len(missing.missing_keys)}, "
              f"unexpected {len(missing.unexpected_keys)})", flush=True)

    accelerator, devices = ("cpu", 1) if device == "cpu" else ("gpu", [int(device.split(":")[1])])
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    best = ModelCheckpoint(dirpath=run_dir, filename="best", monitor="val_macro",
                           mode="min", save_top_k=1, enable_version_counter=False)
    wb = wandb_logger(experiment_name("loo", seed, run_dir, names, tag),
                      group=tag, config=dict(
        protocol="loo", corpora=list(names), n_corpora=len(names), seed=seed,
        hidden=hidden, layers=layers, lr=lr, cap=cap, relations=relations,
        rel_gate=rel_gate, patch_len=patch_len, attn_depth=attn_depth,
        layer_agg=layer_agg,
        precision=precision))
    trainer = L.Trainer(
        max_epochs=epochs, accelerator=accelerator, devices=devices,
        precision=PRECISION[precision], gradient_clip_val=5.0, num_sanity_val_steps=0,
        callbacks=[EarlyStopping("val_macro", mode="min", patience=patience,
                                 min_delta=1e-5), best],
        logger=wb or False, enable_progress_bar=False, enable_model_summary=False)
    t0 = time.time()
    trainer.fit(task, dl["train"], dl["val"])
    trainer.test(task, dl["test"], ckpt_path=best.best_model_path, verbose=False)
    log_test(task.metrics, best.best_model_path)
    wandb_finish()

    row = {"corpora": list(names), "n_corpora": len(names), "seed": seed, "tag": tag,
           "protocol": "loo",
           # A relation whose branch stayed dead leaves its slot at exact zero.
           "fusion_slots": fusion_slot_masses(task.model),
           "epochs_run": trainer.current_epoch, "cap": cap,
           "val_cap": val_cap, "nets_per_epoch": nets,
           "val_macro": float(best.best_model_score) if best.best_model_score is not None else float("nan"),
           "checkpoint": best.best_model_path, "baselines": base, "batch": fold_batch,
           "hidden": hidden, "layers": layers, "lr": lr, "weight_decay": weight_decay,
           "max_epochs": epochs, "precision": precision, "eval_chunk": eval_chunk,
           "window_scales": list(window_scales),
           "freeze_trunk": freeze_trunk, "init_from": init_from,
           "relations": relations, "patch_len": patch_len,
           "domain_balance": domain_balance, "covariates": covariates,
           "rel_gate": rel_gate, "layer_agg": layer_agg,
           "attn_depth": attn_depth, "pe_readout": pe_readout,
           # Recorded, not assumed: a probe that silently trained the trunk would
           # answer a different question than the one asked.
           "n_trainable": sum(p.numel() for p in task.model.parameters() if p.requires_grad),
           "n_params": sum(p.numel() for p in task.model.parameters()),
           "train_batches_per_epoch": len(dl["train"].batch_sampler),
           "seconds": round(time.time() - t0, 1),
           "peak_mem_mb": (round(torch.cuda.max_memory_allocated(device) / 2 ** 20, 1)
                           if device != "cpu" and torch.cuda.is_available() else 0.0),
           "metrics": task.metrics}
    for r in task.model.rel_names:
        assert row["fusion_slots"][r] > 0, (
            f"relation {r!r} never left its zero init: its branch was dead all run")
    json.dump(row, open(done, "w"), indent=2)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs/phase02")
    ap.add_argument("--tag", default="joint")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--cap", type=int, default=64,  # 0 = full proportional pass
                    help="batches drawn from EVERY corpus each epoch (balanced exposure)")
    ap.add_argument("--val-cap", type=int, default=0,
                    help="validation batches per corpus, 0 = every window. The "
                         "test fold is never capped.")
    ap.add_argument("--nets-per-epoch", type=int, default=0,
                    help="draw each epoch from this many corpora, resampled every "
                         "epoch, 0 = all of them. Fixes the cost of an epoch "
                         "independently of how large the pack is.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--window-scales", type=int, nargs="*", default=[],
                    help="train and validate on a random one of these window "
                         "lengths, each the newest steps of the full window; "
                         "every value must divide evenly by --patch-len")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--precision", choices=list(PRECISION), default="bf16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--root", default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--eval-chunk", type=int, default=0)
    ap.add_argument("--freeze-trunk", action="store_true",
                    help="train the heads only: the frozen linear probe")
    ap.add_argument("--init-from", default=None, help="checkpoint to start from")
    ap.add_argument("--no-layer-agg", dest="layer_agg", action="store_false",
                    help="read H^(L) only instead of eq:pi (ablation arm)")
    ap.add_argument("--relations", default="gram+topo",
                    help="default is \\MODEL{} as defined in the paper: the learned\n"
                         "Gram relation plus the observed topology. Ablation values:\n"
                         "gram | topo | ident | none.")
    ap.add_argument("--domain-balance", action="store_true",
                    help="give each domain (flow | speed | graph) an equal share of "
                         "the epoch; --cap is then the total batches per epoch")
    ap.add_argument("--patch-len", type=int, default=12,
                    help="patch the input window instead of a flat Linear(W,.); "
                         "0 = off. Non-zero is what allows mixed W across corpora.")
    ap.add_argument("--batch", type=int, default=0,
                    help="one batch size for every corpus, overriding BATCH")
    ap.add_argument("--covariates", action="store_true",
                    help="feed the calendar covariates u = [tod, dow, doy, t_norm, dt]; "
                         "they are derived from the card's start_date and freq, so no "
                         "corpus rebuild is needed, and they are not node-indexed, so "
                         "they transfer to an unseen network")
    ap.add_argument("--no-rel-gate", dest="rel_gate", action="store_false",
                    help="disable eq:relation-gate (ablation only; on by default)")
    ap.add_argument("--rel-gate", dest="rel_gate", action="store_true", default=True,
                    help="gate the aggregated message per node and channel, so a "
                         "relation that is absent on an unseen corpus can be discounted")
    ap.add_argument("--attn-depth", type=int, default=2,
                    help="causal self-attention layers over the patch sequence; needs --patch-len")
    ap.add_argument("--pe-readout", action="store_true",
                    help="modulate the readout by the supra-Laplacian positional "
                         "encoding of the window's topology (MiNT only; corpora "
                         "without a precomputed PE are untouched)")
    a = ap.parse_args()

    for seed in a.seeds:
        r = pretrain(a.corpora, seed, a.out, a.epochs, a.cap, a.lr, a.patience, a.device,
                     a.hidden, a.layers, a.weight_decay, a.precision, a.root,
                     a.num_workers, a.eval_chunk, a.freeze_trunk, a.init_from, a.tag,
                     a.relations, a.batch, a.patch_len, a.domain_balance, a.covariates,
                     a.rel_gate, a.attn_depth, a.layer_agg, a.pe_readout,
                     a.window_scales, a.val_cap, a.nets_per_epoch)
        print(f"\n{a.tag} seed{seed}  val_macro {r['val_macro']:.4f}  "
              f"{r['epochs_run']} ep  {r['seconds']:.0f}s", flush=True)
        for n, m in r["metrics"].items():
            print(f"  {n:14} " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()
                                           if k in ("MAE", "MSE", "MicroF1", "MacroF1", "AUC")),
                  flush=True)
