"""Zero-shot forecasting on a corpus the checkpoint never trained on.

The forecast head is shared across corpora (``head_key`` returns the task), and
channel indices are derived for an unseen card, so a joint checkpoint can be run
forward on a held-out corpus with no new parameters. The one exception is
Phi_scale: a per-corpus projection has no entry for an unseen corpus and is
and no per-corpus parameter remains, so a joint checkpoint runs forward on an
unseen corpus with no new weights at all.

No gradient touches the target. The target's *test* fold is scored, so the number
is comparable with the supervised column trained on the same split.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from shape.data import ShapeDataset
from shape.model import Shape, head_key
from shape.pretrain import BalancedBatchSampler, MultiCorpus, collate_pack
from shape.train import configure_backends, metrics_for


def _remap_legacy_heads(sd: Dict, cards) -> Dict:
    """Rename ``heads.forecast.*`` to the horizon-specific key.

    Forecast heads used to be shared across every horizon under one key. They are
    now keyed by horizon, so a checkpoint written before that change carries
    ``heads.forecast.*`` and would silently lose its head on a non-strict load.
    Only forecasting corpora are affected, and all of them had H=12.
    """
    hs = {c.H for c in cards if c.task == "forecast"}
    if len(hs) != 1 or not any(k.startswith("heads.forecast.") for k in sd):
        return sd
    h = hs.pop()
    return {(f"heads.forecast-{h}." + k[len("heads.forecast."):]
             if k.startswith("heads.forecast.") else k): v for k, v in sd.items()}


def build_model(meta: Dict, device: str, target: str = None) -> Shape:
    """Rebuild exactly the trained model, then load strictly.

    Every optional component has to be reconstructed from the recorded config. A
    flag missed here does not raise: the module is simply absent, its trained
    weights become *unexpected* keys, and a non-strict load drops them silently.
    The model then scores with a gate that never gates and an encoder missing its
    attention, and the number is reported as a zero-shot result. Asserting on
    ``missing_keys`` alone does not catch this, because nothing is missing.
    """
    cards = [ShapeDataset(n).card for n in meta["corpora"]]
    sd = torch.load(meta["checkpoint"], map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    sd = _remap_legacy_heads(sd, cards)
    model = Shape(cards, hidden_dim=meta["hidden"], num_layers=meta["layers"],
                  relations=meta["relations"],
                  patch_len=int(meta.get("patch_len") or 0),
                  attn_depth=int(meta.get("attn_depth") or 0),
                  rel_gate=bool(meta.get("rel_gate")),
                  layer_agg=bool(meta.get("layer_agg", True)),
                  use_covariates=bool(meta.get("covariates")),
                  covariate_readout=bool(meta.get("covariate_readout")),
                  pe_readout=bool(meta.get("pe_readout")),
                  )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"dropped trained weights: {unexpected}"
    assert all(k.startswith("heads.") for k in missing), missing
    return model.eval().to(device)


def score(model: Shape, target: str, device: str, batch_size: int,
          fold: str = "test") -> Dict:
    ds = MultiCorpus([target], root=None).use_fold(fold)
    dl = DataLoader(ds, batch_sampler=BalancedBatchSampler(
        ds, {target: batch_size}, None, False, torch.Generator().manual_seed(0)),
        collate_fn=collate_pack, num_workers=2)
    card = ds.sets[target].card
    preds, ys, masks = [], [], []
    with torch.no_grad():
        for b in dl:
            x = b["x"].to(device)
            pred = model(x, u=b["u"].to(device), u_mask=b["u_mask"].to(device), card=card,
                         edge_index=b.get("edge_index"),
                         edge_weight=b.get("edge_weight"),
                         num_nodes=b.get("num_nodes"),
                         deg=None if b["deg"] is None else b["deg"].to(device))
            preds.append(pred.float().cpu()); ys.append(b["y"])
            masks.append(None if b["mask"] is None else b["mask"])
    m = None if masks[0] is None else torch.cat(masks)
    out = metrics_for(card, torch.cat(preds), torch.cat(ys), m)
    return out | {"fold": fold, "test_windows": len(ds)}


def persistence(target: str, fold: str = "test") -> Dict:
    """y_hat = x_t repeated over the horizon, on the same test fold."""
    ds = MultiCorpus([target], root=None).use_fold(fold)
    card = ds.sets[target].card
    preds, ys, masks = [], [], []
    for i in range(len(ds)):
        it = ds[i]
        last = it["x"][:, -1, card.channel_groups["real"][0]]        # [N]
        preds.append(last.unsqueeze(-1).expand(-1, card.H).unsqueeze(0))
        ys.append(it["y"].unsqueeze(0))
        masks.append(None if it["mask"] is None else it["mask"].unsqueeze(0))
    m = None if masks[0] is None else torch.cat(masks)
    return metrics_for(card, torch.cat(preds), torch.cat(ys), m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/phase07")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--persistence", action="store_true")
    ap.add_argument("--fold", default="test", choices=("val", "test"),
                    help="which fold of the target to score")
    a = ap.parse_args()
    configure_backends(a.precision)

    if a.persistence:
        print(f"persistence  {a.target}  " +
              "  ".join(f"{k} {v:.4f}" for k, v in persistence(a.target, a.fold).items()), flush=True)

    rows: List[Dict] = []
    for f in sorted(glob.glob(os.path.join(a.runs, "*", "seed*", "train.json"))):
        meta = json.load(open(f))
        if a.target in meta["corpora"]:
            print(f"skip {meta['tag']}: {a.target} is in its training set", flush=True)
            continue
        if not os.path.exists(meta.get("checkpoint") or ""):
            print(f"skip {meta['tag']}: checkpoint missing at "
                  f"{meta.get('checkpoint')!r}", flush=True)
            continue
        # A flat Linear(W, .) encoder *is* the training window, so it cannot be
        # run on a target of a different width; a patched one can, and that is
        # the property the mixed-W arms exist to demonstrate.
        pl = int(meta.get("patch_len") or 0)
        tw = ShapeDataset(a.target).card.W
        trained = {ShapeDataset(n).card.W for n in meta["corpora"]}
        if not pl and tw not in trained:
            print(f"skip {meta['tag']}: flat encoder trained at W={sorted(trained)}, "
                  f"target is W={tw}", flush=True)
            continue
        r = score(build_model(meta, a.device), a.target, a.device, a.batch_size, a.fold)
        r |= {"tag": meta["tag"], "seed": meta["seed"], "trained_on": meta["corpora"],
              "patch_len": pl, "train_windows": sorted(trained), "target_window": tw,
              "k": len(meta["corpora"]), "target": a.target,
              "relations": meta.get("relations"), "checkpoint": meta["checkpoint"]}
        rows.append(r)
        print(f"{meta['tag']:22s} k={r['k']} seed{r['seed']}  MAE {r['MAE']:8.3f}  "
              f"RMSE {r['RMSE']:8.3f}", flush=True)
    if a.out:
        with open(a.out, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
