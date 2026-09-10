"""Strictly-frozen zero-shot evaluation of a pack checkpoint on held-out networks.

No gradient ever touches a test network: the checkpoint is loaded, put in eval
mode, and run forward over the target's *test* fold. MiNT's protocol adds a
forward pass over train+val first, but only to repopulate the node memory of a
memory-based TGN; SHAPE is stateless per window, so that pass has no analogue
here and scoring the test fold directly is the faithful translation.

The test fold is scored (not the whole series) so the number stays comparable
with a supervised column trained on the same split.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from shape.data import ShapeDataset
from shape.model import Shape
from shape.packs import (HomogeneousBatchSampler, PackDataset, collate_pack,
                         dataset_name, pack_names)
from shape.train import PRECISION, configure_backends


def load_checkpoint(path: str, hidden: int, layers: int, device: str) -> Shape:
    """Rebuild the backbone and load the trained weights.

    The cards recorded next to the checkpoint decide the module shapes. Every
    MiNT corpus is graph-classify with identical channel groups and no static
    features, so the pack's own cards are enough -- an unseen network needs no
    new parameters, which is what makes the transfer zero-shot rather than a
    partially reinitialised model.
    """
    meta = json.load(open(os.path.join(os.path.dirname(path), "train.json")))
    # A pack run records MiNT network names; a multi-corpus pretraining run records
    # corpus names directly, and those already include the traffic cards that gave
    # the checkpoint its extra heads. Both rebuild the same backbone.
    names = ([dataset_name(n) for n in meta["networks"]] if "networks" in meta
             else meta["corpora"])
    cards = [ShapeDataset(n).card for n in names]
    state = {k[len("model."):]: v for k, v in
             torch.load(path, map_location="cpu", weights_only=False)["state_dict"].items()
             if k.startswith("model.")}
    # A gram+phys checkpoint carries one normalised adjacency buffer per traffic
    # corpus. They are rebuilt from the checkpoint itself rather than re-derived,
    # so the load can stay strict: a silently dropped weight here would be
    # reported as a zero-shot score.
    phys = {k[len("phys_"):]: v for k, v in state.items() if k.startswith("phys_")}
    model = Shape(cards, hidden_dim=hidden, num_layers=layers,
                  # train_pack records the readout; pretrain does not and takes
                  # Shape's own default, so the fallback differs by run kind.
                  readout=meta.get("readout", "informed" if "networks" in meta else "linear"),
                  relations=meta.get("relations", "legacy"),
                  use_scale=meta.get("scale", False),
                  scale_shared=meta.get("scale_shared", False),
                  patch_len=meta.get("patch_len", 0) or 0,
                  use_covariates=meta.get("covariates", False),
                  # Every optional component has to be rebuilt from the recorded
                  # config: a flag missed here shows up as an unexpected key, and
                  # under a non-strict load it would silently drop trained weights
                  # and be reported as a zero-shot score.
                  rel_gate=meta.get("rel_gate", False),
                  rel_inject=meta.get("rel_inject", False),
                  soft_topk=meta.get("soft_topk", False),
                  attn_depth=meta.get("attn_depth", 0) or 0,
                  covariate_readout=meta.get("covariate_readout", False),
                  pe_readout=meta.get("pe_readout", False),
                  phys_adj=phys or None)
    model.load_state_dict(state)
    return model.to(device).eval(), meta


@torch.no_grad()
def score_network(model: Shape, network: str, device: str, batch_size: int,
                  precision: str) -> Dict:
    ds = PackDataset([network]).use_fold("test")
    card = ds.cards[0]
    dl = DataLoader(ds, batch_sampler=HomogeneousBatchSampler(ds, batch_size, False),
                    collate_fn=collate_pack)
    preds, ys = [], []
    autocast = torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                              enabled=(precision == "bf16"), cache_enabled=False)
    for batch in dl:
        with autocast:
            pred = model(batch["x"].to(device), u=batch["u"].to(device),
                         u_mask=batch["u_mask"].to(device), card=card,
                         num_nodes=batch.get("num_nodes"),
                         deg=None if batch.get("deg") is None else batch["deg"].to(device))
        preds.append(pred.float().reshape(-1).cpu())
        ys.append(batch["y"].reshape(-1).cpu())
    p = torch.cat(preds).numpy()
    t = torch.cat(ys).long().numpy()
    defined = len(set(t.tolist())) > 1
    return {"network": network, "auc": float(roc_auc_score(t, p)) if defined else float("nan"),
            "n_test": int(t.size), "pos_rate": round(float(t.mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/mint-scaling")
    ap.add_argument("--test-pack", default="test")
    ap.add_argument("--out", default=None, help="default <runs>/zeroshot.jsonl")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="must match train_pack --eval-batch-size: it changes the eligible node set")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--precision", choices=list(PRECISION), default="bf16")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    configure_backends(a.precision)
    out_path = a.out or os.path.join(a.runs, "zeroshot.jsonl")
    # ModelCheckpoint writes best.ckpt during training; train.json is written only
    # once the run finishes. A checkpoint without it belongs to a run still in
    # progress -- skipping it lets this be re-run over a live directory instead of
    # aborting the whole pass on the first unfinished pack.
    ckpts = [c for c in sorted(glob.glob(os.path.join(a.runs, "*", "seed*", "best.ckpt")))
             if os.path.exists(os.path.join(os.path.dirname(c), "train.json"))]
    if not ckpts:
        raise SystemExit(f"no finished runs under {a.runs}")
    targets = pack_names(a.test_pack)

    rows: List[Dict] = []
    with open(out_path, "w") as fh:
        for ckpt in ckpts:
            model, meta = load_checkpoint(ckpt, a.hidden, a.layers, a.device)
            trained = set(meta.get("networks") or
                          [n[len("MiNT"):] for n in meta["corpora"] if n.startswith("MiNT")])
            overlap = trained & set(targets)
            assert not overlap, f"{ckpt} trained on test networks: {sorted(overlap)}"
            for network in targets:
                r = score_network(model, network, a.device, a.batch_size, a.precision)
                r |= {"pack": meta.get("pack", meta.get("tag")), "seed": meta["seed"],
                      "checkpoint": ckpt,
                      "val_auc": meta.get("val_auc", meta.get("val_macro")),
                      "ablate": meta.get("ablate", "none"),
                      "readout": meta.get("readout", "informed"),
                      "relations": meta.get("relations", "legacy")}
                rows.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
            done = [r["auc"] for r in rows if r["checkpoint"] == ckpt]
            print(f"{meta.get('pack', meta.get('tag'))} seed{meta['seed']}  "
                  f"mean zero-shot AUC {sum(done)/len(done):.4f} over {len(done)} networks",
                  flush=True)
    print(f"\n{len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
