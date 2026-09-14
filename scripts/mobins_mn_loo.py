#!/usr/bin/env python3
"""MOBINS zero-shot: train jointly on every city but one, test on the held-out one.

The decoders carry no parameter that depends on N -- ``head_n`` is
``Linear(hidden, d*H)`` applied per node and ``ODDecoder`` forms the N^2 block
with a pairwise einsum -- so one model trained on Busan and Daegu emits a
correctly shaped prediction for Seoul's 128 nodes without any surgery. That is
what makes this task transferable at all.

Two constraints decide the pool. The heads fix ``d`` and ``H``, so a corpus can
only join a pool of equal node-feature count and equal cycle; that admits
{Seoul, Busan, Daegu} (d=2, cycle=24) and excludes NYC (d=1) and the two
epidemic corpora (cycle=1). And N differs per city, so a batch may not mix them:
the sampler below deals whole batches from one corpus at a time.

    python scripts/mobins_mn_loo.py --target MobinsSeoulMN --device cuda:4
"""
from __future__ import annotations

import argparse, dataclasses, json, math, os, time
import torch
from torch.utils.data import DataLoader, Dataset

from shape.model import Shape
from shape.tracking import finish as wandb_finish, log_test, wandb_logger
from scripts.mobins_mn import MNSet, ODDecoder, mae_parts, ROOT

POOL = ["MobinsSeoulMN", "MobinsBusanMN", "MobinsDaeguMN"]   # d=2, cycle=24


class MultiCity(Dataset):
    """The pool's windows behind one index, each tagged with its corpus."""

    def __init__(self, corpora, fold, pred_day, od_input):
        self.sets, self.index = [], []
        for ci, name in enumerate(corpora):
            s = MNSet(name, fold, pred_day, od_input=od_input)
            self.sets.append(s)
            self.index += [(ci, i) for i in range(len(s))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ci, j = self.index[i]
        return {**self.sets[ci][j], "corpus": ci}


class OneCityBatches(torch.utils.data.Sampler):
    """Batches that never mix corpora: N differs per city, so a mixed batch
    could not be stacked. Batch order is shuffled, so the optimiser still sees
    the cities interleaved."""

    def __init__(self, ds, batch_size, shuffle, generator=None):
        self.batches = []
        by_c = {}
        for pos, (ci, _) in enumerate(ds.index):
            by_c.setdefault(ci, []).append(pos)
        for ci, rows in by_c.items():
            self.batches += [rows[k:k + batch_size] for k in range(0, len(rows), batch_size)]
        self.shuffle, self.generator = shuffle, generator

    def __iter__(self):
        order = (torch.randperm(len(self.batches), generator=self.generator).tolist()
                 if self.shuffle else range(len(self.batches)))
        for i in order:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)


def run(target, seed, device, epochs, patience, batch, lr, hidden, layers,
        patch_len, rank, pred_day, out_path, od_input=True, weight_decay=0.0):
    torch.manual_seed(seed)
    pool = [c for c in POOL if c != target]
    if target not in POOL:
        raise SystemExit(f"{target} is not in the d=2/cycle=24 pool {POOL}")

    tr = MultiCity(pool, "train", pred_day, od_input)
    va = MultiCity(pool, "val", pred_day, od_input)
    te = MNSet(target, "test", pred_day, od_input=od_input)

    # Every card in the pool shares d and H; the trunk is built from the first
    # and the held-out corpus reuses it, which is the whole point.
    c0 = tr.sets[0].card
    d, H = c0.C, tr.sets[0].H
    c_in = d + max(s.card.num_nodes for s in tr.sets) if od_input else d
    trunk_card = dataclasses.replace(c0, task="forecast", H=H, H_max=H,
                                     C=c_in, channel_groups={"real": [0, c_in]})
    trunk = Shape([trunk_card], hidden_dim=hidden, num_layers=layers,
                  patch_len=patch_len).to(device)
    head_n = torch.nn.Linear(hidden, d * H).to(device)
    head_o = ODDecoder(hidden, H, rank).to(device)
    params = list(trunk.parameters()) + list(head_n.parameters()) + list(head_o.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    n_par = sum(p.numel() for p in params)

    g = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(tr, batch_sampler=OneCityBatches(tr, batch, True, g), num_workers=2)
    dl_va = DataLoader(va, batch_sampler=OneCityBatches(va, batch, False), num_workers=2)
    dl_te = DataLoader(te, batch_size=batch, shuffle=False, num_workers=2)

    wb = wandb_logger(f"mobins-mn-LOO-{target}-p{pred_day}d-seed{seed}",
                      group=f"mobins-mn-loo-{target}",
                      config=dict(protocol="mobins-mn-loo", target=target, pool=pool,
                                  pred_day=pred_day, seed=seed, H=H, d=d, hidden=hidden,
                                  layers=layers, patch_len=patch_len, od_rank=rank,
                                  batch=batch, lr=lr, max_epochs=epochs, patience=patience,
                                  od_input=od_input, params=n_par))

    def forward(bt):
        x = bt["x"].to(device)
        n = x.shape[1]
        # The trunk's channel count is the pool's widest; a narrower city is
        # right-padded with zeros so one encoder serves every N in the pool.
        if x.shape[-1] < c_in:
            x = torch.cat([x, x.new_zeros(*x.shape[:-1], c_in - x.shape[-1])], dim=-1)
        z = trunk.encode(x, card=dataclasses.replace(trunk_card, num_nodes=n),
                         adj=bt["adj"].to(device))
        return head_n(z).view(-1, n, d, H), head_o(z)

    def evaluate(loader):
        for m in (trunk, head_n, head_o):
            m.eval()
        tot = [0.0, 0.0, 0.0]; nb = 0
        with torch.no_grad():
            for bt in loader:
                pn, po = forward(bt)
                m = mae_parts(pn, bt["y_node"].to(device), po, bt["y_od"].to(device))
                tot = [a + b for a, b in zip(tot, m)]; nb += 1
        return [t / max(nb, 1) for t in tot]

    best, bad, best_state, ep, t0 = math.inf, 0, None, -1, time.time()
    for ep in range(epochs):
        for m in (trunk, head_n, head_o):
            m.train()
        run_loss = nb = 0
        for bt in dl_tr:
            pn, po = forward(bt)
            yn, yo = bt["y_node"].to(device), bt["y_od"].to(device)
            loss = ((pn - yn).abs().sum() + (po - yo).abs().sum()) / (yn.numel() + yo.numel())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
            run_loss += loss.item(); nb += 1
        v = evaluate(dl_va)
        if wb:
            wb.log_metrics({"train_loss": run_loss / max(nb, 1), "val_mae_total": v[0],
                            "val_mae_node": v[1], "val_mae_od": v[2]}, step=ep)
        print(f"  epoch {ep:>3}  train {run_loss/max(nb,1):.4f}  val_total {v[0]:.4f} "
              f"(node {v[1]:.4f} od {v[2]:.4f})", flush=True)
        if v[0] < best - 1e-6:
            best, bad = v[0], 0
            best_state = {k: {kk: vv.detach().cpu().clone() for kk, vv in m.state_dict().items()}
                          for k, m in (("trunk", trunk), ("head_n", head_n), ("head_o", head_o))}
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {ep}", flush=True); break

    for k, m in (("trunk", trunk), ("head_n", head_n), ("head_o", head_o)):
        m.load_state_dict(best_state[k])
    zs = evaluate(dl_te)          # the held-out city, never seen in training
    row = {"target": target, "pool": pool, "pred_day": pred_day, "seed": seed,
           "protocol": "mobins-mn-loo", "H": H, "d": d,
           "MAE_total": zs[0], "MAE_node": zs[1], "MAE_od": zs[2],
           "val_mae_total": best, "epochs_run": ep + 1, "params": n_par,
           "max_epochs": epochs, "patience": patience, "batch": batch, "lr": lr,
           "od_input": od_input, "od_rank": rank,
           "seconds": round(time.time() - t0, 1),
           "splits": {"train": len(tr), "val": len(va), "test": len(te)}}
    ck = None
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        ck = os.path.join(os.path.dirname(out_path), f"loo-{target}-p{pred_day}-seed{seed}.pt")
        torch.save(best_state, ck)
    log_test({"MAE_total": zs[0], "MAE_node": zs[1], "MAE_od": zs[2]}, ck)
    wandb_finish()
    print(json.dumps(row, indent=1))
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=POOL)
    ap.add_argument("--pred-day", type=int, default=7, choices=[7, 14, 30])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", required=True,
                    help="pick a free one with ~/.claude/free-gpu; never hardcoded")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--patch-len", type=int, default=24)
    ap.add_argument("--od-rank", type=int, default=8)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--out", default="results/final/mobins_mn_loo.jsonl")
    a = ap.parse_args()
    run(a.target, a.seed, a.device, a.epochs, a.patience, a.batch, a.lr, a.hidden,
        a.layers, a.patch_len, a.od_rank, a.pred_day, a.out, weight_decay=a.weight_decay)
