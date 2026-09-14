"""MOBINS mobility and epidemic corpora into the TGFM contract.

Each dataset ships three csv files: a node time series (INFLOW and OUTFLOW per
node), a static spatial adjacency, and hourly origin-destination matrices.

Both node features are kept as input channels; the forecasting target is INFLOW,
which is the node-level target the reference benchmark reports.

The topology is the OD tensor, not the spatial adjacency. The spatial network is
binary, static and carries only 20.3% of the OD mass (11.6% excluding
self-loops), so a model propagating over it sees a constant graph on a task that
is entirely about time-varying flow. The OD matrix is a genuine ``A_t``: weighted,
directed, and different every hour. Entry ``i`` of the edge list is the OD matrix
at the window *end*, so ``S_topo = A_t`` holds as the method states.

``--spatial`` restores the old behaviour (static binary adjacency, INFLOW only)
for a like-for-like comparison against the corpora built before this change.
"""
from __future__ import annotations

import argparse
import os

import json

import numpy as np
import pandas as pd
import torch

from shape.data import H_MAX
from shape.data import SHAPE_DATA_ROOT

RAW = "/raid/f.decastelli/mobins_raw"

# name -> (folder, freq, context W, horizon H)
#
# The reference benchmark uses a four-day look-back and a seven-day horizon, so
# W and H are four and seven *days* expressed in each dataset's own resolution:
# 96 and 168 hourly steps for transportation, 4 and 7 daily steps for epidemic.
SETS = {
    # <name>OD carries the hourly OD tensor as its topology and both node
    # features; the plain name is the pre-fix corpus (static binary graph,
    # inflow only) and is kept so the two can be compared directly.
    "MobinsSeoulOD": ("Transportation-Seoul", "1h", 96, 168),
    "MobinsBusanOD": ("Transportation-Busan", "1h", 96, 168),
    "MobinsDaeguOD": ("Transportation-Daegu", "1h", 96, 168),
    "MobinsNYCOD":   ("Transportation-NYC",   "1h", 96, 168),
    "MobinsSeoul": ("Transportation-Seoul", "1h",   96, 168),
    "MobinsBusan": ("Transportation-Busan", "1h",   96, 168),
    "MobinsDaegu": ("Transportation-Daegu", "1h",   96, 168),
    "MobinsNYC":   ("Transportation-NYC",   "1h",   96, 168),
    "EpiKorea":    ("Epidemic-Korea",       "1d",    4,   7),
    "EpiNYC":      ("Epidemic-NYC",         "1d",    4,   7),
}

# 75% train+val / 25% test, and the training part split 80/20 -> 60/15/25.
VAL_FRAC, TEST_FRAC = 0.15, 0.25


def load(folder: str):
    """``[L, N, 2]`` node features (inflow, outflow), the spatial graph, and names."""
    d = os.path.join(RAW, folder)
    ts = pd.read_csv(os.path.join(d, "NODE_TIME_SERIES_FEATURES.csv"))
    start = str(ts.iloc[0, 0])
    cols = list(ts.columns[1:])
    # Columns are "<node>_<FEATURE>", two features per node in a fixed order.
    nodes, per_node = [], {}
    for c in cols:
        n, feat = c.rsplit("_", 1)
        if n not in nodes:
            nodes.append(n); per_node[n] = []
        per_node[n].append(c)
    # The Korean cities ship INFLOW and OUTFLOW per station; NYC ships a single
    # RIDERSHIP series. Either is fine -- the target is always the first feature
    # -- but the count must be the same for every node in a city.
    widths = {len(v) for v in per_node.values()}
    if len(widths) != 1 or widths.pop() not in (1, 2):
        raise SystemExit(f"{folder}: features per node must be a uniform 1 or 2, "
                         f"got {sorted({len(v) for v in per_node.values()})}")
    feats = np.stack([ts[per_node[n]].to_numpy(dtype=np.float32) for n in nodes],
                     axis=1)                                   # [L, N, 2]

    adj = pd.read_csv(os.path.join(d, "SPATIAL_NETWORK.csv"), index_col=0)
    a = adj.to_numpy(dtype=np.float32)
    src, dst = np.nonzero(a)
    ei = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    ew = torch.tensor(a[src, dst], dtype=torch.float32)
    return feats, (ei, ew), start, nodes


def load_od(folder: str, n: int):
    """Hourly OD matrices as per-snapshot edge lists.

    Indices are int32: every consumer casts with ``.long()`` before use, and the
    full tensor at int64 would be 2.8 GB per city against 1.7 GB here. Self-loops
    are dropped, since a node's influence on itself is what the residual in
    eq:relation-gate already carries.
    """
    f = os.path.join(RAW, folder, "OD_MOVEMENTS.csv")
    eis, ews = [], []
    for chunk in pd.read_csv(f, chunksize=512):
        V = chunk.drop(columns=["datetime"]).to_numpy(dtype=np.float32)
        V = V.reshape(len(V), n, n)
        for t in range(len(V)):
            m = V[t].copy()
            np.fill_diagonal(m, 0.0)
            src, dst = np.nonzero(m)
            eis.append(torch.tensor(np.stack([src, dst]), dtype=torch.int32))
            ews.append(torch.tensor(m[src, dst], dtype=torch.float32))
    return eis, ews


def build_mobins(name: str, feats: np.ndarray, start: str, W: int, H: int,
                 hold, root: str, freq: str, od=None, adj=None):
    """Write the corpus. ``feats`` is ``[L, N, 2]``; the target is channel 0."""
    L, N, C = feats.shape
    S = L - W - H + 1
    if S <= 0:
        raise SystemExit(f"{name}: {L} steps too short for W={W}, H={H}")
    raw = torch.from_numpy(feats.astype(np.float32))            # [L, N, C]
    signal = raw[: L - H]                                       # [L-H, N, C]
    idx = torch.arange(S)[:, None] + W + torch.arange(H)[None, :]
    y = raw[idx][..., 0].permute(0, 2, 1).contiguous()          # [S, N, H] inflow
    mask = y != 0

    n_val, n_test = hold
    n_train = S - n_val - n_test
    splits = {"train": torch.arange(0, n_train),
              "val": torch.arange(n_train, n_train + n_val),
              "test": torch.arange(n_train + n_val, S)}
    seen = raw[: n_train + W - 1, :, 0]                         # target channel only
    gm, gs = float(seen.mean()), float(seen.std().clamp(min=1e-6))

    if od is not None:
        od_ei, od_ew = od
        # num_snapshots == num_samples for a forecasting corpus, so the dataset
        # reads element i for sample i; sample i's window ends at absolute hour
        # i+W-1, and that is the snapshot S_topo must describe.
        edge_index = [od_ei[i + W - 1] for i in range(S)]
        edge_weight = [od_ew[i + W - 1] for i in range(S)]
        semantics, static_topo = "od-flow", False
    elif adj is not None:
        ei, ew = adj
        edge_index, edge_weight = [ei] * S, [ew] * S
        semantics, static_topo = "given", True
    else:
        edge_index = edge_weight = None
        semantics, static_topo = "none", False

    card = {
        "C": C, "H": H, "H_max": H_MAX, "W": W,
        "channel_groups": {"real": [0, C]},
        "edge_weight_semantics": semantics,
        "freq": freq, "global_mean": gm, "global_std": gs,
        "has_edge_weight": edge_index is not None,
        "metric": "masked_mae", "name": name, "node_cap": None,
        "node_selection": "all", "num_classes": 0, "num_nodes": N,
        "num_samples": S, "num_snapshots": S, "probe_dim": 16, "probe_seed": 0,
        "snapshot_cap": None, "split_kind": "temporal",
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "start_date": start, "static_feature_dim": 0,
        "static_mean": None, "static_std": None,
        "static_topology": static_topo, "task": "forecast",
        "u_available": [1] * 8,
    }
    out = os.path.join(root, name)
    os.makedirs(out, exist_ok=True)
    torch.save({"signal": signal, "static": None, "edge_index": edge_index,
                "edge_weight": edge_weight, "y": y, "mask": mask,
                "splits": splits}, os.path.join(out, "data.pt"))
    json.dump(card, open(os.path.join(out, "card.json"), "w"), indent=1)
    e = "static" if static_topo else (f"OD ~{int(np.mean([len(w) for w in edge_weight[:50]]))} edges/h"
                                      if edge_index else "none")
    print(f"{name}: {L} steps x {N} nodes x {C} feat -> {S} samples, "
          f"target mean {gm:.3f} std {gs:.3f}, topology={e} -> {out}", flush=True)
    return card


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(SETS))
    ap.add_argument("--root", default=SHAPE_DATA_ROOT)
    ap.add_argument("--window", type=int, default=0,
                    help="override the benchmark look-back; the reference uses four\n                          days, and a longer context is reported as its own column")
    ap.add_argument("--standardize", action="store_true",
                    help="store the standard-scaled series, so MAE is in the\n                          reference benchmark's units")
    ap.add_argument("--spatial", action="store_true",
                    help="use the static binary spatial graph and INFLOW only, "
                         "reproducing the corpora built before the OD fix")
    a = ap.parse_args()
    folder, freq, W, H = SETS[a.name]
    W = a.window or W
    feats, adj, start, nodes = load(folder)
    L, N, C = feats.shape
    print(f"{a.name}: {folder}  raw {feats.shape}  freq {freq}  start {start}  "
          f"spatial edges {adj[0].shape[1]}  zeros {100*float((feats==0).mean()):.1f}%")
    if a.spatial:
        feats = feats[..., :1]
        C = 1
    # The reference reports MAE on standard-scaled data, so the corpus stores the
    # scaled series and a raw-unit MAE from this corpus IS their metric. Statistics
    # come from the training period alone, per node and per feature.
    if a.standardize:
        cut = int(L * (1.0 - VAL_FRAC - TEST_FRAC))
        mu = feats[:cut].mean(0, keepdims=True)
        sd = feats[:cut].std(0, keepdims=True)
        sd[sd < 1e-6] = 1.0
        feats = (feats - mu) / sd
        print(f"  standardised per node on the first {cut} steps: "
              f"mu[0,0]={float(mu[0,0,0]):.2f} sd[0,0]={float(sd[0,0,0]):.2f}")

    S = L - W - H + 1
    hold = (int(S * VAL_FRAC), int(S * TEST_FRAC))
    print(f"  samples {S}, folds train/val/test = {S-sum(hold)}/{hold[0]}/{hold[1]}")
    od = None
    if not a.spatial:
        print("  reading OD movements ...", flush=True)
        od = load_od(folder, N)
        print(f"  OD snapshots {len(od[0])} (need {S + W - 1})", flush=True)
    build_mobins(a.name, feats, start, W, H, hold, a.root, freq,
                 od=od, adj=adj if a.spatial else None)


if __name__ == "__main__":
    main()
