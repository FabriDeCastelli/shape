"""MOBINS at its own protocol: the mobility networked time-series task.

\\cite{mobins} Definition 3.1 forecasts the node features *and* the OD movements,
and its Table 4 reports the ``[Total]`` MAE its code returns, averaged over
``d*N + N^2`` dimensions. This exporter reproduces that setting exactly, from
``data_loader.py`` and ``main.py`` of the released implementation:

* windows are cut on a ``[days, 24, dim]`` tensor with ``seq_day=4`` and
  ``pred_day in {7, 14, 30}``, so the look-back is 96 hourly steps and the
  horizon 168/336/720, and consecutive windows are **one day apart**;
* the split is ``test_ratio=0.25`` of the days, then ``train_ratio=0.8`` of the
  remainder, i.e. 60:15:25, and sequences are cut *within* train and *within*
  test so no window straddles the boundary;
* every column is standard-scaled by a scaler fitted on the pre-test days only,
  and the MAE is reported on those scaled units with no inverse transform.

The node features and the OD matrix are stored separately rather than as one
flat ``d*N + N^2`` series: the model reads N nodes with d channels and decodes
the OD block from the node states, so a flat layout would put 16,640 "nodes"
into a Gram relation that is quadratic in them. The targets are cut from the
stored series in ``__getitem__`` -- materialising ``[S, 16640, H]`` for Seoul
would be 193 GB.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

RAW = os.environ.get("MOBINS_RAW", "/raid/f.decastelli/mobins_raw")
OUT = os.environ.get("SHAPE_DATA_ROOT", "/raid/f.decastelli/tgfm_data")

# name -> (folder, hours per day, seq_day, node feature count)
SETS = {
    "MobinsSeoulMN": ("Transportation-Seoul", 24, 4),
    "MobinsBusanMN": ("Transportation-Busan", 24, 4),
    "MobinsDaeguMN": ("Transportation-Daegu", 24, 4),
    "MobinsNYCMN":   ("Transportation-NYC",   24, 4),
    "MobinsEpiKoreaMN": ("Epidemic-Korea", 1, 4),
    "MobinsEpiNYCMN":   ("Epidemic-NYC",   1, 4),
}
TEST_RATIO, TRAIN_RATIO = 0.25, 0.8      # main.py defaults


def load_nodes(folder: str):
    """``[L, N, d]`` node features, column order N0,N1,... per feature."""
    df = pd.read_csv(os.path.join(RAW, folder, "NODE_TIME_SERIES_FEATURES.csv"))
    cols = list(df.columns[1:])                      # column 0 is the timestamp
    names = sorted({c.rsplit("_", 1)[0] for c in cols}, key=lambda s: int(s[1:]))
    feats = sorted({c.rsplit("_", 1)[1] for c in cols})
    arr = np.stack([[df[f"{n}_{f}"].to_numpy(np.float32) for f in feats]
                    for n in names], axis=1)         # [d, N, L]
    return np.transpose(arr, (2, 1, 0)), len(names), len(feats)


def load_od_dense(folder: str, n: int, L: int) -> torch.Tensor:
    """``[L, N, N]`` hourly OD, float32. Self-loops are kept: they are part of
    the target the benchmark scores."""
    f = os.path.join(RAW, folder, "OD_MOVEMENTS.csv")
    out = torch.empty(L, n, n, dtype=torch.float32)
    at = 0
    for chunk in pd.read_csv(f, chunksize=512):
        V = chunk.drop(columns=["datetime"]).to_numpy(np.float32).reshape(-1, n, n)
        k = min(len(V), L - at)
        if k <= 0:
            break
        out[at:at + k] = torch.from_numpy(V[:k])
        at += k
    if at < L:
        raise SystemExit(f"{folder}: OD has {at} snapshots, need {L}")
    return out


def build(name: str, root: str) -> dict:
    """One corpus per city. The horizon is a training argument, not a build one:
    the OD tensor is 1.2 GB and identical across the three prediction lengths, so
    storing it once and cutting the splits at train time saves reading the source
    CSV three times and 2/3 of the disk."""
    folder, cycle, seq_day = SETS[name]
    feats, N, d = load_nodes(folder)
    L = len(feats)
    days = L // cycle
    W = seq_day * cycle
    od = load_od_dense(folder, N, L)
    test_start = days - int(days * TEST_RATIO)

    # --- scaling: per column, fitted on the pre-test days only ---------------
    cut = test_start * cycle

    def scale(t: torch.Tensor) -> torch.Tensor:
        """sklearn ``StandardScaler`` semantics, per column, fitted on train days.

        The zero-variance rule matters and is not a detail: an OD cell that never
        moves in the training days has ``std == 0``, and clamping that to a small
        epsilon turns any non-zero test value into ~1e6. sklearn's
        ``_handle_zeros_in_scale`` sets those columns to scale 1.0 instead, which
        leaves them as plain centred counts. ddof=0 matches it too.
        """
        flat = t.reshape(len(t), -1)
        mu = flat[:cut].mean(0)
        sd = flat[:cut].std(0, unbiased=False)
        sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
        return ((flat - mu) / sd).reshape(t.shape)

    # The propagation graph is built from the RAW OD, before scaling. Standardised
    # flows are centred, so a row sums to about zero and the symmetric
    # normalisation divides by a degree that is numerically noise: the graph then
    # multiplies the states by ~1e3 and training diverges immediately. Raw OD is
    # non-negative and its degrees are real traffic volumes.
    a = od + od.transpose(1, 2)
    deg = a.sum(-1).clamp(min=1.0).pow(-0.5)                       # [L, N]
    adj = (deg.unsqueeze(-1) * a * deg.unsqueeze(1)).half()        # [L, N, N]

    nf = scale(torch.from_numpy(feats))                            # [L, N, d]
    od = scale(od)                                                 # [L, N, N]

    card = {
        "C": d, "H": 0, "H_max": 0, "W": W,
        "channel_groups": {"real": [0, d]},
        "edge_weight_semantics": "od-flow", "freq": f"{cycle}h" if cycle > 1 else "1d",
        "global_mean": None, "global_std": None, "has_edge_weight": True,
        "metric": "mae_total", "name": name, "node_cap": None, "node_selection": "all",
        "num_classes": 0, "num_nodes": N,
        "num_samples": L, "num_snapshots": L,
        "probe_dim": 0, "probe_seed": 0, "snapshot_cap": None,
        "split_kind": "temporal", "static_feature_dim": 0,
        "static_mean": None, "static_std": None, "static_topology": False,
        "task": "forecast_mn",
        "split_sizes": {"train": 0, "val": 0, "test": 0},
        "start_date": None,
        "u_available": [0, 0, 0, 0, 0, 0, 1, 1],
    }
    data = {
        "signal": nf,                 # [L, N, d]  full series; targets are cut from it
        "od": od,                     # [L, N, N]  scaled, this is the target
        "adj": adj,                   # [L, N, N]  normalised from raw OD, fp16
        "static": None, "mask": None, "y": None,
        "edge_index": None, "edge_weight": None,
        "splits": None,        # cut at train time, see scripts/mobins_mn.py
        "days": days, "cycle": cycle, "seq_day": seq_day, "test_start_day": test_start,
    }
    out = os.path.join(root, name)
    os.makedirs(out, exist_ok=True)
    torch.save(data, os.path.join(out, "data.pt"))
    json.dump(card, open(os.path.join(out, "card.json"), "w"), indent=2, sort_keys=True)
    json.dump({"seq_day": seq_day, "cycle": cycle, "days": days,
               "test_start_day": test_start,
               "test_ratio": TEST_RATIO, "train_ratio": TRAIN_RATIO,
               "target": "cat(node_features, flattened_OD)",
               "target_dim": d * N + N * N,
               "source": "kaist-dmlab/MOBINS data_loader.py + main.py"},
              open(os.path.join(out, "protocol.json"), "w"), indent=2)
    return card


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(SETS))
    ap.add_argument("--root", default=OUT)
    a = ap.parse_args()
    c = build(a.name, a.root)
    print(f"  {a.name}  N={c['num_nodes']} d={c['C']} W={c['W']} "
          f"target_dim={c['C']*c['num_nodes'] + c['num_nodes']**2}")
