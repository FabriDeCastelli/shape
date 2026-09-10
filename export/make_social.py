"""Weekly-snapshot D-TDGs from raw temporal edge streams, MiNT's task.

The label is MiNT's network-growth property, Eq. (1) of their benchmark: for the
observation window ending at snapshot ``t``, the class is 1 iff the edge multiset
of the future window is larger than that of the observation window,

    P = 1  iff  |E(t+d1 .. t+d2)| > |E(t-W+1 .. t)| .

Edges are a multi-set: repeats within a snapshot are kept, matching MiNT's
``|E|`` and matching the existing MiNT corpora (snapshot 100 of ARC holds 411
edges over 229 distinct pairs).

Only snapshots whose future window is fully observed carry a label, so the corpus
is truncated by ``d2`` rather than padded with labels that cannot be computed.
"""
from __future__ import annotations

import argparse, csv, gzip, json, os, sys
from typing import Iterator, Tuple

import torch

RAW = "/raid/f.decastelli/data/social_raw"
OUT = os.environ.get("SHAPE_DATA_ROOT", "/raid/f.decastelli/tgfm_data")
WEEK = 7 * 86400

# name -> (file, format).  The six training networks and the two MiNT holds out.
SOCIAL = {
    "MiNTSocCollegeMsg":    ("CollegeMsg.txt.gz",             "snap"),
    "MiNTSocEmailEU":       ("email-Eu-core-temporal.txt.gz", "snap"),
    "MiNTSocMathOverflow":  ("sx-mathoverflow.txt.gz",        "snap"),
    "MiNTSocAskUbuntu":     ("sx-askubuntu.txt.gz",           "snap"),
    "MiNTSocSuperUser":     ("sx-superuser.txt.gz",           "snap"),
    "MiNTSocStackOverflow": ("sx-stackoverflow.txt.gz",       "snap"),
    "MiNTSocLastFM":        ("jodie-lastfm.csv",              "jodie"),
    "MiNTSocRedditB":       ("soc-redditHyperlinks-body.tsv", "reddit"),
}


def edges(path: str, fmt: str) -> Iterator[Tuple[str, str, int]]:
    """(src, dst, unix_seconds) for one raw file."""
    if fmt == "snap":
        with gzip.open(path, "rt") as fh:
            for line in fh:
                a, b, t = line.split()
                yield a, b, int(t)
    elif fmt == "jodie":
        with open(path) as fh:
            next(fh)
            for line in fh:
                p = line.split(",", 4)
                # user and item are separate id spaces in a bipartite stream
                yield "u" + p[0], "i" + p[1], int(float(p[2]))
    elif fmt == "reddit":
        import datetime as dt
        with open(path) as fh:
            r = csv.DictReader(fh, delimiter="\t")
            for row in r:
                ts = dt.datetime.strptime(row["TIMESTAMP"], "%Y-%m-%d %H:%M:%S")
                yield row["SOURCE_SUBREDDIT"], row["TARGET_SUBREDDIT"], int(ts.timestamp())
    else:
        raise ValueError(fmt)


def build(name: str, W: int, d1: int, d2: int, probe_dim: int) -> dict:
    path = os.path.join(RAW, SOCIAL[name][0])
    ids: dict = {}
    bins: dict = {}
    lo = None
    for a, b, t in edges(path, SOCIAL[name][1]):
        lo = t if lo is None or t < lo else lo
    for a, b, t in edges(path, SOCIAL[name][1]):
        s = ids.setdefault(a, len(ids)); d = ids.setdefault(b, len(ids))
        bins.setdefault((t - lo) // WEEK, []).append((s, d))
    T = max(bins) + 1
    ei = [torch.tensor(bins.get(k, [(0, 0)] * 0) or [], dtype=torch.long).reshape(-1, 2).T.contiguous()
          for k in range(T)]
    m = torch.tensor([e.shape[1] for e in ei], dtype=torch.float64)
    cum = torch.cat([torch.zeros(1, dtype=torch.float64), m.cumsum(0)])

    # a label exists only where both windows are fully observed
    t_last = T - 1 - d2
    y = torch.zeros(T, 1, dtype=torch.long)
    for t in range(W - 1, t_last + 1):
        obs = cum[t + 1] - cum[t + 1 - W]
        fut = cum[t + 1 + d2] - cum[t + d1]
        y[t, 0] = int(fut > obs)

    n_snap = t_last + 1                       # snapshots that carry a usable label
    n_samp = n_snap - W + 1
    assert n_samp > 0, f"{name}: only {T} weeks, need > {W + d2}"
    n_tr = int(round(0.70 * n_samp)); n_va = int(round(0.15 * n_samp))
    n_te = n_samp - n_tr - n_va
    card = {
        "C": probe_dim, "H": 1, "H_max": 12, "W": W,
        "channel_groups": {"probe": [0, probe_dim]},
        "edge_weight_semantics": "count", "freq": "7D",
        "global_mean": None, "global_std": None, "has_edge_weight": True,
        "metric": "rocauc", "name": name, "node_cap": None, "node_selection": "all",
        "num_classes": 2, "num_nodes": len(ids), "num_samples": n_samp,
        "num_snapshots": n_snap, "probe_dim": probe_dim, "probe_seed": 0,
        "snapshot_cap": None, "split_kind": "temporal",
        "split_sizes": {"train": n_tr, "val": n_va, "test": n_te},
        "start_date": None, "static_feature_dim": 0, "static_mean": None,
        "static_std": None, "static_topology": False, "task": "graph-classify",
        "u_available": [0, 0, 0, 0, 0, 0, 1, 1],
    }
    data = {"signal": None, "static": None,
            "edge_index": ei[:n_snap],
            "edge_weight": [torch.ones(e.shape[1], 1) for e in ei[:n_snap]],
            "y": y[:n_snap], "mask": None,
            "splits": {"train": torch.arange(n_tr),
                       "val": torch.arange(n_tr, n_tr + n_va),
                       "test": torch.arange(n_tr + n_va, n_samp)}}
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    torch.save(data, os.path.join(d, "data.pt"))
    json.dump(card, open(os.path.join(d, "card.json"), "w"), indent=2, sort_keys=True)
    # DatasetCard has a fixed field set, so the label rule lives beside it.
    json.dump({"W": W, "d1": d1, "d2": d2, "eq": "|E(t+d1..t+d2)| > |E(t-W+1..t)|"},
              open(os.path.join(d, "label_rule.json"), "w"), indent=2)
    return card


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(SOCIAL))
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--d1", type=int, default=1, help="first snapshot of the future window")
    ap.add_argument("--d2", type=int, default=12, help="last snapshot of the future window")
    ap.add_argument("--probe-dim", type=int, default=16)
    a = ap.parse_args()
    for n in a.datasets:
        c = build(n, a.window, a.d1, a.d2, a.probe_dim)
        print(f"  {n:18s} N={c['num_nodes']:8d} snaps={c['num_snapshots']:5d} "
              f"samples={c['num_samples']:5d} split={tuple(c['split_sizes'].values())} "
              f"pos_rate={float(torch.load(os.path.join(OUT,n,'data.pt'),weights_only=False)['y'].float().mean()):.3f}",
              flush=True)
