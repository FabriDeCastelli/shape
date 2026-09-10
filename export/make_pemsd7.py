"""Build the PeMSD7 traffic-*speed* corpora from the STGCN release.

The project's speed corpora were Metr-LA and PeMS-Bay alone, which is one corpus
too few to hold a quantity out and still have that quantity in the pretraining
pool. PeMSD7(M) and PeMSD7(L) are the other two standard speed benchmarks: 228
and 1026 District-7 stations, 5-minute speed over the weekdays of May and June
2012, shipped with a station-to-station road distance matrix.

Two conventions are set deliberately.

*The graph is built as a top-k neighbourhood, not by STGCN's Gaussian threshold.*
Thresholding the shipped kernel gives average degree 31 (228 stations) and 217
(1026), an order of magnitude denser than the k-NN graphs of every other corpus
in the suite. A ``gram+phys`` ablation across a pool whose degree distributions
differ that much measures the graphs, not the relation.

*Splits are 6:2:2, and PeMSD7 is a pretraining corpus only.* The literature uses
a day-based split here, so a supervised number computed on this corpus would not
be comparable to a published one. The card records the convention so it cannot be
mistaken for one later.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from export.make_pems_raw import build
from shape.data import SHAPE_DATA_ROOT


def knn_graph(dist: np.ndarray, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k nearest neighbours by road distance, Gaussian weighted.

    sigma is the standard deviation of the kept distances, so the weights span a
    useful range whatever unit the source file is in.
    """
    d = dist.astype(np.float64).copy()
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :k]
    src = np.repeat(np.arange(len(d)), k)
    dst = nn.reshape(-1)
    kept = d[src, dst]
    w = np.exp(-((kept / kept.std()) ** 2))
    return (torch.tensor(np.stack([src, dst]), dtype=torch.long),
            torch.tensor(w, dtype=torch.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory holding PeMSD7_V_*.csv and PeMSD7_W_*.csv")
    ap.add_argument("--size", choices=["228", "1026"], required=True)
    ap.add_argument("--name", default=None, help="default PeMSD7M / PeMSD7L")
    ap.add_argument("--root", default=SHAPE_DATA_ROOT)
    ap.add_argument("--top-k", type=int, default=8)
    a = ap.parse_args()

    v = pd.read_csv(os.path.join(a.dir, f"PeMSD7_V_{a.size}.csv"), header=None)
    w = pd.read_csv(os.path.join(a.dir, f"PeMSD7_W_{a.size}.csv"), header=None)
    speed = v.to_numpy(dtype=np.float32)                      # [T, N]
    if speed.shape[1] != w.shape[0]:
        raise SystemExit(f"{speed.shape[1]} columns against a {w.shape} distance matrix")
    print(f"PeMSD7({a.size}): {speed.shape[0]} steps x {speed.shape[1]} stations, "
          f"speed {speed.min():.1f}-{speed.max():.1f}, missing {(speed == 0).mean():.4%}", flush=True)

    adj = knn_graph(w.to_numpy(), a.top_k)
    name = a.name or ("PeMSD7M" if a.size == "228" else "PeMSD7L")
    # 2012-05-01 is the first weekday of the STGCN release; only weekdays are
    # present, so the calendar covariates are not continuous and the card's
    # start_date is the series origin rather than a dense clock.
    build(name, speed, list(range(speed.shape[1])), "2012-05-01", adj, a.root,
          edge_semantics="road_distance")


if __name__ == "__main__":
    main()
