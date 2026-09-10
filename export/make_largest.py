"""Carve SHAPE corpora out of the LargeST California PeMS flow archive.

LargeST ships one HDF5 per year holding 5-minute flow for all 8,600 mainline
stations in California, written by pandas in "fixed" format. That layout is
plain HDF5 datasets, so h5py can slice the columns of one district without
reading the whole 7.2 GB file into memory.

The point of this file is ``--district``: PeMS03/04/07/08 are districts 3, 4, 7
and 8, so districts 5, 6, 10, 11 and 12 are corpora the project has never seen,
in the same physical quantity (vehicles per 5 min) and at the same sampling
rate. Windowing, targets, masking and splits are delegated to
``make_pems_raw.build`` so a LargeST corpus is conventionally identical to the
PeMS corpora already in the suite.
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

from export.make_pems_raw import build
from shape.data import SHAPE_DATA_ROOT


def read_frame(path: str, cols: Optional[List[int]] = None, max_steps: int = 0
               ) -> Tuple[np.ndarray, List[int], pd.DatetimeIndex]:
    """``[T, N]`` flow for the requested station ids, read lazily.

    pandas' fixed format stores ``axis0`` as the columns, ``axis1`` as the index
    and ``block0_values`` as ``[len(axis1), len(axis0)]``. Column order in the
    file is not the order of ``cols``, and h5py requires a sorted index list, so
    the selection is sorted for the read and permuted back afterwards.
    """
    with h5py.File(path, "r") as fh:
        grp = fh[list(fh.keys())[0]]
        ids = [int(x) for x in grp["axis0"][:]]
        stamps = pd.to_datetime(grp["axis1"][:])
        vals = grp["block0_values"]
        if vals.shape[0] != len(stamps):
            raise SystemExit(f"unexpected layout: {vals.shape} vs {len(stamps)} stamps")
        n = min(max_steps, len(stamps)) if max_steps else len(stamps)
        stamps = stamps[:n]
        if cols is None:
            return vals[:n], ids, stamps
        pos = {sid: i for i, sid in enumerate(ids)}
        missing = [c for c in cols if c not in pos]
        if missing:
            raise SystemExit(f"{len(missing)} requested stations absent, e.g. {missing[:5]}")
        want = sorted(pos[c] for c in cols)
        # Slice time in HDF5 first, then take columns in numpy: h5py fancy
        # indexing along axis 1 walks the whole dataset and is far slower.
        flow = np.asarray(vals[:n, :])[:, want]
        got = [ids[i] for i in want]
    return flow, got, stamps


def haversine_knn(meta: pd.DataFrame, ids: List[int], k: int, sigma_km: float):
    """Gaussian-weighted k-NN graph over great-circle distance between stations."""
    m = meta.set_index("ID").reindex(ids)
    lat, lon = np.radians(m["Lat"].to_numpy()), np.radians(m["Lng"].to_numpy())
    dlat, dlon = lat[:, None] - lat[None, :], lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    dist = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    np.fill_diagonal(dist, np.inf)
    nn = np.argsort(dist, axis=1)[:, :k]
    src = np.repeat(np.arange(len(ids)), k)
    dst = nn.reshape(-1)
    w = np.exp(-(dist[src, dst] / sigma_km) ** 2)
    return (torch.tensor(np.stack([src, dst]), dtype=torch.long),
            torch.tensor(w, dtype=torch.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="one LargeST year file, e.g. 2019.h5")
    ap.add_argument("--meta", required=True, help="LargeST metadata.csv")
    ap.add_argument("--district", type=int, required=True)
    ap.add_argument("--name", default=None, help="corpus name (default PemsD<district>)")
    ap.add_argument("--root", default=SHAPE_DATA_ROOT)
    ap.add_argument("--months", type=int, default=0,
                    help="keep only the first N months, 0 = the whole year")
    ap.add_argument("--min-observed", type=float, default=0.9)
    ap.add_argument("--graph", action="store_true", help="add a haversine k-NN graph")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--sigma-km", type=float, default=2.0)
    a = ap.parse_args()

    meta = pd.read_csv(a.meta)
    want = meta[meta["District"] == a.district]["ID"].astype(int).tolist()
    if not want:
        raise SystemExit(f"district {a.district} has no stations in {a.meta}")
    print(f"district {a.district}: {len(want)} stations in metadata", flush=True)

    # 5-minute sampling: a month is about 288 * 31 steps; read a little extra and
    # trim exactly by timestamp below.
    max_steps = a.months * 288 * 31 + 288 if a.months else 0
    flow, ids, stamps = read_frame(a.h5, want, max_steps)
    if a.months:
        keep = stamps < stamps[0] + pd.DateOffset(months=a.months)
        flow, stamps = flow[keep], stamps[keep]
    flow = np.nan_to_num(flow, nan=0.0)            # PeMS gaps are zeros, and the mask keys off zero

    observed = (flow != 0).mean(axis=0)
    keep = observed >= a.min_observed
    if not keep.any():
        raise SystemExit(f"no station reaches min_observed={a.min_observed}")
    flow, ids = flow[:, keep], [i for i, k in zip(ids, keep) if k]
    print(f"  kept {len(ids)}/{len(want)} stations at >= {a.min_observed:.0%} observed, "
          f"{flow.shape[0]} steps from {stamps[0].date()} to {stamps[-1].date()}", flush=True)

    adj = haversine_knn(meta, ids, a.top_k, a.sigma_km) if a.graph else None
    build(a.name or f"PemsD{a.district}", flow, ids, str(stamps[0].date()), adj, a.root)


if __name__ == "__main__":
    main()
