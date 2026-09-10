"""Build a SHAPE corpus from raw Caltrans PeMS "Station 5-Minute" exports.

``export/build.py`` converts TIDES' own exports; this converts the raw
Clearinghouse files, so a district TIDES never shipped (D11 San Diego, D12
Orange County) can become a corpus with the same conventions.

The layout conventions are read off the existing PeMS corpora rather than
assumed, and ``_self_check`` asserts them:

* the raw series has ``L`` steps; ``signal`` stores only ``raw[:L-H]``, because
  the final ``H`` steps are targets and never inputs;
* ``num_samples = L - W - H + 1``, and window ``i`` is ``raw[i:i+W]``;
* ``y[i, :, h] = raw[i + W + h]``;
* a masked-MAE corpus marks a missing reading by a literal ``0``, so the mask is
  ``y != 0`` -- PeMS writes gaps as zero flow rather than as NaN;
* global mean/std come from ``raw[: n_train + W - 1]`` only, which is exactly
  what the last training window has seen.

Clearinghouse station files are headerless gzipped CSV. The first twelve columns
are fixed; per-lane groups follow and are ignored.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from shape.data import H_MAX, SHAPE_DATA_ROOT, W_DEFAULT

# Station 5-Minute column layout, per the PeMS user guide.
TIMESTAMP, STATION, TOTAL_FLOW = 0, 1, 9
USECOLS = [TIMESTAMP, STATION, TOTAL_FLOW]


def read_station_5min(paths: List[str]) -> pd.DataFrame:
    """Concatenate Clearinghouse files into a [time x station] flow frame."""
    frames = []
    for p in sorted(paths):
        df = pd.read_csv(p, header=None, usecols=USECOLS,
                         names=["ts", "station", "flow"], compression="infer")
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["ts"] = pd.to_datetime(raw["ts"], format="%m/%d/%Y %H:%M:%S")
    wide = raw.pivot_table(index="ts", columns="station", values="flow",
                           aggfunc="first")
    # A station absent for a whole file reads as NaN; PeMS' own convention for a
    # missing sample is a zero flow, and the mask keys off zero, so match it.
    return wide.sort_index().fillna(0.0)


def drop_sparse(wide: pd.DataFrame, min_observed: float) -> pd.DataFrame:
    """Discard stations that are mostly zero, which are dead loops not roads."""
    observed = (wide != 0).mean(axis=0)
    keep = observed[observed >= min_observed].index
    if len(keep) == 0:
        raise SystemExit(f"no station reaches min_observed={min_observed}")
    return wide[keep]


def haversine_adj(meta: pd.DataFrame, ids: List[int], k: int,
                  sigma_km: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """k-nearest-neighbour graph over great-circle distance, Gaussian weighted.

    PeMS' published adjacencies use road-network distance, which the metadata
    files do not carry. Great-circle distance is a proxy and is recorded as such
    in the card (``edge_weight_semantics = "haversine"``), so no downstream
    result can silently claim the road graph.
    """
    m = meta.set_index("ID").reindex(ids)
    lat, lon = np.radians(m["Latitude"].to_numpy()), np.radians(m["Longitude"].to_numpy())
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    dist = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))      # km
    np.fill_diagonal(dist, np.inf)
    nn = np.argsort(dist, axis=1)[:, :k]
    src = np.repeat(np.arange(len(ids)), k)
    dst = nn.reshape(-1)
    w = np.exp(-(dist[src, dst] / sigma_km) ** 2)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return edge_index, torch.tensor(w, dtype=torch.float32)


def build(name: str, flow: np.ndarray, ids: List[int], start: str,
          adj: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
          root: str = SHAPE_DATA_ROOT, W: int = W_DEFAULT, H: int = 12,
          val_frac: float = 0.2, hold: Optional[Tuple[int, int]] = None,
          edge_semantics: str = "haversine",
          freq: str = "5min") -> Dict:
    """Write ``<root>/<name>/{data.pt,card.json}`` from a raw [L, N] flow array."""
    L, N = flow.shape
    S = L - W - H + 1
    if S <= 0:
        raise SystemExit(f"{name}: {L} steps is too short for W={W}, H={H}")
    raw = torch.from_numpy(flow.astype(np.float32))               # [L, N]
    signal = raw[: L - H].unsqueeze(-1)                           # [L-H, N, 1]
    idx = torch.arange(S)[:, None] + W + torch.arange(H)[None, :]  # [S, H]
    y = raw[idx].permute(0, 2, 1).contiguous()                    # [S, N, H]
    mask = y != 0

    # val and test each take floor(val_frac * S) and train takes the remainder.
    # This is the rule the shipped PeMS corpora use: on PeMS08 it reproduces
    # 10701/3566/3566 exactly, where int(0.6 * S) would give 10699/3566/3568.
    #
    # ``hold`` overrides that count. Widening W shrinks S, which slides the fold
    # boundaries and makes the model predict different timestamps than the
    # published protocol does. Passing the *reference* protocol's val/test size
    # pins both folds to the same target indices and takes the shortfall off the
    # front of training instead, which is the only fold that may lose windows.
    # 6:2:2 corpora have equal folds; Metr-LA and PeMS-Bay are 7:1:2 and do not,
    # so the two sizes are pinned independently rather than assumed equal.
    n_val, n_test = (int(S * val_frac),) * 2 if hold is None else hold
    if n_val + n_test >= S:
        raise SystemExit(f"{name}: hold={hold} leaves no training windows (S={S})")
    n_train = S - n_val - n_test
    splits = {"train": torch.arange(0, n_train),
              "val": torch.arange(n_train, n_train + n_val),
              "test": torch.arange(n_train + n_val, S)}
    seen = raw[: n_train + W - 1]
    gm, gs = float(seen.mean()), float(seen.std().clamp(min=1e-6))

    edge_index = edge_weight = None
    if adj is not None:
        ei, ew = adj
        edge_index = [ei] * S          # static topology: one object, reused
        edge_weight = [ew] * S

    card = {
        "C": 1, "H": H, "H_max": H_MAX, "W": W,
        "channel_groups": {"real": [0, 1]},
        "edge_weight_semantics": edge_semantics if adj is not None else "none",
        "freq": freq, "global_mean": gm, "global_std": gs,
        "has_edge_weight": adj is not None,
        "metric": "masked_mae", "name": name, "node_cap": None,
        "node_selection": "all", "num_classes": 0, "num_nodes": N,
        "num_samples": S, "num_snapshots": S, "probe_dim": 16, "probe_seed": 0,
        "snapshot_cap": None, "split_kind": "temporal",
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "start_date": start, "static_feature_dim": 0,
        "static_mean": None, "static_std": None,
        "static_topology": adj is not None, "task": "forecast",
        "u_available": [1] * 8,
    }
    out = os.path.join(root, name)
    os.makedirs(out, exist_ok=True)
    torch.save({"signal": signal, "static": None, "edge_index": edge_index,
                "edge_weight": edge_weight, "y": y, "mask": mask,
                "splits": splits}, os.path.join(out, "data.pt"))
    json.dump(card, open(os.path.join(out, "card.json"), "w"), indent=1)
    # Provenance lives beside the card, not in it: DatasetCard is a fixed schema
    # shared by every corpus and an extra key there breaks every loader.
    json.dump({"source": "caltrans-pems-station-5min", "station_ids": ids},
              open(os.path.join(out, "provenance.json"), "w"))
    print(f"{name}: {L} steps x {N} stations -> {S} samples, "
          f"mean {gm:.1f} std {gs:.1f}, graph={'yes' if adj else 'no'} -> {out}",
          flush=True)
    return card


def _self_check() -> None:
    """The conventions this file claims, asserted against a synthetic series."""
    L, N, W, H = 200, 5, W_DEFAULT, 12
    flow = np.arange(L * N, dtype=np.float32).reshape(L, N) + 1.0
    root = os.path.join(os.path.dirname(__file__), "_selfcheck")
    build("SelfCheck", flow, list(range(N)), "2020-01-01", root=root)
    b = torch.load(os.path.join(root, "SelfCheck", "data.pt"), weights_only=False)
    raw = torch.from_numpy(flow)
    S = L - W - H + 1
    assert b["y"].shape == (S, N, H), b["y"].shape
    assert b["signal"].shape == (L - H, N, 1), b["signal"].shape
    for i in (0, 37, S - 1):
        for h in (0, 5, H - 1):
            assert torch.equal(b["y"][i, :, h], raw[i + W + h]), (i, h)
    assert torch.equal(b["signal"][:, :, 0], raw[: L - H])
    card = json.load(open(os.path.join(root, "SelfCheck", "card.json")))
    assert card["num_samples"] == S == sum(card["split_sizes"].values())
    import shutil
    shutil.rmtree(root)
    print("self-check ok: y[i,:,h] == raw[i+W+h], signal == raw[:L-H], splits partition")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="corpus name, e.g. Pems11")
    ap.add_argument("--glob", required=True,
                    help="station 5-minute files, e.g. 'raw/d11/d11_text_station_5min_*.txt.gz'")
    ap.add_argument("--meta", default=None,
                    help="station metadata file (tab-separated) for the graph")
    ap.add_argument("--root", default=SHAPE_DATA_ROOT)
    ap.add_argument("--min-observed", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--sigma-km", type=float, default=2.0)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        _self_check()
        return

    paths = sorted(glob.glob(a.glob))
    if not paths:
        raise SystemExit(f"no files matched {a.glob!r}")
    print(f"reading {len(paths)} files ...", flush=True)
    wide = drop_sparse(read_station_5min(paths), a.min_observed)
    ids = [int(c) for c in wide.columns]
    adj = None
    if a.meta:
        meta = pd.read_csv(a.meta, sep="\t")
        adj = haversine_adj(meta, ids, a.top_k, a.sigma_km)
    build(a.name, wide.to_numpy(), ids, str(wide.index[0].date()), adj, a.root)


if __name__ == "__main__":
    main()
