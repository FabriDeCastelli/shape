"""Normalise TIDES' exports into the SHAPE layout, one card each.

Reads whatever a corpus happens to be stored as -- ``DynamicData`` or a plain
dict, lag windows or one-hot bins or a static embedding -- and writes
``$SHAPE_DATA_ROOT/<name>/{data.pt,card.json}``.

The per-dataset knowledge lives in :data:`SPEC` and nowhere else. Everything
downstream reads the card.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from shape.data import (CALENDAR, FREQ_BANDS, H_MAX, PROBE_DIM, PROBE_SEED,
                           SHAPE_DATA_ROOT, W_DEFAULT, DatasetCard, storage_of)

SRC_ROOT = "/raid/f.decastelli/data/pyg"


@dataclass(frozen=True)
class Spec:
    """How one corpus maps onto the shared layout."""
    src: str                       # <task-dir>/<name>/<setting>
    task: str
    metric: str
    num_classes: int = 0
    H: int = 1
    channels_are_windows: bool = True
    probe: bool = False            # add P_t R structural channels
    edge_weight_semantics: str = "none"
    decode_onehot: Optional[tuple] = None   # block widths to argmax back to ordinals


SPEC: Dict[str, Spec] = {
    "ChickenPox": Spec("node-level-forecasting/ChickenPox/default", "forecast", "mse"),
    "WikiMat": Spec("node-level-forecasting/WikiMat/default", "forecast", "mse"),
    # 16 channels are 5 degree bins + 11 transitivity bins; argmax recovers the
    # ordinal each block encodes, which is the signal the model should see.
    "TwitterTennis": Spec("node-level-forecasting/TwitterTennis/default", "forecast", "mse",
                          channels_are_windows=False, edge_weight_semantics="mentions",
                          decode_onehot=(5, 11)),
    "Pems03": Spec("node-level-forecasting/Pems03/tides", "forecast", "masked_mae", H=12,
                   edge_weight_semantics="distance"),
    "Pems04": Spec("node-level-forecasting/Pems04/tides", "forecast", "masked_mae", H=12,
                   edge_weight_semantics="distance"),
    "Pems07": Spec("node-level-forecasting/Pems07/tides", "forecast", "masked_mae", H=12,
                   edge_weight_semantics="distance"),
    "Pems08": Spec("node-level-forecasting/Pems08/tides", "forecast", "masked_mae", H=12,
                   edge_weight_semantics="distance"),
    "MetrLA": Spec("node-level-forecasting/MetrLA/tides", "forecast", "masked_mae", H=12,
                   edge_weight_semantics="distance"),
    "PemsBay": Spec("node-level-forecasting/PemsBay/tides", "forecast", "masked_mae", H=12,
                    edge_weight_semantics="distance"),
    # Static embeddings, growing topology: the probe carries the time variation.
    "ArXiv": Spec("node-level-forecasting/ArXiv/default", "node-classify", "accuracy",
                  num_classes=40, channels_are_windows=False, probe=True),
    # Embeddings evolve (42% of nodes change per step), so no probe is needed.
    "DBLP10": Spec("node-level-forecasting/DBLP10/default", "node-classify", "accuracy",
                   num_classes=10, channels_are_windows=False),
}


def mint_spec(name: str) -> Spec:
    """A MiNT token network: no node features at all, so probe channels only."""
    setting = os.listdir(os.path.join(SRC_ROOT, "graph-level-classification", name))[0]
    return Spec(f"graph-level-classification/{name}/{setting}", "graph-classify", "rocauc",
                num_classes=2, channels_are_windows=False, probe=True,
                edge_weight_semantics="transfer-amount")


def unroll(x: torch.Tensor) -> torch.Tensor:
    """Lag windows in the channel axis -> the univariate series they came from."""
    return torch.cat([x[0, :, :-1].transpose(0, 1), x[:, :, -1]], dim=0).unsqueeze(-1)


def as_list(v: Any, T: int) -> Optional[List[torch.Tensor]]:
    if v is None:
        return None
    return v if isinstance(v, list) else [v] * T


def _global_stats(signal, static, splits, split_kind, W):
    """Train-only z-score statistics.

    Leakage is the whole risk here. Under a temporal split the last training
    sample reads ``signal[n_train-1 : n_train-1+W]``, so exactly
    ``n_train + W - 1`` steps have been seen and nothing beyond may inform the
    statistics. Under a node split every snapshot is visible but only the
    training nodes are.
    """
    gm = gs = sm = ss = None
    if signal is not None:
        if split_kind == "temporal":
            seen = signal[: len(splits["train"]) + W - 1]
        else:
            seen = signal[:, splits["train"].bool()]
        gm, gs = float(seen.mean()), float(seen.std().clamp(min=1e-6))
    if static is not None:
        seen = static if split_kind == "temporal" else static[splits["train"].bool()]
        sm, ss = float(seen.mean()), float(seen.std().clamp(min=1e-6))
    return gm, gs, sm, ss


def build(name: str, spec: Spec, root: str = SHAPE_DATA_ROOT) -> DatasetCard:
    blob = torch.load(os.path.join(SRC_ROOT, spec.src, "data.pt"),
                      weights_only=False, map_location="cpu")
    st = storage_of(blob)
    static_keys = getattr(blob, "_static_keys", set()) or set()

    edge_index = st["edge_index"]
    T = int(st.get("num_snapshots", len(edge_index) if isinstance(edge_index, list) else len(st["y"])))
    edge_index = as_list(edge_index, T)
    edge_weight = as_list(st.get("edge_attr"), T)

    x = st.get("signal", st.get("x"))
    signal, static = None, None
    if x is None:
        num_nodes = int(st["num_nodes"])
    elif "x" in static_keys and x.dim() == 2:
        # One embedding per node, no time axis.
        num_nodes = x.shape[0]
        static = x.float()
        if len(static.unique()) == 1:
            static = None                      # a constant placeholder is not a feature
    else:
        num_nodes = x.shape[1]
        signal = unroll(x.float()) if spec.channels_are_windows else x.float()
        if spec.decode_onehot:
            blocks, off = [], 0
            for width in spec.decode_onehot:
                blocks.append(signal[..., off:off + width].argmax(-1).float())
                off += width
            signal = torch.stack(blocks, dim=-1)

    if static is None and signal is None and not spec.probe:
        raise ValueError(f"{name}: no channels at all -- needs probe=True")

    groups, c = {}, 0
    if signal is not None:
        groups["real"] = [0, signal.shape[-1]]
        c = signal.shape[-1]
    if spec.probe:
        groups["probe"] = [c, c + PROBE_DIM]
        c += PROBE_DIM
    if static is not None:
        groups["static"] = [c, c + static.shape[-1]]
        c += static.shape[-1]

    seq_len = signal.shape[0] if signal is not None else T
    num_samples = max(0, seq_len - W_DEFAULT + 1)

    y = st["y"]
    if y.dim() == 2 and spec.task == "forecast":
        y = y.unsqueeze(-1)
    mask = st.get("mask")
    if mask is not None and spec.metric.startswith("masked"):
        # PeMS writes a missing reading as a literal 0 while its stored mask
        # stays all-ones, so the stored mask alone marks nothing invalid.
        mask = mask.bool() & (y != 0)

    if "train_mask" in st:                     # transductive: the split is over nodes
        split_kind = "node"
        splits = {k: st[f"{k}_mask"][0] for k in ("train", "val", "test")}
        sizes = {k: int(v.sum()) for k, v in splits.items()}
    else:
        split_kind = "temporal"
        raw = st.get("splits") or _tides_split(spec.src)
        if raw is None:
            # Inventing a split here is how two comparisons were silently run
            # against the wrong test set: a plausible-looking 0.1/0.1 fallback
            # is indistinguishable from the real thing until a metric goes NaN.
            raise FileNotFoundError(
                f"{name}: no split found. Expected `splits` in the export or an artifact under "
                f"{os.path.join(SRC_ROOT.replace('/pyg', '/splits'), spec.src)}. "
                f"Refusing to invent one."
            )
        splits = _align(raw, T - num_samples)
        sizes = {k: int(len(v)) for k, v in splits.items()}

    gm, gs, sm, ss = _global_stats(signal, static, splits, split_kind, W_DEFAULT)

    cal = CALENDAR.get(name, {"freq": None, "start": None})
    bands = list(FREQ_BANDS.get(cal["freq"], FREQ_BANDS[None]))
    if cal["start"] is None:
        bands[:6] = [0] * 6

    card = DatasetCard(
        name=name, task=spec.task, metric=spec.metric, num_classes=spec.num_classes,
        num_nodes=num_nodes, num_snapshots=T, num_samples=num_samples,
        W=W_DEFAULT, H=spec.H, H_max=H_MAX,
        C=c, channel_groups=groups,
        static_feature_dim=0 if static is None else static.shape[-1],
        static_topology=not isinstance(st.get("edge_index"), list),
        has_edge_weight=edge_weight is not None,
        edge_weight_semantics=spec.edge_weight_semantics,
        node_cap=None, snapshot_cap=None, node_selection="all",
        split_kind=split_kind, split_sizes=sizes,
        freq=cal["freq"], start_date=cal["start"], u_available=bands,
        global_mean=gm, global_std=gs, static_mean=sm, static_std=ss,
        probe_seed=PROBE_SEED, probe_dim=PROBE_DIM,
    )

    out = os.path.join(root, name)
    os.makedirs(out, exist_ok=True)
    torch.save({"signal": signal, "static": static, "edge_index": edge_index,
                "edge_weight": edge_weight, "y": y, "mask": mask, "splits": splits},
               os.path.join(out, "data.pt"))
    card.save(os.path.join(out, "card.json"))
    return card


def _tides_split(src: str) -> Optional[Dict[str, torch.Tensor]]:
    """TIDES' own split artifact, so a number computed here is computed on its test set."""
    import glob
    hits = glob.glob(os.path.join(SRC_ROOT.replace("/pyg", "/splits"), src, "*.pt"))
    return torch.load(hits[0], weights_only=False) if hits else None


def _align(raw: Dict[str, torch.Tensor], washout: int) -> Dict[str, torch.Tensor]:
    """Snapshot-level indices re-indexed onto the post-washout sample axis.

    The sample axis starts at snapshot ``washout``, so it is the *trailing* train
    indices that fall off; val and test shift down by the same amount.
    """
    out = {k: torch.as_tensor(raw[k]) for k in ("train", "val", "test")}
    if washout <= 0:
        return out
    return {"train": out["train"][:-washout], "val": out["val"] - washout,
            "test": out["test"] - washout}


def all_specs() -> Dict[str, Spec]:
    specs = dict(SPEC)
    mint_dir = os.path.join(SRC_ROOT, "graph-level-classification")
    if os.path.isdir(mint_dir):
        for n in sorted(os.listdir(mint_dir)):
            if n.startswith("MiNT"):
                specs[n] = mint_spec(n)
    return specs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", help="default: everything in SPEC + all MiNT")
    ap.add_argument("--root", default=SHAPE_DATA_ROOT)
    a = ap.parse_args()

    specs = all_specs()
    names = a.datasets or list(specs)
    print(f"root: {a.root}")
    for n in names:
        try:
            c = build(n, specs[n], a.root)
            print(f"  {c.name:16s} {c.task:15s} N={c.num_nodes:<7,} T={c.num_snapshots:<6,} "
                  f"samples={c.num_samples:<6,} C={c.C:<4} {c.channel_groups}")
        except Exception as exc:
            print(f"  {n:16s} FAILED: {type(exc).__name__}: {exc}")
