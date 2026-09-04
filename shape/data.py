"""One shape for every dataset: the card, the builder, and the reader.

Every corpus in this project stores something different -- lag windows in the
channel axis, one-hot degree bins, static 128-d embeddings, or nothing at all --
and the model should not know that. A dataset is normalised once into
``signal``/``edge_index``/``edge_weight`` plus a :class:`DatasetCard`, and read
back as the eight tensors ``forward`` takes.

Channels are grouped, not merged: ``real`` are the dataset's own temporal
signals, ``probe`` are ``P_t R`` structural traces for corpora whose nodes carry
no evolving features, ``static`` are time-invariant node embeddings broadcast
across the window. The card records the boundaries so a later stage can send
only the temporal groups through a foundation model.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional

import torch

# Processed datasets + their cards. One variable, so the corpus can move disk
# without touching anything else.
SHAPE_DATA_ROOT = os.path.abspath(
    os.environ.get("SHAPE_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "..", "tgfm_data"))
)

W_DEFAULT = 12
H_MAX = 12
PROBE_DIM = 16
PROBE_SEED = 0

# Sampling calendar per corpus. Only what the source actually documents: a
# dataset whose loader carries no timestamps gets ``None`` and its calendar
# bands are zeroed with u_mask=0, rather than being given an invented date.
CALENDAR: Dict[str, Dict[str, Optional[str]]] = {
    "Pems03": {"freq": "5min", "start": "2018-09-01"},
    "Pems04": {"freq": "5min", "start": "2018-01-01"},
    "Pems07": {"freq": "5min", "start": "2017-05-01"},
    "Pems08": {"freq": "5min", "start": "2016-07-01"},
    "MetrLA": {"freq": "5min", "start": "2012-03-01"},
    "PemsBay": {"freq": "5min", "start": "2017-01-01"},
    "ChickenPox": {"freq": "W", "start": "2005-01-03"},
    "WikiMat": {"freq": "D", "start": None},
    "TwitterTennis": {"freq": None, "start": None},
    "ArXiv": {"freq": "Y", "start": None},
    "DBLP10": {"freq": "Y", "start": None},
}
# u = [tod sin, tod cos, dow sin, dow cos, doy sin, doy cos, t_norm, dt].
# Which bands a frequency can support at all; t_norm and dt are always real.
FREQ_BANDS = {
    "5min": [1, 1, 1, 1, 1, 1, 1, 1],
    "D": [0, 0, 1, 1, 1, 1, 1, 1],
    "W": [0, 0, 0, 0, 1, 1, 1, 1],
    "Y": [0, 0, 0, 0, 0, 0, 1, 1],
    None: [0, 0, 0, 0, 0, 0, 1, 1],
}


@dataclass(frozen=True)
class DatasetCard:
    """Everything a sampler needs to know without opening the tensors."""

    name: str
    task: str                       # forecast | node-classify | graph-classify
    metric: str
    num_classes: int

    num_nodes: int
    num_snapshots: int
    num_samples: int                # after the W washout
    W: int
    H: int
    H_max: int

    C: int
    channel_groups: Dict[str, List[int]]   # name -> [start, stop) into the C axis
    static_feature_dim: int

    static_topology: bool
    has_edge_weight: bool
    edge_weight_semantics: str      # distance | transfer-amount | mentions | none

    node_cap: Optional[int]
    snapshot_cap: Optional[int]
    node_selection: str             # all | top-degree

    split_kind: str                 # temporal | node
    split_sizes: Dict[str, int]

    freq: Optional[str]
    start_date: Optional[str]
    u_available: List[int]          # 8 flags, one per u band

    # Global z-score statistics, computed from the TRAINING portion only. One
    # scalar pair per dataset, applied to the signal (and, for forecasting, to
    # the target so it lives on the same scale the encoder can represent).
    # ``None`` for a corpus with no real signal, e.g. probe-only MiNT.
    global_mean: Optional[float] = None
    global_std: Optional[float] = None
    static_mean: Optional[float] = None
    static_std: Optional[float] = None

    probe_seed: int = PROBE_SEED
    probe_dim: int = PROBE_DIM

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2, sort_keys=True)

    @staticmethod
    def load(path: str) -> "DatasetCard":
        with open(path) as fh:
            return DatasetCard(**json.load(fh))


def storage_of(obj: Any) -> Dict[str, Any]:
    """The key/value store, whether the corpus was saved as ``DynamicData`` or a dict."""
    return obj._storage if hasattr(obj, "_storage") else obj


def probe_matrix(num_nodes: int, dim: int = PROBE_DIM, seed: int = PROBE_SEED) -> torch.Tensor:
    """The fixed Gaussian node probe ``R``.

    Held constant across the window and across snapshots, so temporal variation
    in ``P_t R`` comes from the evolving topology and not from resampled noise.
    """
    return torch.randn(num_nodes, dim, generator=torch.Generator().manual_seed(seed))


def propagate(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor],
              probe: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """``P_t R`` for one snapshot, with ``P_t = D_t^{-1} A_t`` (weights kept).

    Row-normalised rather than symmetric: the trace should read as "where a walk
    from this node lands", and on MiNT the edge weight is a transfer amount that
    only reaches the model through this product.
    """
    out = torch.zeros(num_nodes, probe.shape[1], dtype=probe.dtype)
    if edge_index is None or edge_index.numel() == 0:
        return out
    src, dst = edge_index[0].long(), edge_index[1].long()
    w = torch.ones(src.numel(), dtype=probe.dtype) if edge_weight is None \
        else edge_weight.reshape(-1).to(probe.dtype)
    out.index_add_(0, src, probe[dst] * w.unsqueeze(1))
    degree = torch.zeros(num_nodes, dtype=probe.dtype).index_add_(0, src, w)
    return out / degree.clamp(min=1e-12).unsqueeze(1)


def calendar(name: str, num_steps: int, W: int) -> tuple:
    """``(u, u_mask)`` of shape ``[num_steps, W, 8]``.

    A band the sampling frequency cannot support is zeroed and masked off, so a
    weekly corpus reports "no time of day" instead of a fabricated midnight.
    """
    spec = CALENDAR.get(name, {"freq": None, "start": None})
    freq, start = spec["freq"], spec["start"]
    bands = list(FREQ_BANDS.get(freq, FREQ_BANDS[None]))
    if start is None:                     # no anchor => no wall-clock at all
        bands[:6] = [0] * 6

    u = torch.zeros(num_steps, W, 8)
    # t_norm: position within the window. dt: spacing, constant here since every
    # corpus is sampled on a regular grid.
    u[:, :, 6] = torch.linspace(0, 1, W).unsqueeze(0)
    u[:, :, 7] = 1.0 / max(num_steps, 1)

    if start is not None and freq is not None:
        import pandas as pd
        idx = pd.date_range(start=start, periods=num_steps + W - 1, freq=freq)
        tod = torch.tensor((idx.hour * 60 + idx.minute).values / 1440.0, dtype=torch.float32)
        dow = torch.tensor(idx.dayofweek.values / 7.0, dtype=torch.float32)
        doy = torch.tensor((idx.dayofyear.values - 1) / 366.0, dtype=torch.float32)
        for j, series in enumerate((tod, dow, doy)):
            win = series.unfold(0, W, 1)[:num_steps]
            u[:, :, 2 * j] = torch.sin(2 * torch.pi * win)
            u[:, :, 2 * j + 1] = torch.cos(2 * torch.pi * win)

    mask = torch.tensor(bands, dtype=torch.bool).view(1, 1, 8).expand(num_steps, W, 8).clone()
    u = u * mask
    return u, mask


class ShapeDataset:
    """Reads one normalised corpus and yields the model's eight tensors.

    Windows are cut on access rather than materialised: they are a view of
    ``signal``, and storing them would cost about a terabyte on ArXiv alone.
    """

    def __init__(self, name: str, root: str = SHAPE_DATA_ROOT):
        self.dir = os.path.join(root, name)
        self.card = DatasetCard.load(os.path.join(self.dir, "card.json"))
        blob = torch.load(os.path.join(self.dir, "data.pt"), weights_only=False, map_location="cpu")
        self.signal = blob["signal"]                 # [T_seq, N, C_real] or None
        self.static = blob.get("static")             # [N, D_static] or None
        self.edge_index = blob["edge_index"]         # list[T] of [2, E]
        self.edge_weight = blob.get("edge_weight")   # list[T] of [E] or None
        self.y = blob["y"]
        self.mask = blob.get("mask")
        self.splits = blob["splits"]
        c = self.card
        if c.global_std:
            # The encoder is scale- and shift-invariant, so an un-normalised
            # target is unreachable for it: PeMS08 sat at MAE 89.0 against a
            # historical-average 89.8. Both signal and target move onto one
            # scale here; the trainer maps predictions back for reporting.
            self.signal = (self.signal - c.global_mean) / c.global_std
            if c.task == "forecast":
                self.y = (self.y - c.global_mean) / c.global_std
        if c.static_std and self.static is not None:
            self.static = (self.static - c.static_mean) / c.static_std

        self.probe = probe_matrix(c.num_nodes, c.probe_dim, c.probe_seed) \
            if c.channel_groups.get("probe") else None
        self.u, self.u_mask = calendar(self.card.name, self.card.num_samples, self.card.W)

    def __len__(self) -> int:
        return self.card.num_samples

    def _edges_at(self, t: int):
        ei = self.edge_index[t] if isinstance(self.edge_index, list) else self.edge_index
        ew = None
        if self.edge_weight is not None:
            ew = self.edge_weight[t] if isinstance(self.edge_weight, list) else self.edge_weight
        return ei, ew

    def __getitem__(self, i: int) -> Dict[str, Any]:
        card, W = self.card, self.card.W
        t = i + (card.num_snapshots - card.num_samples)      # absolute snapshot of the window end
        parts = []
        if self.signal is not None:
            # signal is the unrolled series, so the window ending at snapshot t
            # starts at i in its own axis.
            parts.append(self.signal[i:i + W].permute(1, 0, 2))          # [N, W, C_real]
        if self.probe is not None:
            steps = [propagate(*self._edges_at(max(0, t - W + 1 + k)), self.probe, card.num_nodes)
                     for k in range(W)]
            parts.append(torch.stack(steps, dim=1))                      # [N, W, probe_dim]
        if self.static is not None:
            parts.append(self.static.unsqueeze(1).expand(-1, W, -1))     # [N, W, D_static]
        x = torch.cat(parts, dim=-1)

        ei, ew = self._edges_at(t)
        return {
            "x": x,
            "edge_index": ei,
            "edge_weight": ew,
            "u": self.u[i],
            "u_mask": self.u_mask[i],
            "x_tsfm": None,      # produced by a later stage; the slot is fixed now
            "pe": None,
            "mask": None if self.mask is None else self.mask[t],
            "y": self.y[t],
        }


def _self_check() -> None:
    """Runs against synthetic tensors: shapes, channel groups, probe determinism."""
    N, T, W, Creal, Dstat = 6, 20, W_DEFAULT, 1, 4
    ei = [torch.randint(0, N, (2, 10)) for _ in range(T)]
    ew = [torch.rand(10) + 0.1 for _ in range(T)]

    R = probe_matrix(N)
    assert torch.equal(R, probe_matrix(N)), "probe must be reproducible from its seed"
    p0 = propagate(ei[0], ew[0], R, N)
    assert p0.shape == (N, PROBE_DIM) and torch.isfinite(p0).all()
    assert torch.equal(propagate(None, None, R, N), torch.zeros(N, PROBE_DIM)), "empty graph -> zeros"
    # An isolated node has no outgoing walk, so its trace must be exactly zero.
    lone = torch.tensor([[0, 1], [1, 0]])
    assert propagate(lone, None, R, N)[2].abs().sum() == 0

    u, um = calendar("ChickenPox", T, W)
    assert u.shape == (T, W, 8) and um.shape == (T, W, 8)
    assert not um[..., :4].any(), "weekly data has no time-of-day or day-of-week"
    assert um[..., 4:].all(), "day-of-year, t_norm and dt are available weekly"
    assert (u[..., :4] == 0).all(), "masked-off bands must be zero, not fabricated"
    u2, um2 = calendar("TwitterTennis", T, W)
    assert not um2[..., :6].any(), "no anchor date -> no wall-clock bands"
    assert um2[..., 6:].all(), "t_norm and dt survive without a calendar"

    groups = {"real": [0, Creal], "probe": [Creal, Creal + PROBE_DIM],
              "static": [Creal + PROBE_DIM, Creal + PROBE_DIM + Dstat]}
    card = DatasetCard(
        name="Synthetic", task="forecast", metric="mae", num_classes=0,
        num_nodes=N, num_snapshots=T, num_samples=T - W + 1, W=W, H=1, H_max=H_MAX,
        C=Creal + PROBE_DIM + Dstat, channel_groups=groups, static_feature_dim=Dstat,
        static_topology=False, has_edge_weight=True, edge_weight_semantics="transfer-amount",
        node_cap=None, snapshot_cap=None, node_selection="all",
        split_kind="temporal", split_sizes={"train": 5, "val": 2, "test": 2},
        freq="W", start_date="2005-01-03", u_available=FREQ_BANDS["W"],
    )
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "Synthetic"))
        card.save(os.path.join(d, "Synthetic", "card.json"))
        assert DatasetCard.load(os.path.join(d, "Synthetic", "card.json")) == card, "card must round-trip"
        torch.save({"signal": torch.randn(T + W - 1, N, Creal), "static": torch.randn(N, Dstat),
                    "edge_index": ei, "edge_weight": ew, "y": torch.randn(T, N, 1),
                    "mask": None, "splits": {"kind": "temporal"}},
                   os.path.join(d, "Synthetic", "data.pt"))
        ds = ShapeDataset("Synthetic", root=d)
        assert len(ds) == T - W + 1
        b = ds[0]
        assert b["x"].shape == (N, W, card.C), b["x"].shape
        assert b["u"].shape == (W, 8) and b["u_mask"].shape == (W, 8)
        assert b["x_tsfm"] is None and b["pe"] is None
        # The static block must be constant across the window; the probe must not be.
        s = slice(*groups["static"])
        assert (b["x"][:, 0, s] == b["x"][:, -1, s]).all(), "static channels must not vary in time"
        p = slice(*groups["probe"])
        assert not torch.equal(b["x"][:, 0, p], b["x"][:, -1, p]), "probe channels must vary with topology"
    print("tgfm self-check ok")


if __name__ == "__main__":
    _self_check()
