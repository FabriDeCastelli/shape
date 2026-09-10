"""Re-window an existing forecasting corpus at a longer input context.

Our own ridge probe puts 4.4-6.7 MAE on the table between W=12 and W=288 --
larger than every architectural effect measured in this project combined -- and
the two strongest published methods on this benchmark (STEP, STD-MAE) are both
long-context methods. The model needs no change: ``TemporalGraphEncoder`` is a
``Linear(window, 2*hidden)``, so it simply widens.

Built as a separate corpus on disk rather than a runtime flag, the same way
``make_season.py`` works, so nothing in a running suite can pick up a changed
window mid-flight.

The raw series is reconstructed from the stored corpus rather than re-read from
source: ``signal`` holds ``raw[:L-H]`` and the final target row holds the last
``H`` steps, which together are exactly ``raw``. Windowing, targets, masking and
splits then go through ``make_pems_raw.build``, which reproduces the shipped
PeMS corpora bit-for-bit.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from export.make_pems_raw import build
from shape.data import SHAPE_DATA_ROOT


def raw_series(blob: dict, H: int) -> torch.Tensor:
    """``[L, N]`` original series, from ``signal`` plus the final target row."""
    sig = blob["signal"][..., 0]                       # [L-H, N]
    tail = blob["y"][-1].T                             # [H, N] -- raw[L-H:L]
    return torch.cat([sig, tail], dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="existing corpus name, e.g. Pems08")
    ap.add_argument("--window", type=int, required=True, help="new W, e.g. 288")
    ap.add_argument("--name", default=None, help="default <source>W<window>")
    ap.add_argument("--root", default=SHAPE_DATA_ROOT, help="where to write")
    ap.add_argument("--src-root", default=SHAPE_DATA_ROOT, help="where to read the source")
    ap.add_argument("--stride", type=int, default=1,
                    help="DO NOT USE for like-for-like comparisons: striding the raw "
                         "series also strides the targets, so H=12 stops meaning one "
                         "hour ahead and the corpus is no longer comparable to the "
                         "W=12 baseline. Input-only striding needs a data-layer "
                         "change, not a rebuild.")
    ap.add_argument("--align-splits", action="store_true",
                    help="pin val/test to the source corpus's fold sizes, so the "
                         "widened corpus predicts the SAME target timestamps as "
                         "the published protocol and the shortfall comes off the "
                         "front of training. Without this the folds slide with W "
                         "and the numbers are not comparable to the baselines.")
    a = ap.parse_args()

    src = os.path.join(a.src_root, a.source)
    card = json.load(open(os.path.join(src, "card.json")))
    blob = torch.load(os.path.join(src, "data.pt"), weights_only=False, map_location="cpu")
    raw = raw_series(blob, card["H"])
    print(f"{a.source}: raw {tuple(raw.shape)}, W {card['W']} -> {a.window}", flush=True)

    if a.stride > 1:
        # A strided context covers the same span with fewer inputs, which is the
        # cheap alternative to patching if the full-resolution window is too big.
        raw = raw[:: a.stride]
        print(f"  strided by {a.stride}: {tuple(raw.shape)}", flush=True)

    adj = None
    ei = blob.get("edge_index")
    if ei is not None:
        adj = (ei[0], blob["edge_weight"][0])          # static topology
    ids = json.load(open(os.path.join(src, "provenance.json")))["station_ids"] \
        if os.path.exists(os.path.join(src, "provenance.json")) \
        else list(range(raw.shape[1]))

    hold = None
    if a.align_splits:
        hold = (card["split_sizes"]["val"], card["split_sizes"]["test"])
        print(f"  aligning folds: val={hold[0]} test={hold[1]} "
              f"(source W={card['W']})", flush=True)
    build(a.name or f"{a.source}W{a.window}", raw.numpy(), ids, card["start_date"],
          adj, a.root, W=a.window, H=card["H"], hold=hold)


if __name__ == "__main__":
    main()
