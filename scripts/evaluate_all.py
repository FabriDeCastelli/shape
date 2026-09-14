#!/usr/bin/env python3
"""Score every finished run in runs/final/ into results/final/<domain>.jsonl.

    python scripts/evaluate_all.py --device cuda:1

Idempotent: a (checkpoint, target, fold) already in the results file is skipped,
so this can be run repeatedly while the queue is still draining. The target of a
run is derived from where it lives -- a leave-one-out directory is named after
the corpus it held out, and the graph-classification domains hold out a fixed
test set named by the benchmark.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shape.evaluate import EVAL_BATCH, evaluate                      # noqa: E402
from shape.train import configure_backends                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "final")

# The held-out sets of the two graph-classification benchmarks. A run whose
# directory is a pack name is scored on every one of these.
TEST_SETS = {
    "mint": ["MiNT" + n for n in
             ("MIR DOGE20 MUTE EVERMOON DERC ADX HOICHI SDEX BAG XCN ETH2x-FLI "
              "stkAAVE GLM QOM WOJAK DINO Metis REPv2 TRAC BEPRO").split()],
    "social": ["MiNTSocMathOverflow", "MiNTSocRedditB"],
}


def arm_of(run_dir: str) -> str:
    """Which ablation variant this run is, from where it lives.

    The recorded config cannot answer this on its own: `--relations gram` and the
    full model differ there, but `no-layer-agg` and `no-scale` are both just
    supervised fits whose protocol field reads "sup". The directory is the one
    place the variant is unambiguous.
    """
    for p in run_dir.split(os.sep):
        if p.startswith("abl-"):
            return p
    return "full"


def targets_for(domain: str, run_dir: str) -> List[str]:
    """Which corpora this run is a result *about*."""
    if domain in TEST_SETS:
        return TEST_SETS[domain]
    # flow and transport: supervised runs sit under <corpus>/seed<k>, held-out
    # runs under loo/<corpus>/main/seed<k>. Either way the corpus names the dir.
    parts = run_dir.split(os.sep)
    for p in reversed(parts):
        if p not in ("main", "sup") and not p.startswith(("seed", "abl-", "loo")):
            return [p]
    raise ValueError(f"cannot tell what {run_dir!r} is a result about")


def done(path: str) -> set:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if "seen_target" in r:
                out.add((os.path.realpath(r["checkpoint"]), r["target"], r["fold"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fold", default="test", choices=("val", "test"))
    ap.add_argument("--batch-size", type=int, default=EVAL_BATCH)
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--domains", nargs="*", default=["flow", "mint", "social", "transport"])
    a = ap.parse_args()
    configure_backends(a.precision)
    os.makedirs(RESULTS, exist_ok=True)

    for domain in a.domains:
        out = os.path.join(RESULTS, f"{domain}.jsonl")
        seen, n = done(out), 0
        for meta_path in sorted(glob.glob(
                os.path.join(ROOT, "runs", "final", domain, "**", "train.json"),
                recursive=True)):
            run_dir = os.path.dirname(meta_path)
            # Resolved, because the dedup key is compared as a string: the same
            # checkpoint reached by a relative and an absolute path scored twice.
            ckpt = os.path.realpath(
                json.load(open(meta_path)).get("checkpoint") or run_dir)
            for target in targets_for(domain, run_dir):
                if (ckpt, target, a.fold) in seen:
                    continue
                try:
                    row = evaluate(run_dir, target, a.fold, a.device, a.batch_size)
                except Exception as e:                      # one bad run must not stop the sweep
                    print(f"  FAILED {run_dir} on {target}: {type(e).__name__}: {e}", flush=True)
                    continue
                row["arm"] = arm_of(run_dir)
                row["run_dir"] = os.path.relpath(run_dir, ROOT)
                with open(out, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                key = "AUC" if row["metric"] == "rocauc" else "MAE"
                print(f"{domain:10s} {target:22s} "
                      f"{'sup' if row['seen_target'] else 'zero-shot':10s} "
                      f"{key} {row.get(key, float('nan')):.4f}", flush=True)
                n += 1
        print(f"{domain}: {n} new row(s) -> {out}", flush=True)


if __name__ == "__main__":
    main()
