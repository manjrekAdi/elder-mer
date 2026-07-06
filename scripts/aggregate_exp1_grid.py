#!/usr/bin/env python3
"""
aggregate_exp1_grid.py -- build the Experiment 1 backbone x readout comparison table.
=====================================================================================

Scans a grid produced by run_exp1_grid.sh:
    <grid>/<modelshort>_<readout>/foldN/predictions.csv

For each completed cell it combines the folds' held-out predictions (all 91
actors) and computes, reusing analyze_exp1_age_gap's functions:
  - overall accuracy / macro-F1 / weighted-F1
  - the per-actor age regression (accuracy and macro-F1 slope per decade + p),
    i.e. the young-vs-old gap for that backbone+readout.

Writes <grid>/grid_summary.csv and prints a readable table sorted by model then
readout, so you can compare backbones (does the age gap persist across all of
them?) and readouts (head vs generative) side by side.

USAGE:
    python scripts/aggregate_exp1_grid.py --grid data/crema_d/exp1_grid
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_exp1_age_gap import load_predictions, per_actor_table, ols, overall_metrics


def split_cell_name(name):
    """qwen7b_head -> ('qwen7b','head'); gemma9b_generative -> ('gemma9b','generative')."""
    for ro in ("head", "generative"):
        if name.endswith("_" + ro):
            return name[: -(len(ro) + 1)], ro
    return name, "?"


def main():
    ap = argparse.ArgumentParser(description="Aggregate the Experiment 1 backbone x readout grid")
    ap.add_argument("--grid", default="data/crema_d/exp1_grid")
    ap.add_argument("--out", default=None, help="output CSV (default <grid>/grid_summary.csv)")
    args = ap.parse_args()

    grid = Path(args.grid)
    if not grid.exists():
        raise SystemExit(f"[error] grid dir not found: {grid}")
    out_csv = Path(args.out) if args.out else grid / "grid_summary.csv"

    cells = sorted(p for p in grid.iterdir() if p.is_dir())
    rows_out = []
    for cell in cells:
        model, ro = split_cell_name(cell.name)
        try:
            files, rows = load_predictions(str(cell / "fold*" / "predictions.csv"))
        except SystemExit:
            continue  # no predictions in this cell yet
        if not rows:
            continue
        ov = overall_metrics(rows)
        table = per_actor_table(rows)
        acc_fit = ols([d["age"] for d in table], [d["accuracy"] for d in table])
        f1_fit = ols([d["age"] for d in table], [d["macro_f1"] for d in table])
        rows_out.append({
            "model": model, "readout": ro, "folds": len(files),
            "n_clips": ov["n"], "n_actors": len(table),
            "accuracy": round(ov["accuracy"], 4),
            "macro_f1": round(ov["macro_f1"], 4),
            "weighted_f1": round(ov["weighted_f1"], 4),
            "acc_slope_decade": round(acc_fit["slope_decade"], 4),
            "acc_slope_p": round(acc_fit["p"], 4),
            "macrof1_slope_decade": round(f1_fit["slope_decade"], 4),
            "macrof1_slope_p": round(f1_fit["p"], 4),
        })

    if not rows_out:
        raise SystemExit(f"[error] no completed cells (with predictions.csv) under {grid}")

    rows_out.sort(key=lambda r: (r["model"], r["readout"]))
    fields = ["model", "readout", "folds", "n_clips", "n_actors",
              "accuracy", "macro_f1", "weighted_f1",
              "acc_slope_decade", "acc_slope_p", "macrof1_slope_decade", "macrof1_slope_p"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    # readable table
    print(f"\nExperiment 1 grid summary  ({len(rows_out)} cells)  ->  {out_csv}\n")
    hdr = (f"{'model':11s} {'readout':10s} {'folds':>5s} {'acc':>7s} {'macroF1':>8s} "
           f"{'accSlope/dec':>13s} {'p':>7s}  sig")
    print(hdr)
    print("-" * len(hdr))
    for r in rows_out:
        sig = "*" if r["acc_slope_p"] < 0.05 else ""
        note = " (partial)" if r["folds"] < 5 else ""
        print(f"{r['model']:11s} {r['readout']:10s} {r['folds']:>5d} {r['accuracy']:>7.4f} "
              f"{r['macro_f1']:>8.4f} {r['acc_slope_decade']:>+13.4f} {r['acc_slope_p']:>7.4f}  {sig}{note}")
    print("\n(* = age slope significant at p<0.05; a negative slope = the young-vs-old gap.)")


if __name__ == "__main__":
    main()
