#!/usr/bin/env python3
"""
elderreact_visual_reliability.py -- OpenFace reliability on in-the-wild elderly video
======================================================================================

Follow-up to Experiment 3's null result on CREMA-D (all reliability signals
age-flat because OpenFace saturates at its 0.98 confidence cap on clean
studio video). This runs the identical extraction + aggregation over
ElderReact's 1,323 in-the-wild clips and compares the reliability
distributions against CREMA-D's.

What it decides: whether the visual confidence gate (Contribution 2) has
signal to act on in realistic elderly recordings. ElderReact has no
per-person ages, so this is a distribution comparison (studio vs in-the-wild
elderly), not an age regression.

Usage:
    python scripts/elderreact_visual_reliability.py             # extract + report
    python scripts/elderreact_visual_reliability.py --analyze   # report only

Outputs (under data/elderreact/visual_reliability/):
    per_frame/<clip>.csv    raw OpenFace output (reused later as gate inputs)
    per_clip.csv            one row per clip: signals + split/gender
    summary.txt             distribution stats + CREMA-D comparison
"""

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from experiment3_visual_gate import (
    aggregate_clip, output_ok, run_extraction, percentile)

SPLITS = ("train", "dev", "test")


def load_clips(er_dir: Path):
    """Clip list with split + gender from the annotation files."""
    gender = {}
    for s in SPLITS:
        lab = er_dir / "_elderreact_repo" / "Annotations" / f"{s}_labels.txt"
        if lab.exists():
            for line in lab.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 8:
                    gender[parts[0]] = parts[7]
    clips = []
    for s in SPLITS:
        for video in sorted((er_dir / f"ElderReact_{s}").glob("*.mp4")):
            clips.append({"stem": video.stem, "video": video, "split": s,
                          "gender": gender.get(video.name, "")})
    return clips


def summarize(rows, label):
    conf = sorted(float(r["conf_mean"]) for r in rows)
    succ = [float(r["success_rate"]) for r in rows]
    n = len(conf)
    frac = lambda xs, pred: 100.0 * sum(pred(x) for x in xs) / len(xs)
    return [
        f"{label}: {n} clips",
        f"  conf_mean      : mean={sum(conf)/n:.4f}  median={percentile(conf,50):.4f}  "
        f"p10={percentile(conf,10):.4f}  min={conf[0]:.4f}",
        f"  conf_mean<0.95 : {frac(conf, lambda x: x < 0.95):.1f}%   "
        f"<0.90: {frac(conf, lambda x: x < 0.90):.1f}%   "
        f"<0.80: {frac(conf, lambda x: x < 0.80):.1f}%",
        f"  any tracking failure (success<1): {frac(succ, lambda x: x < 1):.1f}%",
        f"  success_rate   : mean={sum(succ)/n:.4f}  p10={percentile(sorted(succ),10):.4f}",
    ]


def main():
    ap = argparse.ArgumentParser(description="OpenFace reliability on ElderReact.")
    ap.add_argument("--data", default="./data/elderreact")
    ap.add_argument("--cremad-perclip", default="./data/crema_d/experiment3_visual/per_clip.csv")
    ap.add_argument("--analyze", action="store_true", help="skip extraction")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    er_dir = Path(args.data).expanduser().resolve()
    out_dir = er_dir / "visual_reliability"
    per_frame_dir = out_dir / "per_frame"
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = load_clips(er_dir)
    print(f"[data] {len(clips)} ElderReact clips")
    if not clips:
        sys.exit("[error] no videos found; check --data")

    if not args.analyze:
        run_extraction(clips, per_frame_dir, args.workers)

    rows = []
    miss = 0
    for c in clips:
        p = per_frame_dir / f"{c['stem']}.csv"
        agg = aggregate_clip(p) if output_ok(p) else None
        if agg is None:
            miss += 1
            continue
        rows.append({"file": c["stem"], "split": c["split"],
                     "gender": c["gender"], **agg})
    if miss:
        print(f"[aggregate] {miss} clips without usable output")
    if not rows:
        sys.exit("[error] nothing aggregated; run extraction first.")
    with open(out_dir / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[aggregate] wrote {len(rows)} rows -> {out_dir/'per_clip.csv'}")

    lines = ["OpenFace reliability: ElderReact (in-the-wild elderly) vs CREMA-D (studio)",
             "=" * 76, ""]
    lines += summarize(rows, "ElderReact")
    cremad = Path(args.cremad_perclip)
    if cremad.exists():
        with open(cremad, newline="") as f:
            lines += [""] + summarize(list(csv.DictReader(f)), "CREMA-D (reference)")
    else:
        lines += ["", f"[note] CREMA-D per_clip.csv not found at {cremad}; no comparison."]
    lines += ["", "READING", "-------",
              "If ElderReact shows a fat low-confidence tail where CREMA-D is",
              "saturated at ~0.98, the visual gate has real signal on in-the-wild",
              "elderly video, and Experiment 4 validates the gate there."]
    report = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
