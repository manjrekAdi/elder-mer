#!/usr/bin/env python3
"""
experiment3_visual_gate.py -- Experiment 3: does OpenFace reliability fall with actor age?
===========================================================================================

The H1 decision gate (proposal section 3.5, primary form). Runs OpenFace
FeatureExtraction over CREMA-D videos, aggregates per-clip reliability
signals, and regresses each signal on actor age across all 91 actors.

Signals tested (H1 names them explicitly):
  1. face-detection/tracking confidence  (OpenFace `confidence` column)
       -> conf_mean, conf_p10, conf_min
  2. tracking success rate               (OpenFace `success` column)
       -> success_rate
  3. landmark-fit instability            (no direct fit-error column exists,
       so we use temporal landmark jitter: mean frame-to-frame displacement
       of the 68 landmarks, normalised by inter-ocular distance)
       -> jitter_mean

Decision logic (H1): a significant negative slope on confidence (or success)
confirms the visual gate's primary signal. If confidence is flat, the gate
reroutes to whichever fallback signal carries the age trend. Only if ALL
signals are age-flat does the visual instance of Contribution 2 lose its
premise on CREMA-D.

Per-frame CSVs are kept on disk (-2Dfp -aus -pose -pdmparams): the same
confidence signals become the gate inputs in Experiment 4, so this
extraction is reused, not thrown away.

Usage:
    # smoke test (a few clips)
    python scripts/experiment3_visual_gate.py --limit 2

    # full overnight extraction (all 7,442 clips, resumable, 8 workers)
    python scripts/experiment3_visual_gate.py --full

    # aggregation + regression only (after extraction is done)
    python scripts/experiment3_visual_gate.py --analyze

Outputs (under <data>/experiment3_visual/):
    per_frame/<clip>.csv     raw OpenFace output, one per clip (kept for Exp 4)
    per_clip.csv             one row per clip: signals + age/sex/emotion
    per_actor.csv            per-actor means (each actor counts once)
    regression.txt           slope, 95% CI, p per signal + H1 verdict
    <signal>_vs_age.png      scatter + fitted line per signal
"""

import argparse
import csv
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OPENFACE_BIN = Path.home() / "tools/OpenFace/build/bin/FeatureExtraction"
OPENFACE_FLAGS = ["-2Dfp", "-aus", "-pose", "-pdmparams"]

# signals regressed on age: (column in per_clip.csv, expected direction if H1 holds)
SIGNALS = {
    "conf_mean":    "negative",
    "conf_p10":     "negative",
    "conf_min":     "negative",
    "success_rate": "negative",
    "jitter_mean":  "positive",
}


def ensure(pkg, pip_name=None):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"[setup] installing {pip_name or pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pip_name or pkg],
                       check=True)
        return __import__(pkg)


# ----------------------------------------------------------------------------
# clip list from metadata.csv (+ sex from VideoDemographics.csv)
# ----------------------------------------------------------------------------
def load_clips(data_dir: Path):
    meta = data_dir / "metadata.csv"
    if not meta.exists():
        sys.exit(f"[error] {meta} not found. Run download_cremad.py first.")
    sex_by_actor = {}
    demo = data_dir / "VideoDemographics.csv"
    if demo.exists():
        with open(demo, newline="") as f:
            for row in csv.DictReader(f):
                sex_by_actor[str(row["ActorID"]).strip()] = row["Sex"]
    clips = []
    with open(meta, newline="") as f:
        for row in csv.DictReader(f):
            try:
                age = int(row["age"])
            except (ValueError, KeyError):
                continue
            stem = Path(row["file"]).stem
            video = data_dir / "VideoFlash" / f"{stem}.flv"
            clips.append({
                "stem": stem, "video": video, "actor_id": row["actor_id"],
                "age": age, "sex": sex_by_actor.get(row["actor_id"], ""),
                "emotion": row.get("emotion", ""),
            })
    return clips


# ----------------------------------------------------------------------------
# extraction (parallel, resumable)
# ----------------------------------------------------------------------------
def output_ok(csv_path: Path) -> bool:
    """A clip is done if its CSV exists and has a header plus >=1 data row."""
    try:
        with open(csv_path) as f:
            f.readline()
            return bool(f.readline())
    except OSError:
        return False


def extract_one(clip, per_frame_dir: Path):
    out_csv = per_frame_dir / f"{clip['stem']}.csv"
    if output_ok(out_csv):
        return "skipped"
    if not clip["video"].exists():
        return f"missing video: {clip['video'].name}"
    cmd = [str(OPENFACE_BIN), "-f", str(clip["video"]),
           "-out_dir", str(per_frame_dir)] + OPENFACE_FLAGS
    # Clamp per-process threading to 1: with N parallel workers, OpenCV's
    # spin-waiting parallel_for pool (and OpenBLAS's) oversubscribe the cores
    # catastrophically (measured ~400x slowdown at 8 workers x 10 threads).
    env = {**os.environ, "OPENCV_FOR_THREADS_NUM": "1",
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    res = subprocess.run(cmd, cwd=OPENFACE_BIN.parent, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res.returncode != 0 or not output_ok(out_csv):
        return f"FAILED: {clip['stem']} ({res.stderr.decode(errors='replace')[-120:].strip()})"
    return "ok"


def run_extraction(clips, per_frame_dir: Path, workers: int):
    if not OPENFACE_BIN.exists():
        sys.exit(f"[error] OpenFace binary not found at {OPENFACE_BIN}")
    per_frame_dir.mkdir(parents=True, exist_ok=True)
    done = ok = skipped = 0
    failures = []
    print(f"[extract] {len(clips)} clips, {workers} workers -> {per_frame_dir}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_one, c, per_frame_dir): c for c in clips}
        for fut in as_completed(futures):
            status = fut.result()
            done += 1
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failures.append(status)
            if done % 100 == 0 or done == len(clips):
                print(f"  [{done}/{len(clips)}] ok={ok} skipped={skipped} failed={len(failures)}")
    if failures:
        print(f"[extract] {len(failures)} failures (first 10):")
        for msg in failures[:10]:
            print(f"    {msg}")
    return failures


# ----------------------------------------------------------------------------
# per-clip aggregation
# ----------------------------------------------------------------------------
def percentile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    idx = min(int(q / 100 * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def aggregate_clip(csv_path: Path):
    """Parse one OpenFace per-frame CSV -> reliability aggregates."""
    confs, successes = [], []
    jitters = []
    prev_lm = None
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        # OpenFace pads column names with a leading space
        cols = {c.strip(): c for c in reader.fieldnames}
        xcols = [cols[f"x_{i}"] for i in range(68) if f"x_{i}" in cols]
        ycols = [cols[f"y_{i}"] for i in range(68) if f"y_{i}" in cols]
        for row in reader:
            # guard against truncated rows (e.g. a clip killed mid-write)
            if row.get(cols["confidence"]) is None or None in row.values():
                continue
            conf = float(row[cols["confidence"]])
            succ = int(row[cols["success"]])
            confs.append(conf)
            successes.append(succ)
            if xcols and succ == 1:
                lm = [(float(row[x]), float(row[y])) for x, y in zip(xcols, ycols)]
                # inter-ocular distance: outer eye corners, landmarks 36 and 45
                iod = math.dist(lm[36], lm[45]) or 1.0
                if prev_lm is not None:
                    disp = sum(math.dist(a, b) for a, b in zip(lm, prev_lm)) / len(lm)
                    jitters.append(disp / iod)
                prev_lm = lm
            else:
                prev_lm = None   # don't bridge jitter across failed frames
    if not confs:
        return None
    sc = sorted(confs)
    return {
        "n_frames": len(confs),
        "conf_mean": sum(confs) / len(confs),
        "conf_p10": percentile(sc, 10),
        "conf_min": sc[0],
        "success_rate": sum(successes) / len(successes),
        "jitter_mean": (sum(jitters) / len(jitters)) if jitters else float("nan"),
    }


def run_aggregation(clips, per_frame_dir: Path, out_dir: Path):
    rows = []
    miss = 0
    for c in clips:
        csv_path = per_frame_dir / f"{c['stem']}.csv"
        if not output_ok(csv_path):
            miss += 1
            continue
        agg = aggregate_clip(csv_path)
        if agg is None:
            miss += 1
            continue
        rows.append({"file": c["stem"], "actor_id": c["actor_id"], "age": c["age"],
                     "sex": c["sex"], "emotion": c["emotion"], **agg})
    if miss:
        print(f"[aggregate] {miss} clips without usable output (not yet extracted or failed)")
    if not rows:
        sys.exit("[error] no per-frame CSVs to aggregate; run extraction first.")
    out = out_dir / "per_clip.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[aggregate] wrote {len(rows)} rows -> {out}")
    return rows


# ----------------------------------------------------------------------------
# per-actor regression + verdict (the H1 result)
# ----------------------------------------------------------------------------
def run_regression(rows, out_dir: Path):
    ensure("scipy")
    from scipy import stats

    by_actor = {}
    for r in rows:
        a = by_actor.setdefault(r["actor_id"], {"age": r["age"], "sex": r["sex"],
                                                **{s: [] for s in SIGNALS}})
        for s in SIGNALS:
            if not math.isnan(r[s]):
                a[s].append(r[s])
    per_actor = []
    for actor_id, v in by_actor.items():
        row = {"actor_id": actor_id, "age": v["age"], "sex": v["sex"],
               "n_clips": len(v["conf_mean"])}
        for s in SIGNALS:
            row[s] = sum(v[s]) / len(v[s]) if v[s] else float("nan")
        per_actor.append(row)
    per_actor.sort(key=lambda x: x["age"])

    with open(out_dir / "per_actor.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_actor[0].keys()))
        w.writeheader()
        w.writerows(per_actor)

    if len(per_actor) < 3:
        print(f"[regress] only {len(per_actor)} actors -- too few; this was a smoke "
              "test. The H1 verdict needs the full run.")
        return

    ages = [a["age"] for a in per_actor]
    lines = ["Experiment 3: OpenFace reliability vs actor age (H1, visual primary form)",
             "=" * 74,
             f"actors: {len(per_actor)}   age range: {min(ages)}-{max(ages)}   "
             f"clips: {sum(a['n_clips'] for a in per_actor)}", ""]
    confirming = []
    for s, direction in SIGNALS.items():
        pairs = [(a["age"], a[s]) for a in per_actor if not math.isnan(a[s])]
        if len(pairs) < 3:
            lines.append(f"{s:14s}  insufficient data")
            continue
        xs, ys = zip(*pairs)
        res = stats.linregress(xs, ys)
        ci = stats.t.ppf(0.975, len(xs) - 2) * res.stderr
        sig = res.pvalue < 0.05 and (
            (direction == "negative" and res.slope < 0) or
            (direction == "positive" and res.slope > 0))
        if sig:
            confirming.append(s)
        lines.append(f"{s:14s}  slope/decade={res.slope*10:+.5f}  "
                     f"95%CI/yr=[{res.slope-ci:+.6f},{res.slope+ci:+.6f}]  "
                     f"r={res.rvalue:+.3f}  p={res.pvalue:.5f}  "
                     f"H1-direction={direction}  {'CONFIRMS H1' if sig else 'flat/ns'}")
        plot_signal(xs, ys, res, s, out_dir)

    lines.append("")
    lines.append("VERDICT")
    lines.append("-------")
    if "conf_mean" in confirming or "conf_p10" in confirming or "conf_min" in confirming:
        lines.append("H1 CONFIRMED on the primary signal: detection confidence declines "
                     "with age. The visual gate uses detection confidence.")
    elif confirming:
        lines.append(f"H1 holds via fallback signal(s): {', '.join(confirming)}. "
                     "Per the proposal, the visual gate reroutes to the signal "
                     "carrying the age trend (detection confidence is age-flat).")
    else:
        lines.append("No signal shows a significant age trend on CREMA-D. The visual "
                     "gate premise is not demonstrable here; escalate to ElderReact "
                     "and report H1 (visual) as not supported on clean studio data.")
    report = "\n".join(lines) + "\n"
    (out_dir / "regression.txt").write_text(report)
    print("\n" + report)


def plot_signal(xs, ys, res, name, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        plt.scatter(xs, ys, alpha=0.7, label="per-actor mean")
        xr = [min(xs), max(xs)]
        plt.plot(xr, [res.intercept + res.slope * x for x in xr], "r-",
                 label=f"slope/decade={res.slope*10:+.4f}, p={res.pvalue:.3f}")
        plt.xlabel("Actor age (years)")
        plt.ylabel(name)
        plt.title(f"CREMA-D: {name} vs actor age (Experiment 3)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_vs_age.png", dpi=130)
        plt.close()
    except Exception as e:
        print(f"[plot] {name} skipped ({e})")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Experiment 3: OpenFace reliability vs age (H1).")
    ap.add_argument("--data", default="./data/crema_d", help="CREMA-D dir (with metadata.csv)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only N clips")
    ap.add_argument("--full", action="store_true", help="extract ALL clips")
    ap.add_argument("--analyze", action="store_true",
                    help="skip extraction; aggregate + regress existing output")
    ap.add_argument("--workers", type=int, default=8, help="parallel OpenFace processes")
    args = ap.parse_args()

    data_dir = Path(args.data).expanduser().resolve()
    out_dir = data_dir / "experiment3_visual"
    per_frame_dir = out_dir / "per_frame"
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = load_clips(data_dir)
    print(f"[data] {len(clips)} clips listed in metadata")

    if not args.analyze:
        work = clips if args.full else clips[:args.limit or 2]
        if not args.full:
            print(f"[mode] smoke test: {len(work)} clips (use --full for the real run)")
        failures = run_extraction(work, per_frame_dir, args.workers)
        if args.full and failures:
            print(f"[note] {len(failures)} clips failed; re-running the script retries "
                  "only those (resumable).")

    rows = run_aggregation(clips, per_frame_dir, out_dir)
    run_regression(rows, out_dir)


if __name__ == "__main__":
    main()
