#!/usr/bin/env python3
"""
wer_vs_age.py  --  Experiment 3: does Whisper's word-error-rate rise with speaker age?
======================================================================================

This is the Option A / Option B decision gate for the ASR-aware text branch
(Contribution 2). It transcribes CREMA-D clips with Whisper large-v3, computes
word error rate (WER) against the known ground-truth sentence for each clip, and
regresses per-clip WER on the actor's age across all 91 actors.

  - A clear positive slope (WER rises with age)  => Option A is viable
    (CREMA-D can anchor the ASR-error premise).
  - A flat / negligible slope                    => Option B
    (move the WER-vs-age analysis to a harder elderly corpus; CREMA-D's clean
     scripted speech doesn't stress the ASR enough).

Implementation choices (made deliberately):
  * faster-whisper  : fast on Apple Silicon, exposes per-word confidence (reused
                      later for Contribution 2's confidence mask).
  * large-v3        : the proposal's committed model. Using the BEST ASR means a
                      flat slope genuinely implies "even strong ASR copes here",
                      rather than "a weak model struggled" (which would confound age).

Usage:
    # quick stratified subset (default, ~50 clips across the age range)
    python wer_vs_age.py --data ./data/crema_d --subset 50

    # full run (all 7,442 clips)
    python wer_vs_age.py --data ./data/crema_d --full

Outputs (written to <data>/experiment3/):
    wer_per_clip.csv     one row per processed clip: file, actor, age, wer, conf, hyp, ref
    wer_by_actor.csv     mean WER + mean age per actor
    regression.txt       slope, intercept, r, p-value, 95% CI, verdict
    wer_vs_age.png       scatter of per-actor WER vs age with fitted line
"""

import argparse
import csv
import os
import sys
import subprocess
import re
from pathlib import Path


# ----------------------------------------------------------------------------
# dependency bootstrap
# ----------------------------------------------------------------------------
def ensure(pkg, pip_name=None, import_name=None):
    import_name = import_name or pkg
    try:
        return __import__(import_name)
    except ImportError:
        print(f"[setup] installing {pip_name or pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pip_name or pkg],
                       check=True)
        return __import__(import_name)


# ----------------------------------------------------------------------------
# text normalisation + WER (no external jiwer dependency required)
# ----------------------------------------------------------------------------
def normalise(text: str) -> list:
    """Lowercase, strip punctuation AND apostrophes, collapse whitespace -> word list.
    Apostrophes are removed from BOTH sides so 'don't' vs 'dont' or 'o'clock' vs
    'oclock' are treated as the same word (formatting, not a real ASR error).
    ASR and reference are normalised identically so WER reflects genuine errors."""
    text = text.lower()
    text = text.replace("'", "")                # don't -> dont, o'clock -> oclock
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return text.split()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate via Levenshtein edit distance on word lists.
    WER = (substitutions + deletions + insertions) / number_of_reference_words."""
    r = normalise(reference)
    h = normalise(hypothesis)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    # DP edit-distance matrix
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,        # deletion
                          d[i][j - 1] + 1,        # insertion
                          d[i - 1][j - 1] + cost) # substitution/match
    return d[len(r)][len(h)] / len(r)


# ----------------------------------------------------------------------------
# load metadata + (optionally) stratified subsetting
# ----------------------------------------------------------------------------
def load_rows(data_dir: Path):
    meta = data_dir / "metadata.csv"
    if not meta.exists():
        sys.exit(f"[error] {meta} not found. Run download_cremad.py first.")
    rows = []
    with open(meta, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("transcript"):    # skip any clip with no reference
                continue
            try:
                row["age"] = int(row["age"])
            except (ValueError, KeyError):
                continue
            rows.append(row)
    return rows


def stratified_subset(rows, n):
    """Pick ~n clips spread across the age range so even a small run hints at the trend.
    Buckets actors by age band, then samples clips proportionally."""
    import random
    random.seed(42)
    bands = {}
    for r in rows:
        b = r["age"] // 10 * 10           # 20s, 30s, 40s, ...
        bands.setdefault(b, []).append(r)
    per_band = max(1, n // max(1, len(bands)))
    picked = []
    for b in sorted(bands):
        clips = bands[b]
        random.shuffle(clips)
        picked.extend(clips[:per_band])
    random.shuffle(picked)
    return picked[:n]


# ----------------------------------------------------------------------------
# transcription
# ----------------------------------------------------------------------------
def build_model(model_size: str):
    ensure("faster_whisper", "faster-whisper")
    from faster_whisper import WhisperModel
    # int8 keeps memory low and runs well on the M4 CPU/Metal path
    print(f"[model] loading faster-whisper '{model_size}' (first run downloads weights) ...")
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe(model, wav_path: Path):
    """Return (hypothesis_text, mean_confidence). Confidence from avg word probability;
    falls back to exp(avg_logprob) per segment when word probs are unavailable."""
    import math
    segments, _ = model.transcribe(str(wav_path), language="en", word_timestamps=True)
    words_text, probs = [], []
    seg_logprobs = []
    for seg in segments:
        seg_logprobs.append(seg.avg_logprob)
        if seg.words:
            for w in seg.words:
                words_text.append(w.word)
                if w.probability is not None:
                    probs.append(w.probability)
        else:
            words_text.append(seg.text)
    hyp = "".join(words_text).strip()
    if probs:
        conf = sum(probs) / len(probs)
    elif seg_logprobs:
        conf = math.exp(sum(seg_logprobs) / len(seg_logprobs))
    else:
        conf = float("nan")
    return hyp, conf


# ----------------------------------------------------------------------------
# regression + plot
# ----------------------------------------------------------------------------
def regress_and_report(per_actor, out_dir: Path):
    ensure("scipy", "scipy")
    from scipy import stats
    ages = [a["age"] for a in per_actor]
    wers = [a["mean_wer"] for a in per_actor]

    res = stats.linregress(ages, wers)
    slope, intercept, r, p = res.slope, res.intercept, res.rvalue, res.pvalue
    # 95% CI on slope
    tcrit = stats.t.ppf(0.975, len(ages) - 2)
    ci = tcrit * res.stderr

    # verdict logic: is the slope both positive and statistically credible?
    wer_rise_per_decade = slope * 10 * 100   # percentage points per 10 years
    if p < 0.05 and slope > 0:
        verdict = ("OPTION A VIABLE: WER rises significantly with age. "
                   f"~{wer_rise_per_decade:.1f} WER percentage points per decade "
                   "(p < 0.05). CREMA-D can anchor the ASR-error premise (H1).")
    elif slope > 0:
        verdict = ("INCONCLUSIVE / LEAN OPTION B: slope is positive but not "
                   f"significant (p = {p:.3f}). Effect ~{wer_rise_per_decade:.1f} pts/decade. "
                   "A full run may sharpen this; if it stays weak, move H1 to a harder "
                   "elderly corpus.")
    else:
        verdict = ("OPTION B: no positive WER-vs-age slope on CREMA-D "
                   f"(slope {slope:.5f}, p = {p:.3f}). The clean scripted clips don't "
                   "stress the ASR. Move the WER-vs-age analysis to an elderly corpus.")

    report = out_dir / "regression.txt"
    with open(report, "w") as f:
        f.write("Experiment 3: WER vs Age (per-actor regression)\n")
        f.write("=" * 55 + "\n")
        f.write(f"actors analysed : {len(ages)}\n")
        f.write(f"age range       : {min(ages)}-{max(ages)}\n")
        f.write(f"mean WER        : {sum(wers)/len(wers):.4f}\n\n")
        f.write(f"slope           : {slope:.6f} WER per year\n")
        f.write(f"                  ({wer_rise_per_decade:.2f} WER percentage points per decade)\n")
        f.write(f"95% CI on slope : [{slope-ci:.6f}, {slope+ci:.6f}]\n")
        f.write(f"intercept       : {intercept:.4f}\n")
        f.write(f"Pearson r       : {r:.4f}\n")
        f.write(f"p-value         : {p:.5f}\n\n")
        f.write("VERDICT\n-------\n" + verdict + "\n")
    print("\n" + open(report).read())

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [min(ages), max(ages)]
        ys = [intercept + slope * x for x in xs]
        plt.figure(figsize=(7, 5))
        plt.scatter(ages, wers, alpha=0.7, label="per-actor mean WER")
        plt.plot(xs, ys, "r-", label=f"fit: slope={slope:.4f}, p={p:.3f}")
        plt.xlabel("Actor age (years)")
        plt.ylabel("Mean WER")
        plt.title("CREMA-D: Whisper WER vs actor age (Experiment 3)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "wer_vs_age.png", dpi=130)
        print(f"[plot] saved {out_dir/'wer_vs_age.png'}")
    except Exception as e:
        print(f"[plot] skipped ({e})")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Experiment 3: Whisper WER vs actor age on CREMA-D.")
    ap.add_argument("--data", default="./data/crema_d", help="CREMA-D dir (with metadata.csv)")
    ap.add_argument("--model", default="large-v3", help="faster-whisper model size")
    ap.add_argument("--subset", type=int, default=50, help="number of clips for a quick run")
    ap.add_argument("--full", action="store_true", help="process ALL clips (overrides --subset)")
    args = ap.parse_args()

    data_dir = Path(args.data).expanduser().resolve()
    out_dir = data_dir / "experiment3"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(data_dir)
    print(f"[data] {len(rows)} clips with transcripts available")

    if args.full:
        work = rows
        print(f"[mode] FULL run: {len(work)} clips (this will take a while on large-v3)")
    else:
        work = stratified_subset(rows, args.subset)
        print(f"[mode] SUBSET run: {len(work)} clips, age-stratified")

    model = build_model(args.model)

    # transcribe each clip
    per_clip_path = out_dir / "wer_per_clip.csv"
    results = []
    with open(per_clip_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "actor_id", "age", "wer", "confidence", "reference", "hypothesis"])
        for i, r in enumerate(work, 1):
            wav = data_dir / r["file"]
            if not wav.exists():
                # metadata stores path relative to data_dir; AudioWAV is a symlink
                wav = data_dir / "AudioWAV" / Path(r["file"]).name
            try:
                hyp, conf = transcribe(model, wav)
            except Exception as e:
                print(f"  [skip] {wav.name}: {e}")
                continue
            e = wer(r["transcript"], hyp)
            results.append({"actor_id": r["actor_id"], "age": r["age"], "wer": e})
            w.writerow([r["file"], r["actor_id"], r["age"], f"{e:.4f}",
                        f"{conf:.4f}", r["transcript"], hyp])
            if i % 10 == 0 or i == len(work):
                print(f"  [{i}/{len(work)}] {wav.name}  WER={e:.3f}  conf={conf:.3f}")

    if not results:
        sys.exit("[error] no clips processed; check audio paths.")

    # aggregate per actor (so each actor counts once in the regression)
    by_actor = {}
    for r in results:
        by_actor.setdefault(r["actor_id"], {"age": r["age"], "wers": []})
        by_actor[r["actor_id"]]["wers"].append(r["wer"])
    per_actor = [{"actor_id": a, "age": v["age"],
                  "mean_wer": sum(v["wers"]) / len(v["wers"]), "n": len(v["wers"])}
                 for a, v in by_actor.items()]

    with open(out_dir / "wer_by_actor.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["actor_id", "age", "mean_wer", "n"])
        w.writeheader()
        w.writerows(sorted(per_actor, key=lambda x: x["age"]))

    print(f"\n[clips] processed {len(results)} clips across {len(per_actor)} actors")
    if len(per_actor) < 3:
        print("[warn] too few actors for a meaningful regression on this subset; "
              "use --full or a larger --subset for the real result.")
        return
    regress_and_report(per_actor, out_dir)

    print("\nReminder: the SUBSET run is a smoke test. The decision-gate result")
    print("for Option A vs B should come from the --full run over all 91 actors.")


if __name__ == "__main__":
    main()
