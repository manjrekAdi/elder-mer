#!/usr/bin/env python3
"""
audio_reliability.py -- audio-channel analog of experiment3_visual_gate / elderreact_visual_reliability
========================================================================================================

The confidence-gated projector (Contribution 2) gates the audio channel by a
"voice-activity / signal-quality estimate". Before committing the audio gate to
CREMA-D (task 4.2), we need the same check Experiment 3 ran for the visual
channel: does CREMA-D's studio audio SATURATE at high quality (no low-quality
tail for the gate to act on), the way OpenFace confidence saturated at ~0.98?
And does ElderReact's in-the-wild audio have a fat low-quality tail?

Per-clip signal-quality estimates (all cheap, transparent proxies; the gate would
use one of these or a learned VAD, but for the *spread* question any sensible
estimator separates studio speech from in-the-wild YouTube audio):

  1. snr_db        energy-based SNR: 10log10(mean speech-frame power / mean
                   noise-frame power). Background music/noise raises the noise
                   floor and collapses this -> low SNR. Primary signal.
  2. speech_ratio  fraction of frames above the noise floor (energy VAD).
                   Music-only / silent clips -> low ratio.
  3. flatness      mean spectral flatness over speech frames. Voiced speech is
                   tonal (low flatness); noise/music is flat (high). Secondary.

Reads CREMA-D from AudioWAV/ directly; extracts ElderReact audio from the mp4s
with ffmpeg (16 kHz mono), same as probe_speaker_clustering.py.

Usage:
    python scripts/audio_reliability.py --limit 20        # smoke test, both sets
    python scripts/audio_reliability.py --full            # all clips, both sets
"""
import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

SR = 16000
FRAME = int(0.025 * SR)   # 25 ms
HOP = int(0.010 * SR)     # 10 ms
VAD_DB_ABOVE_FLOOR = 6.0  # a frame is "speech" if >6 dB above the noise floor


def load_mono_16k(wav_path: Path) -> np.ndarray:
    y, sr = sf.read(str(wav_path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def signals(y: np.ndarray):
    """Per-clip reliability signals from a 16 kHz mono waveform."""
    if y.size < FRAME:
        return None
    # frame -> linear power per frame
    frames = librosa.util.frame(y, frame_length=FRAME, hop_length=HOP)  # (FRAME, nframes)
    power = (frames ** 2).mean(axis=0) + 1e-12
    log_e = 10.0 * np.log10(power)
    # noise floor = 10th-percentile frame energy; speech = >6 dB above it
    floor = np.percentile(log_e, 10)
    speech = log_e > floor + VAD_DB_ABOVE_FLOOR
    speech_ratio = float(speech.mean())
    if speech.sum() >= 1 and (~speech).sum() >= 1:
        snr_db = 10.0 * math.log10(power[speech].mean() / power[~speech].mean())
    else:
        # all frames same side of the threshold: degenerate, cap it
        snr_db = float("nan")
    # spectral flatness over speech frames (tonal speech low, music/noise high)
    flat = librosa.feature.spectral_flatness(y=y.astype(np.float32),
                                             n_fft=FRAME, hop_length=HOP)[0]
    m = min(len(flat), len(speech))
    sp = speech[:m]
    flat_speech = float(flat[:m][sp].mean()) if sp.any() else float(flat[:m].mean())
    return {"snr_db": snr_db, "speech_ratio": speech_ratio, "flatness": flat_speech}


def crema_clips(limit):
    wavs = sorted((Path("data/crema_d/AudioWAV")).glob("*.wav"))
    return [{"file": w.name, "wav": w} for w in (wavs[:limit] if limit else wavs)]


def elderreact_clips(limit):
    out = []
    for s in ("train", "dev", "test"):
        for v in sorted((Path(f"data/elderreact/ElderReact_{s}")).glob("*.mp4")):
            out.append({"file": v.name, "video": v})
    return out[:limit] if limit else out


def process_crema(c, _tmp):
    return signals(load_mono_16k(c["wav"]))


def process_elder(c, tmp):
    wav = Path(tmp) / (c["file"] + ".wav")
    try:
        subprocess.run(["ffmpeg", "-y", "-threads", "1", "-i", str(c["video"]),
                        "-ac", "1", "-ar", str(SR), "-vn", str(wav)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return signals(load_mono_16k(wav))
    finally:
        wav.unlink(missing_ok=True)


def run_dataset(name, clips, proc, out_dir, workers):
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp()
    rows, fails, done = [], 0, 0
    print(f"[{name}] {len(clips)} clips, {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(proc, c, tmp): c for c in clips}
        for fut in as_completed(futs):
            c = futs[fut]
            done += 1
            try:
                s = fut.result()
                if s and not any(math.isnan(v) for v in s.values()):
                    rows.append({"file": c["file"], **s})
                else:
                    fails += 1
            except Exception:
                fails += 1
            if done % 500 == 0 or done == len(clips):
                print(f"  [{done}/{len(clips)}] ok={len(rows)} skipped={fails}")
    with open(out_dir / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "snr_db", "speech_ratio", "flatness"])
        w.writeheader()
        w.writerows(rows)
    print(f"[{name}] wrote {len(rows)} rows ({fails} skipped) -> {out_dir/'per_clip.csv'}")
    return rows


def stats(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return v


def fmt_block(name, rows):
    snr = stats(rows, "snr_db")
    spr = stats(rows, "speech_ratio")
    flt = stats(rows, "flatness")
    L = [f"{name}: {len(rows)} clips",
         f"  snr_db        : mean={snr.mean():.2f}  median={np.median(snr):.2f}  "
         f"p10={np.percentile(snr,10):.2f}  min={snr.min():.2f}",
         f"  snr_db<20     : {100*np.mean(snr<20):.1f}%   <15: {100*np.mean(snr<15):.1f}%   "
         f"<10: {100*np.mean(snr<10):.1f}%",
         f"  speech_ratio  : mean={spr.mean():.3f}  p10={np.percentile(spr,10):.3f}  "
         f"min={spr.min():.3f}   (<0.5: {100*np.mean(spr<0.5):.1f}%)",
         f"  flatness      : mean={flt.mean():.4f}  p90={np.percentile(flt,90):.4f}   "
         f"(>0.10: {100*np.mean(flt>0.10):.1f}%)"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    limit = 0 if args.full else (args.limit or 20)

    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    cre = run_dataset("CREMA-D", crema_clips(limit), process_crema,
                      Path("data/crema_d/audio_reliability"), args.workers)
    eld = run_dataset("ElderReact", elderreact_clips(limit), process_elder,
                      Path("data/elderreact/audio_reliability"), args.workers)

    report = ["Audio signal-quality spread: ElderReact (in-the-wild) vs CREMA-D (studio)",
              "=" * 74, "",
              fmt_block("ElderReact", eld), "",
              fmt_block("CREMA-D (reference)", cre), "",
              "READING",
              "-------",
              "If CREMA-D saturates at high SNR / high speech-ratio with no low-quality",
              "tail (like OpenFace conf at ~0.98), the AUDIO gate -- like the visual gate",
              "-- has nothing to act on there, and its 'helps' demonstration must move to",
              "ElderReact. A fat low-quality tail on ElderReact confirms signal to gate.", ""]
    text = "\n".join(report) + "\n"
    Path("data/elderreact/audio_reliability/summary.txt").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
