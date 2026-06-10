#!/usr/bin/env python3
"""
download_cremad.py
==================
Downloads the CREMA-D dataset (audio + demographics) for the ASR-vs-age
experiment (Experiment 3). Designed to run on an M4 MacBook.

Two methods are provided:

  METHOD 1 (recommended): Hugging Face `datasets`. One call, no Git LFS,
           audio arrives already decoded at 16 kHz. Best for the WER experiment.

  METHOD 2 (fallback): clone the official GitHub repo's AudioWAV/ folder
           via git + Git LFS. Use only if Hugging Face is unavailable.

Usage:
    python download_cremad.py --method hf   --out ./crema_d      # default
    python download_cremad.py --method git  --out ./crema_d

After download you'll have:
    <out>/AudioWAV/<sample>.wav        (7,442 clips, 16 kHz mono)
    <out>/VideoDemographics.csv       (per-actor age, sex, race, ethnicity)
    <out>/metadata.csv                (one row per clip: file, actor, age, sentence, emotion)

The 12 CREMA-D sentences and their codes are baked in below, so you have the
exact ground-truth transcripts needed to compute WER without any extra lookup.
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# CREMA-D reference data: the 12 sentence codes -> exact ground-truth transcript
# (these are the canonical sentences every actor read; they ARE your WER reference)
# ----------------------------------------------------------------------------
SENTENCES = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

EMOTIONS = {
    "ANG": "anger", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}

DEMO_URL = "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/VideoDemographics.csv"


def run(cmd, **kw):
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kw)


def need(pkg, pip_name=None):
    """Import a package, installing it on first use."""
    pip_name = pip_name or pkg
    try:
        return __import__(pkg)
    except ImportError:
        print(f"[setup] installing {pip_name} ...")
        run([sys.executable, "-m", "pip", "install", "--quiet", pip_name])
        return __import__(pkg)


def fetch_demographics(out: Path) -> dict:
    """Download VideoDemographics.csv and return {actor_id: age}."""
    import urllib.request
    dest = out / "VideoDemographics.csv"
    if not dest.exists():
        print("[demographics] downloading VideoDemographics.csv ...")
        urllib.request.urlretrieve(DEMO_URL, dest)
    age_by_actor = {}
    with open(dest, newline="") as f:
        for row in csv.DictReader(f):
            age_by_actor[str(row["ActorID"]).strip()] = int(row["Age"])
    print(f"[demographics] {len(age_by_actor)} actors, "
          f"ages {min(age_by_actor.values())}-{max(age_by_actor.values())}")
    return age_by_actor


def parse_filename(fname: str):
    """
    CREMA-D filename: 1001_DFA_ANG_XX.wav
      -> actor=1001, sentence_code=DFA, emotion_code=ANG, intensity=XX
    Returns (actor, sentence_code, emotion_code, intensity) or None if malformed.
    """
    stem = Path(fname).stem
    parts = stem.split("_")
    if len(parts) != 4:
        return None
    actor, scode, ecode, intensity = parts
    return actor, scode, ecode, intensity


def write_metadata(out: Path, wav_dir: Path, age_by_actor: dict):
    """Build metadata.csv: one row per clip with everything the WER step needs."""
    rows = []
    for wav in sorted(wav_dir.glob("*.wav")):
        parsed = parse_filename(wav.name)
        if not parsed:
            print(f"  [skip] unrecognised filename: {wav.name}")
            continue
        actor, scode, ecode, intensity = parsed
        rows.append({
            "file": str(wav.relative_to(out)),
            "actor_id": actor,
            "age": age_by_actor.get(actor, ""),
            "sentence_code": scode,
            "transcript": SENTENCES.get(scode, ""),
            "emotion_code": ecode,
            "emotion": EMOTIONS.get(ecode, ""),
            "intensity": intensity,
        })
    meta = out / "metadata.csv"
    with open(meta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[metadata] wrote {len(rows)} rows -> {meta}")
    return len(rows)


# ----------------------------------------------------------------------------
# METHOD 1: Hugging Face
# ----------------------------------------------------------------------------
def download_hf(out: Path):
    """
    Pulls CREMA-D via the `datasets` library and writes 16 kHz WAVs to AudioWAV/.
    The myleslinder/crema-d mirror exposes actor_id, sentence, emotion, audio array.
    """
    datasets = need("datasets")
    sf = need("soundfile")
    import numpy as np  # soundfile pulls numpy

    wav_dir = out / "AudioWAV"
    wav_dir.mkdir(parents=True, exist_ok=True)

    print("[hf] loading myleslinder/crema-d (this downloads ~1-2 GB on first run) ...")
    ds = datasets.load_dataset("myleslinder/crema-d", split="train")

    # Reverse-map the sentence text back to its code so filenames match the
    # canonical CREMA-D scheme (actor_sentence_emotion_intensity.wav).
    text_to_code = {v.lower(): k for k, v in SENTENCES.items()}
    emo_label_names = ds.features["label"].names if "label" in ds.features else None

    n = 0
    for ex in ds:
        actor = str(ex.get("actor_id", "")).strip()
        sent_text = str(ex.get("sentence", "")).strip().lower()
        scode = text_to_code.get(sent_text, "UNK")
        # emotion: prefer explicit label name, fall back to NEU
        ecode = "NEU"
        if emo_label_names is not None and "label" in ex:
            name = emo_label_names[ex["label"]].upper()[:3]
            ecode = name if name in EMOTIONS else ecode
        intensity = str(ex.get("emotion_intensity", "XX"))[:2].upper() or "XX"

        audio = ex["audio"]
        arr, srate = audio["array"], audio["sampling_rate"]
        fname = f"{actor}_{scode}_{ecode}_{intensity}.wav"
        sf.write(wav_dir / fname, np.asarray(arr), srate)
        n += 1
        if n % 500 == 0:
            print(f"  ... wrote {n} clips")
    print(f"[hf] wrote {n} WAV files -> {wav_dir}")
    return wav_dir


# ----------------------------------------------------------------------------
# METHOD 2: GitHub + Git LFS (audio only)
# ----------------------------------------------------------------------------
def download_git(out: Path):
    """
    Sparse-clones only the AudioWAV/ directory from the official repo using
    Git LFS. Heavier and slower than the HF route but fully official.
    """
    out.mkdir(parents=True, exist_ok=True)
    repo = out / "_cremad_repo"

    # Ensure git-lfs exists
    try:
        run(["git", "lfs", "version"], stdout=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[git] Git LFS not found. Install it first:")
        print("       macOS:  brew install git-lfs && git lfs install")
        sys.exit(1)

    if not repo.exists():
        print("[git] sparse-cloning AudioWAV/ from CheyneyComputerScience/CREMA-D ...")
        run(["git", "clone", "--filter=blob:none", "--no-checkout",
             "https://github.com/CheyneyComputerScience/CREMA-D.git", str(repo)])
        run(["git", "-C", str(repo), "sparse-checkout", "init", "--cone"])
        run(["git", "-C", str(repo), "sparse-checkout", "set", "AudioWAV"])
        run(["git", "-C", str(repo), "checkout", "master"])
        run(["git", "-C", str(repo), "lfs", "pull", "--include", "AudioWAV/*"])

    # Symlink the AudioWAV folder up to the expected location
    src = repo / "AudioWAV"
    dst = out / "AudioWAV"
    if not dst.exists():
        os.symlink(src.resolve(), dst)
    print(f"[git] AudioWAV ready -> {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser(description="Download CREMA-D (audio + demographics).")
    ap.add_argument("--method", choices=["hf", "git"], default="hf",
                    help="hf = Hugging Face (recommended); git = official repo via LFS")
    ap.add_argument("--out", default="./crema_d", help="output directory")
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[init] output directory: {out}\n")

    age_by_actor = fetch_demographics(out)

    if args.method == "hf":
        wav_dir = download_hf(out)
    else:
        wav_dir = download_git(out)

    n = write_metadata(out, wav_dir, age_by_actor)

    print("\n" + "=" * 60)
    print("DONE. Summary:")
    print(f"  audio dir   : {wav_dir}")
    print(f"  clips       : {n}")
    print(f"  demographics: {out/'VideoDemographics.csv'}")
    print(f"  metadata    : {out/'metadata.csv'}")
    print("=" * 60)
    print("\nNext step: run the WER-vs-age experiment on metadata.csv.")
    print("Each row already has the exact ground-truth `transcript`, so you can")
    print("transcribe each `file` with Whisper and compute WER against it.")


if __name__ == "__main__":
    main()
