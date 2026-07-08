#!/usr/bin/env python3
"""
extract_elderreact_features.py -- ElderReact features for the gate (Exp 4),
per-person calibration (Exp 5), and transfer (Exp 6).
==========================================================================

Mirrors extract_features.py (CREMA-D) but for ElderReact, reusing the same frozen
HuBERT audio encoder and OpenFace visual aggregation:
  - audio  : extract audio from the mp4 with ffmpeg -> frozen HuBERT -> [T, 1024]
  - visual : aggregate the OpenFace per-frame CSVs (already extracted in Exp 2.3b)
             -> [T, 17] AU intensities + per-frame confidence/success (the gate input)

ElderReact differs from CREMA-D and those differences matter downstream:
  - MULTI-LABEL: 6 binary emotions per clip (Anger, Disgust, Fear, Happiness,
    Sadness, Surprise); a clip can have several. (CREMA-D was single-label 6-class.)
  - Gender only, NO per-person ages -> no age regression here (the age story is
    CREMA-D's; ElderReact tests transfer + subgroup fairness by gender).
  - In-the-wild spontaneous video -> OpenFace confidence has a fat low-confidence
    tail (35% of clips < 0.95 per Exp 2.3b), which is exactly where the confidence
    gate has signal to act on. This is why ElderReact is the gate's real testbed.

Text (Whisper transcription of the spontaneous speech) is handled separately (the
ASR-text gate for Exp 6).

OUTPUT:
    data/elderreact/features/audio/<stem>.pt   {"feats":[T,1024], "model":..., "sr":16000}
    data/elderreact/features/visual/<stem>.pt  {"feats":[T,17], "conf":[T], "success":[T], ...}
    data/elderreact/features/index.csv          stem, split, audio_path, visual_path,
                                                anger, disgust, fear, happiness, sadness,
                                                surprise, gender, valence, n_audio, n_visual

RUN (in the `elder-mer` conda env):
    python scripts/extract_elderreact_features.py --limit 8     # smoke test
    python scripts/extract_elderreact_features.py               # all 1,323 (resumable)
"""
import argparse
import csv
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# reuse the CREMA-D feature machinery (frozen HuBERT + OpenFace aggregation)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import AudioEncoder, extract_visual_openface

# ElderReact multi-label emotion order (from the dataset README's label format)
EMOTIONS6 = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
SPLITS = ("train", "dev", "test")


def load_clips(data_root: Path, limit):
    """Parse the annotation files -> per-clip records with the 6 binary labels,
    gender, valence, and split. Label line format:
    filename Anger Disgust Fear Happiness Sadness Surprise Gender Valence"""
    clips = []
    for split in SPLITS:
        ann = data_root / "_elderreact_repo" / "Annotations" / f"{split}_labels.txt"
        vids = data_root / f"ElderReact_{split}"
        if not ann.exists():
            sys.exit(f"[error] {ann} not found.")
        for line in ann.read_text().splitlines():
            p = line.split()
            if len(p) < 9:
                continue
            fname = p[0]
            clips.append({
                "stem": Path(fname).stem, "mp4": vids / fname, "split": split,
                "labels": [int(x) for x in p[1:7]],
                "gender": p[7], "valence": float(p[8]),
            })
    clips.sort(key=lambda c: c["stem"])
    return clips[:limit] if limit else clips


def extract_audio_from_mp4(mp4: Path, tmp: str) -> Path:
    """ffmpeg: mp4 -> 16 kHz mono wav (what HuBERT/load_16k_mono expects)."""
    wav = Path(tmp) / (mp4.stem + ".wav")
    subprocess.run(["ffmpeg", "-y", "-threads", "1", "-i", str(mp4),
                    "-ac", "1", "-ar", "16000", "-vn", str(wav)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav


def already_done(path: Path):
    return path.exists() and path.stat().st_size > 0


def run(args):
    import torch

    data_root = Path(args.data_root)
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "visual").mkdir(parents=True, exist_ok=True)
    per_frame_dir = data_root / "visual_reliability" / "per_frame"

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    store_dtype = torch.float16 if args.fp16 else torch.float32

    clips = load_clips(data_root, args.limit)
    print(f"[ElderReact] {len(clips)} clips | device={device} | dtype={'fp16' if args.fp16 else 'fp32'}")

    # ---- audio: ffmpeg from mp4 -> frozen HuBERT (GPU-bound: sequential) ----
    if not args.skip_audio:
        enc = AudioEncoder(args.audio_model, device, store_dtype)
        tmp = tempfile.mkdtemp()
        done = skipped = failed = 0
        for i, c in enumerate(clips, 1):
            dst = out / "audio" / f"{c['stem']}.pt"
            if already_done(dst) and not args.overwrite:
                skipped += 1
            elif not c["mp4"].exists():
                failed += 1
                print(f"  [audio][miss] {c['mp4'].name}")
            else:
                try:
                    wav = extract_audio_from_mp4(c["mp4"], tmp)
                    feats = enc.encode(wav)
                    torch.save({"feats": feats, "model": args.audio_model, "sr": 16000}, dst)
                    wav.unlink(missing_ok=True)
                    c["n_audio"] = int(feats.shape[0])
                    done += 1
                except Exception as e:
                    failed += 1
                    print(f"  [audio][err] {c['stem']}: {str(e)[:80]}")
            if i % 100 == 0 or i == len(clips):
                print(f"  [audio] {i}/{len(clips)}  new={done} skip={skipped} fail={failed}")
    else:
        print("[audio] skipped")

    # ---- visual: aggregate OpenFace per-frame CSVs (threaded, CPU/IO) ----
    if not args.skip_visual:
        def do_one(c):
            dst = out / "visual" / f"{c['stem']}.pt"
            if already_done(dst) and not args.overwrite:
                return ("skip", c, None)
            res, err = extract_visual_openface(c["stem"], per_frame_dir, args.visual_cols)
            if err:
                return ("fail", c, err)
            torch.save({"feats": torch.from_numpy(res["feats"]).to(store_dtype),
                        "conf": torch.from_numpy(res["conf"]),
                        "success": torch.from_numpy(res["success"]),
                        "cols": res["cols"], "source": res["source"]}, dst)
            c["n_visual"] = int(res["feats"].shape[0])
            return ("done", c, None)

        done = skipped = failed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(do_one, c): c for c in clips}
            for n, fut in enumerate(as_completed(futs), 1):
                status, c, err = fut.result()
                done += status == "done"; skipped += status == "skip"; failed += status == "fail"
                if status == "fail":
                    print(f"  [visual][fail] {c['stem']}: {err}")
                if n % 200 == 0 or n == len(clips):
                    print(f"  [visual] {n}/{len(clips)}  new={done} skip={skipped} fail={failed}")
    else:
        print("[visual] skipped")

    write_index(clips, out)


def write_index(clips, out: Path):
    import torch
    fields = (["stem", "split", "audio_path", "visual_path"] + EMOTIONS6
              + ["gender", "valence", "n_audio", "n_visual"])
    rows = []
    for c in clips:
        ap = out / "audio" / f"{c['stem']}.pt"
        vp = out / "visual" / f"{c['stem']}.pt"
        n_a = c.get("n_audio", "")
        n_v = c.get("n_visual", "")
        if n_a == "" and already_done(ap):
            try: n_a = int(torch.load(ap, map_location="cpu")["feats"].shape[0])
            except Exception: n_a = ""
        if n_v == "" and already_done(vp):
            try: n_v = int(torch.load(vp, map_location="cpu")["feats"].shape[0])
            except Exception: n_v = ""
        row = {"stem": c["stem"], "split": c["split"],
               "audio_path": f"audio/{c['stem']}.pt" if already_done(ap) else "",
               "visual_path": f"visual/{c['stem']}.pt" if already_done(vp) else "",
               "gender": c["gender"], "valence": c["valence"], "n_audio": n_a, "n_visual": n_v}
        for e, lbl in zip(EMOTIONS6, c["labels"]):
            row[e] = lbl
        rows.append(row)
    idx = out / "index.csv"
    with open(idx, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    have_a = sum(1 for r in rows if r["audio_path"])
    have_v = sum(1 for r in rows if r["visual_path"])
    print(f"[index] {len(rows)} clips -> {idx}  (audio cached: {have_a}, visual cached: {have_v})")


def main():
    ap = argparse.ArgumentParser(description="ElderReact frozen-encoder feature extraction")
    ap.add_argument("--data-root", default="data/elderreact")
    ap.add_argument("--out", default="data/elderreact/features")
    ap.add_argument("--audio-model", default="facebook/hubert-large-ls960-ft")
    ap.add_argument("--visual-cols", choices=["au_r", "au_rc", "au_pose"], default="au_r")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-audio", action="store_true")
    ap.add_argument("--skip-visual", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
