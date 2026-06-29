#!/usr/bin/env python3
"""
extract_features.py -- Task 3.1: cache frozen-encoder features for CREMA-D.
===========================================================================

The pipeline freezes its encoders, so running them is pure one-way computation:
the same clip always yields the same features. This script runs each frozen
encoder over all 7,442 CREMA-D clips ONCE and caches the result to disk, so the
projector+LoRA training loop (Task 3.2) only ever loads small pre-computed
tensors -- it never touches raw audio/video. This is also the M4->Atlas
interface: extract here on the Mac (MPS), then ship the feature tensors up.

Because the encoders are frozen, caching is bit-for-bit identical to recomputing
every epoch -- no accuracy is lost. (The only thing it forecloses is per-epoch
augmentation of the frozen branch, which this pipeline does not use.)

TWO BRANCHES (Experiment 1 is bimodal; the text branch is deferred to Exp 2):

  audio  -- frozen HuBERT over each WAV -> a per-frame sequence [T_a, 1024].
            HuBERT is the Emotion-LLaMA audio encoder and beats Whisper-as-
            encoder for emotion (~0.84 vs ~0.53 F1, per the lit review).

  visual -- SWAPPABLE encoder (--visual):
              openface : aggregate the per-frame OpenFace AU intensities already
                         extracted in Experiment 3 -> [T_v, D]. Cheap (no new
                         compute), interpretable, and the same tool that feeds
                         the confidence gate (Contribution 2). DEFAULT.
              mae      : deep VideoMAE features over the raw video. Higher
                         absolute-F1 ceiling, closer to the Emotion-LLaMA
                         template; needed to match published baselines. NOT yet
                         implemented -- a documented drop-in slot (see
                         extract_visual_mae). Start with openface; add mae later
                         as a second cache without touching anything else.

The visual .pt also bundles the per-frame `confidence`/`success` arrays -- the
native reliability signal Experiment 4's gate scales tokens by -- aligned to the
feature frames, so the gate has its inputs ready.

OUTPUT (per-clip, resumable):
    <out>/audio/<stem>.pt     {"feats":[T_a,1024] float, "model":..., "sr":16000}
    <out>/visual/<stem>.pt    {"feats":[T_v,D] float, "conf":[T_v], "success":[T_v],
                               "cols":[...], "source":"openface"}
    <out>/index.csv           one row per clip: stem, paths, actor_id, age, sex,
                              emotion_code, emotion, label_idx, n_audio, n_visual

LABELS: 6 emotions sorted alphabetically -> ANG=0 DIS=1 FEA=2 HAP=3 NEU=4 SAD=5.

RUN (in the `elder-mer` conda env, which has torch+transformers+librosa):
    python scripts/extract_features.py --limit 8        # smoke test both branches
    python scripts/extract_features.py --skip-visual    # audio only
    python scripts/extract_features.py                  # full run, all 7,442
    python scripts/extract_features.py --fp16           # half-size cache

Then ship to Atlas:
    rsync -a data/crema_d/features/ atlas:~/data/elder-mer-data/crema_d/features/
"""
import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]   # sorted -> label index
LABEL = {e: i for i, e in enumerate(EMOTIONS)}


# ----------------------------------------------------------------------------
# clip list + demographics
# ----------------------------------------------------------------------------
def load_clips(data_root: Path, limit: int):
    """One entry per CREMA-D clip from metadata.csv, joined with sex from
    VideoDemographics.csv. Returns dicts with stem/age/sex/emotion."""
    meta = data_root / "metadata.csv"
    demo = data_root / "VideoDemographics.csv"
    if not meta.exists():
        sys.exit(f"[error] {meta} not found. Run scripts/setup/download_cremad.py first.")

    sex_by_actor = {}
    if demo.exists():
        with open(demo, newline="") as f:
            for row in csv.DictReader(f):
                sex_by_actor[str(row["ActorID"]).strip()] = row["Sex"].strip()

    clips = []
    with open(meta, newline="") as f:
        for row in csv.DictReader(f):
            stem = Path(row["file"]).stem            # AudioWAV/1001_DFA_ANG_XX.wav -> 1001_DFA_ANG_XX
            code = row["emotion_code"].strip()
            if code not in LABEL:
                continue
            clips.append({
                "stem": stem,
                "actor_id": str(row["actor_id"]).strip(),
                "age": int(float(row["age"])),
                "sex": sex_by_actor.get(str(row["actor_id"]).strip(), ""),
                "emotion_code": code,
                "emotion": row["emotion"].strip(),
                "label_idx": LABEL[code],
            })
    clips.sort(key=lambda c: c["stem"])
    return clips[:limit] if limit else clips


# ----------------------------------------------------------------------------
# audio branch: frozen HuBERT -> [T, 1024]
# ----------------------------------------------------------------------------
def load_16k_mono(wav_path: Path):
    import soundfile as sf
    import librosa
    y, sr = sf.read(str(wav_path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return y


class AudioEncoder:
    """Frozen HuBERT. Lazily loaded so --skip-audio needs no model download."""
    def __init__(self, model_name, device, dtype):
        import torch
        from transformers import AutoFeatureExtractor, HubertModel
        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        print(f"[audio] loading {model_name} on {device} ...")
        self.fe = AutoFeatureExtractor.from_pretrained(model_name)
        # .train(False) puts the module in inference mode (equivalent to .eval())
        self.model = HubertModel.from_pretrained(model_name).to(device).train(False)

    def encode(self, wav_path: Path):
        y = load_16k_mono(wav_path)
        inputs = self.fe(y, sampling_rate=16000, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model(inputs.input_values.to(self.device))
        feats = out.last_hidden_state.squeeze(0).to("cpu", self.dtype)  # [T, 1024]
        return feats


# ----------------------------------------------------------------------------
# visual branch: swappable encoder
# ----------------------------------------------------------------------------
def select_visual_cols(header, kind):
    """Pick which OpenFace columns become the per-frame visual feature vector.
    Derived from the CSV header so we never hardcode the exact AU set."""
    au_r = [c for c in header if c.startswith("AU") and c.endswith("_r")]   # 17 intensities
    au_c = [c for c in header if c.startswith("AU") and c.endswith("_c")]   # 18 presences
    pose = [c for c in header if c.startswith("pose_")]                     # 6 head-pose
    if kind == "au_r":
        return au_r
    if kind == "au_rc":
        return au_r + au_c
    if kind == "au_pose":
        return au_r + au_c + pose
    sys.exit(f"[error] unknown --visual-cols {kind}")


def extract_visual_openface(stem, per_frame_dir: Path, cols_kind):
    """Aggregate one clip's OpenFace per-frame CSV into a feature sequence plus
    the aligned reliability signal (confidence/success) the gate will use."""
    csv_path = per_frame_dir / f"{stem}.csv"
    if not csv_path.exists():
        return None, f"no OpenFace CSV ({csv_path.name})"
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
    header = [h.strip() for h in header]
    feat_cols = select_visual_cols(header, cols_kind)
    if not feat_cols:
        return None, "no feature columns matched"
    idx = {h: i for i, h in enumerate(header)}

    feats, conf, succ = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for parts in reader:
            if len(parts) < len(header):
                continue
            try:
                feats.append([float(parts[idx[c]]) for c in feat_cols])
                conf.append(float(parts[idx["confidence"]]))
                succ.append(int(float(parts[idx["success"]])))
            except (ValueError, KeyError):
                continue
    if not feats:
        return None, "no usable frames"
    return {
        "feats": np.asarray(feats, dtype=np.float32),    # [T_v, D]
        "conf": np.asarray(conf, dtype=np.float32),      # [T_v]  gate input
        "success": np.asarray(succ, dtype=np.float32),   # [T_v]
        "cols": feat_cols,
        "source": "openface",
    }, None


def extract_visual_mae(stem, data_root: Path):
    """DROP-IN SLOT for deep VideoMAE features over the raw FLV (higher absolute
    ceiling, matches the Emotion-LLaMA template). Intentionally not implemented:
    the OpenFace-first plan gives a valid pipeline now; fill this in to add a
    second visual cache later without changing anything else. Would: decode
    data_root/VideoFlash/<stem>.flv -> sample frames -> frozen VideoMAE ->
    [T_v, 768]; return the same dict shape (feats, plus conf=ones)."""
    raise NotImplementedError(
        "visual=mae not implemented yet. Use --visual openface (default). "
        "See extract_visual_mae() for the drop-in contract.")


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def already_done(path: Path):
    return path.exists() and path.stat().st_size > 0


def run(args):
    import torch

    data_root = Path(args.data_root)
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "visual").mkdir(parents=True, exist_ok=True)
    per_frame_dir = data_root / "experiment3_visual" / "per_frame"
    wav_dir = data_root / "AudioWAV"

    # device + storage dtype
    if args.device == "auto":
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    store_dtype = torch.float16 if args.fp16 else torch.float32

    clips = load_clips(data_root, args.limit)
    print(f"[3.1] {len(clips)} clips | device={device} | dtype={'fp16' if args.fp16 else 'fp32'}"
          f" | visual={args.visual}:{args.visual_cols}")

    # ---- audio (GPU-bound: sequential on the one MPS/CUDA device) ----
    if not args.skip_audio:
        enc = AudioEncoder(args.audio_model, device, store_dtype)
        done = skipped = failed = 0
        for i, c in enumerate(clips, 1):
            dst = out / "audio" / f"{c['stem']}.pt"
            if already_done(dst) and not args.overwrite:
                skipped += 1
            else:
                wav = wav_dir / f"{c['stem']}.wav"
                if not wav.exists():
                    failed += 1
                    print(f"  [audio][miss] {wav.name}")
                else:
                    try:
                        feats = enc.encode(wav)
                        torch.save({"feats": feats, "model": args.audio_model,
                                    "sr": 16000}, dst)
                        done += 1
                        c["n_audio"] = int(feats.shape[0])
                    except Exception as e:
                        failed += 1
                        print(f"  [audio][err] {c['stem']}: {str(e)[:80]}")
            if i % 200 == 0 or i == len(clips):
                print(f"  [audio] {i}/{len(clips)}  new={done} skip={skipped} fail={failed}")
    else:
        print("[audio] skipped (--skip-audio)")

    # ---- visual (CPU/IO-bound: thread the OpenFace aggregation) ----
    if not args.skip_visual:
        if args.visual == "mae":
            extract_visual_mae(clips[0]["stem"], data_root)  # raises with guidance

        def do_one(c):
            dst = out / "visual" / f"{c['stem']}.pt"
            if already_done(dst) and not args.overwrite:
                return ("skip", c, None)
            res, err = extract_visual_openface(c["stem"], per_frame_dir, args.visual_cols)
            if err:
                return ("fail", c, err)
            t = {"feats": torch.from_numpy(res["feats"]).to(store_dtype),
                 "conf": torch.from_numpy(res["conf"]),
                 "success": torch.from_numpy(res["success"]),
                 "cols": res["cols"], "source": res["source"]}
            torch.save(t, dst)
            c["n_visual"] = int(res["feats"].shape[0])
            return ("done", c, None)

        done = skipped = failed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(do_one, c): c for c in clips}
            for n, fut in enumerate(as_completed(futs), 1):
                status, c, err = fut.result()
                if status == "done":
                    done += 1
                elif status == "skip":
                    skipped += 1
                else:
                    failed += 1
                    print(f"  [visual][fail] {c['stem']}: {err}")
                if n % 1000 == 0 or n == len(clips):
                    print(f"  [visual] {n}/{len(clips)}  new={done} skip={skipped} fail={failed}")
    else:
        print("[visual] skipped (--skip-visual)")

    # ---- index.csv (rebuilt from on-disk caches so it's correct after resumes) ----
    write_index(clips, out)


def write_index(clips, out: Path):
    import torch
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
        rows.append({
            "stem": c["stem"],
            "audio_path": f"audio/{c['stem']}.pt" if already_done(ap) else "",
            "visual_path": f"visual/{c['stem']}.pt" if already_done(vp) else "",
            "actor_id": c["actor_id"], "age": c["age"], "sex": c["sex"],
            "emotion_code": c["emotion_code"], "emotion": c["emotion"],
            "label_idx": c["label_idx"], "n_audio": n_a, "n_visual": n_v,
        })
    idx_path = out / "index.csv"
    with open(idx_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    have_a = sum(1 for r in rows if r["audio_path"])
    have_v = sum(1 for r in rows if r["visual_path"])
    print(f"[index] {len(rows)} clips -> {idx_path}  (audio cached: {have_a}, visual cached: {have_v})")


def main():
    ap = argparse.ArgumentParser(description="Task 3.1 frozen-encoder feature extraction for CREMA-D")
    ap.add_argument("--data-root", default="data/crema_d")
    ap.add_argument("--out", default="data/crema_d/features")
    ap.add_argument("--audio-model", default="facebook/hubert-large-ls960-ft",
                    help="frozen HuBERT checkpoint (e.g. facebook/hubert-large-ll60k "
                         "preserves more paralinguistic/emotion info; ls960-ft is ASR-tuned)")
    ap.add_argument("--visual", choices=["openface", "mae"], default="openface")
    ap.add_argument("--visual-cols", choices=["au_r", "au_rc", "au_pose"], default="au_r",
                    help="openface feature set: au_r=17 intensities (default), "
                         "au_rc=+18 presences, au_pose=+head pose")
    ap.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    ap.add_argument("--fp16", action="store_true", help="store features as float16 (half size)")
    ap.add_argument("--limit", type=int, default=0, help="process only first N clips (smoke test)")
    ap.add_argument("--workers", type=int, default=8, help="threads for visual aggregation")
    ap.add_argument("--skip-audio", action="store_true")
    ap.add_argument("--skip-visual", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="re-extract even if cached")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
