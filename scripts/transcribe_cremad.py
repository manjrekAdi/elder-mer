#!/usr/bin/env python3
"""
transcribe_cremad.py -- Task 0.5 + Experiment 2: Whisper transcription of CREMA-D.
==================================================================================

Experiment 2 adds the ASR-text channel on top of audio+video. This script runs
Whisper (faster-whisper) over every CREMA-D clip and caches, per clip:
  - the transcript (the text the trimodal model reads), and
  - Whisper's per-word confidences + a mean/min summary.

The transcript feeds Experiment 2 (naive trimodal). The per-word confidences are
the native reliability signal the Experiment-4 ASR-text gate will act on, so we
capture them now even though Exp 2 itself does not gate. This is also the salvage
of the retired wer_vs_age.py `transcribe()` into a reusable utility (task 0.5).

Note: CREMA-D's 12 fixed sentences are clean and scripted, so Whisper transcribes
them near-perfectly and the text carries little emotion -- no text *gain* is
expected on CREMA-D (that is measured later on ElderReact). Exp 2 just establishes
the ungated-trimodal baseline.

OUTPUT: data/crema_d/transcripts.csv, one row per clip:
    stem, transcript, mean_conf, min_conf, n_words, word_confs(JSON list of [word,prob])

RUN (in the `elder-mer` conda env):
    python scripts/transcribe_cremad.py --limit 10          # smoke test
    python scripts/transcribe_cremad.py                     # all 7,442 (resumable)
    python scripts/transcribe_cremad.py --model base.en     # faster/smaller
    python scripts/transcribe_cremad.py --device cuda --compute-type float16   # on Atlas GPU
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path


def build_model(model_size, device, compute_type, cpu_threads):
    from faster_whisper import WhisperModel
    print(f"[model] loading faster-whisper '{model_size}' (device={device}, {compute_type}) "
          "-- first run downloads weights ...")
    return WhisperModel(model_size, device=device, compute_type=compute_type,
                        cpu_threads=cpu_threads)


def transcribe(model, wav_path: Path):
    """Return (transcript, mean_conf, min_conf, word_confs). Confidence from per-word
    probability; falls back to exp(avg_logprob) per segment when word probs are absent.
    word_confs is a list of [word, probability] (probability may be None)."""
    segments, _ = model.transcribe(str(wav_path), language="en", word_timestamps=True)
    text_parts, word_confs, probs, seg_logprobs = [], [], [], []
    for seg in segments:
        seg_logprobs.append(seg.avg_logprob)
        if seg.words:
            for w in seg.words:
                text_parts.append(w.word)
                p = float(w.probability) if w.probability is not None else None
                word_confs.append([w.word.strip(), p])
                if p is not None:
                    probs.append(p)
        else:
            text_parts.append(seg.text)
    transcript = "".join(text_parts).strip()
    if probs:
        mean_conf, min_conf = sum(probs) / len(probs), min(probs)
    elif seg_logprobs:
        mean_conf = min_conf = math.exp(sum(seg_logprobs) / len(seg_logprobs))
    else:
        mean_conf = min_conf = float("nan")
    return transcript, mean_conf, min_conf, word_confs


def load_clips(data_root: Path, limit):
    meta = data_root / "metadata.csv"
    if not meta.exists():
        sys.exit(f"[error] {meta} not found. Run scripts/setup/download_cremad.py first.")
    wav_dir = data_root / "AudioWAV"
    clips = []
    with open(meta, newline="") as f:
        for row in csv.DictReader(f):
            stem = Path(row["file"]).stem
            clips.append({"stem": stem, "wav": wav_dir / f"{stem}.wav"})
    clips.sort(key=lambda c: c["stem"])
    return clips[:limit] if limit else clips


FIELDS = ["stem", "transcript", "mean_conf", "min_conf", "n_words", "word_confs"]


def main():
    ap = argparse.ArgumentParser(description="Whisper transcription of CREMA-D (Exp 2 text channel)")
    ap.add_argument("--data-root", default="data/crema_d")
    ap.add_argument("--out", default="data/crema_d/transcripts.csv")
    ap.add_argument("--model", default="small.en",
                    help="faster-whisper model (tiny/base/small/medium(.en) | large-v3)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--compute-type", default="int8", help="int8 (cpu) | float16 (cuda)")
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N clips (smoke test)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    clips = load_clips(Path(args.data_root), args.limit)

    # resume: skip stems already transcribed
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            for r in csv.DictReader(f):
                done.add(r["stem"])
    todo = [c for c in clips if c["stem"] not in done]
    print(f"[0.5/Exp2] {len(clips)} clips | {len(done)} already done | {len(todo)} to transcribe")
    if not todo:
        print("[done] all clips already transcribed."); return

    model = build_model(args.model, args.device, args.compute_type, args.cpu_threads)
    new_file = not out.exists() or out.stat().st_size == 0
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        ok = fail = 0
        for i, c in enumerate(todo, 1):
            if not c["wav"].exists():
                fail += 1
                print(f"  [miss] {c['wav'].name}")
                continue
            try:
                transcript, mean_conf, min_conf, word_confs = transcribe(model, c["wav"])
                w.writerow({"stem": c["stem"], "transcript": transcript,
                            "mean_conf": f"{mean_conf:.4f}", "min_conf": f"{min_conf:.4f}",
                            "n_words": len(word_confs), "word_confs": json.dumps(word_confs)})
                f.flush()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"  [err] {c['stem']}: {str(e)[:80]}")
            if i % 200 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] ok={ok} fail={fail}")
    print(f"[done] transcribed {ok} clips ({fail} failed) -> {out}")


if __name__ == "__main__":
    main()
