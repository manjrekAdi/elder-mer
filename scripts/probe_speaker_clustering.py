#!/usr/bin/env python3
"""
probe_speaker_clustering.py -- feasibility probe for recovering ElderReact person IDs
=====================================================================================

ElderReact ships no speaker/person identity (filenames are source-video titles),
which blocks per-person calibration (Contribution 3). This probe tests whether a
pretrained speaker-embedding model can recover the ~46 recurring identities by
clustering, WITHOUT ground-truth IDs.

Validation uses gender (the one per-clip identity attribute we DO have): a person
cannot change gender across clips, so if clips cluster by speaker, clusters must be
gender-pure. Two parameter-free signals:

  1. same-gender nearest-neighbour rate  -- for each clip, is its cosine-nearest
     other clip the same gender?  Baseline (random) = sum(p_gender^2) ~ 0.54.
  2. weighted cluster gender-purity      -- agglomerative cluster into ~46 groups,
     average majority-gender fraction.   Baseline ~ max(p_gender) ~ 0.63.

Values WELL above baseline => embeddings track speaker identity => full clustering
is worth doing. Near baseline => fall back to per-recording-condition calibration.

Usage:
    python scripts/probe_speaker_clustering.py --n 300
"""
import argparse, csv, random, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path
import numpy as np

DATA = Path("data/elderreact")
SPLITS = ("train", "dev", "test")


def load_clips():
    g = {}
    for s in SPLITS:
        for line in (DATA / "_elderreact_repo/Annotations" / f"{s}_labels.txt").read_text().splitlines():
            p = line.split()
            if len(p) >= 8:
                g[p[0]] = p[7]
    clips = []
    for s in SPLITS:
        for v in sorted((DATA / f"ElderReact_{s}").glob("*.mp4")):
            if v.name in g:
                clips.append({"file": v.name, "video": v, "gender": g[v.name]})
    return clips


def extract_wav(video: Path, wav: Path):
    subprocess.run(["ffmpeg", "-y", "-i", str(video), "-ac", "1", "-ar", "16000",
                    "-vn", str(wav)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from resemblyzer import VoiceEncoder, preprocess_wav
    from sklearn.cluster import AgglomerativeClustering

    clips = load_clips()
    random.seed(args.seed)
    random.shuffle(clips)
    work = clips[:args.n]
    gtot = Counter(c["gender"] for c in work)
    print(f"[probe] {len(work)} clips sampled  (gender mix: {dict(gtot)})")

    enc = VoiceEncoder()
    embs, kept = [], []
    tmp = Path(tempfile.mkdtemp())
    for i, c in enumerate(work, 1):
        wav = tmp / (c["file"] + ".wav")
        try:
            extract_wav(c["video"], wav)
            embs.append(enc.embed_utterance(preprocess_wav(wav)))
            kept.append(c)
        except Exception as e:
            print(f"  [skip] {c['file']}: {str(e)[:60]}")
        finally:
            wav.unlink(missing_ok=True)
        if i % 50 == 0:
            print(f"  embedded {i}/{len(work)}")
    X = np.array(embs)
    gen = np.array([c["gender"] for c in kept])

    # drop degenerate embeddings (silent / music-only clips -> zero or NaN vector)
    norms = np.linalg.norm(X, axis=1)
    good = np.isfinite(norms) & (norms > 1e-6)
    dropped = int((~good).sum())
    X, gen = X[good], gen[good]
    n = len(X)
    print(f"[probe] embedded {n} usable clips, dim={X.shape[1]} "
          f"({dropped} dropped: silent/music-only audio)")

    # cosine similarity matrix
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -1)

    # signal 1: same-gender nearest neighbour
    nn = S.argmax(1)
    same_gender_nn = float(np.mean(gen[nn] == gen))
    p = np.array([np.mean(gen == g) for g in set(gen)])
    nn_baseline = float(np.sum(p ** 2))

    # signal 2: cluster gender-purity. A random 300-clip draw from 46 recurring
    # people contains nearly all 46, so k~46 is correct (NOT 46*n/total).
    purity_baseline = float(p.max())
    purity_at = {}
    for k in (30, 46, 60):
        lab = AgglomerativeClustering(n_clusters=min(k, n), metric="cosine",
                                      linkage="average").fit_predict(X)
        purity_at[k] = sum(Counter(gen[lab == c]).most_common(1)[0][1]
                           for c in set(lab)) / n
    purity = purity_at[46]

    print("\n" + "=" * 60)
    print("FEASIBILITY PROBE RESULTS")
    print("=" * 60)
    print(f"same-gender NN rate : {same_gender_nn:.3f}   (random baseline {nn_baseline:.3f})")
    print(f"cluster purity      : " +
          "  ".join(f"k={k}:{v:.3f}" for k, v in purity_at.items()) +
          f"   (random baseline {purity_baseline:.3f})")
    strong = same_gender_nn > 0.85 and purity > 0.85
    lift = same_gender_nn - nn_baseline
    print(f"lift over baseline  : +{lift:.3f}")
    print("-" * 60)
    if strong:
        print("VERDICT: FEASIBLE. Voice embeddings track speaker identity well")
        print("above chance. Scale to all 1,323 clips + add a face cross-check;")
        print("Contribution 3's per-person grouping can be recovered.")
    elif lift > 0.15:
        print("VERDICT: PROMISING but not clean. Real signal, but expect cluster")
        print("noise. Try a stronger encoder (ECAPA-TDNN) before committing.")
    else:
        print("VERDICT: WEAK. Embeddings barely beat chance. Fall back to")
        print("per-recording-condition calibration (no identity needed).")
    print("=" * 60)


if __name__ == "__main__":
    main()
