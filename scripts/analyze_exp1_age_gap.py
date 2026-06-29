#!/usr/bin/env python3
"""
analyze_exp1_age_gap.py -- Experiment 1 deliverable: the within-CREMA-D age gap.
================================================================================

Experiment 1 establishes the young-vs-old performance gap that every later
experiment must close. The backbone (Task 3.2) is trained speaker-independently
in k folds; each actor lands in exactly one test fold, so concatenating the
folds' predictions.csv gives held-out predictions for ALL 91 actors across the
full 20-74 age range. This script regresses emotion-recognition performance on
ACTOR AGE as a continuous covariate (per the proposal: not a 60+ subgroup split,
since CREMA-D has only six actors aged 60+).

Two views of the same question "does baseline accuracy fall with age?":
  * per-actor   : regress each actor's accuracy / macro-F1 on their age (n=91).
                  Slope/decade, 95% CI, Pearson r, p. The headline result.
  * per-clip    : correctness ~ age via point-biserial r and a logistic odds
                  ratio per decade (n=7,442). More power, ignores per-actor
                  clustering -- reported as support, not the primary test.

A negative, significant age slope = the baseline gap is real and age-attributable
(same recording setup for all actors, so it is not a confound). A flat slope is
also a valid finding (as on the visual gate, Exp 3): it would say the encoders'
emotion signal does not degrade with age on clean studio data, and the age story
lives on ElderReact instead.

INPUT : each fold's predictions.csv from train_backbone.py (stem, actor_id, age,
        sex, emotion_code, true_idx, pred_idx, ...).
OUTPUT: <out>/regression.txt   human-readable summary (experiment3 style)
        <out>/per_actor.csv     per-actor age, n, accuracy, macro_f1, sex
        <out>/age_gap.png       per-actor accuracy vs age, with OLS fit

USAGE:
    python scripts/analyze_exp1_age_gap.py \
        --pred-glob "data/crema_d/exp1_qwen0.5b/fold*/predictions.csv" \
        --out data/crema_d/exp1_qwen0.5b/analysis
"""
import argparse
import csv
import glob
from pathlib import Path

import numpy as np

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]


def load_predictions(pred_glob):
    files = sorted(glob.glob(pred_glob))
    if not files:
        raise SystemExit(f"[error] no predictions.csv matched {pred_glob!r}. "
                         "Run scripts/train_backbone.py (all folds) first.")
    rows = []
    for fp in files:
        for r in csv.DictReader(open(fp)):
            rows.append(r)
    return files, rows


def per_actor_table(rows):
    """One record per actor: age, n_clips, accuracy, macro_f1, sex."""
    from sklearn.metrics import f1_score, accuracy_score
    by_actor = {}
    for r in rows:
        by_actor.setdefault(r["actor_id"], []).append(r)
    table = []
    for actor, rs in by_actor.items():
        y = [int(r["true_idx"]) for r in rs]
        p = [int(r["pred_idx"]) for r in rs]
        table.append({
            "actor_id": actor,
            "age": int(rs[0]["age"]),
            "sex": rs[0].get("sex", ""),
            "n": len(rs),
            "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, labels=list(range(len(EMOTIONS))),
                                       average="macro", zero_division=0)),
        })
    table.sort(key=lambda d: d["age"])
    return table


def ols(x, y):
    """scipy linregress -> slope/yr, /decade, 95% CI/yr, r, p."""
    from scipy import stats
    lr = stats.linregress(x, y)
    ci = 1.96 * lr.stderr
    return {"slope_yr": lr.slope, "slope_decade": lr.slope * 10,
            "ci_lo": lr.slope - ci, "ci_hi": lr.slope + ci,
            "r": lr.rvalue, "p": lr.pvalue}


def per_clip_age_effect(rows):
    """Correctness vs age across all clips: point-biserial r + logistic OR/decade."""
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    age = np.array([int(r["age"]) for r in rows], dtype=float)
    correct = np.array([int(r["true_idx"] == r["pred_idx"]) for r in rows], dtype=int)
    pb = stats.pointbiserialr(correct, age)        # r, p
    # logistic correctness ~ age; odds ratio per decade = exp(beta*10)
    lr = LogisticRegression()
    lr.fit(age.reshape(-1, 1), correct)
    beta = float(lr.coef_[0][0])
    return {"pb_r": pb.correlation, "pb_p": pb.pvalue,
            "or_decade": float(np.exp(beta * 10)), "base_acc": float(correct.mean())}


def overall_metrics(rows):
    from sklearn.metrics import f1_score, accuracy_score
    y = [int(r["true_idx"]) for r in rows]
    p = [int(r["pred_idx"]) for r in rows]
    return {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, p, average="weighted", zero_division=0),
        "n": len(rows),
    }


def plot(table, fit, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ages = np.array([d["age"] for d in table])
    accs = np.array([d["accuracy"] for d in table])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ages, accs, s=24, alpha=0.7, label="actor")
    xs = np.linspace(ages.min(), ages.max(), 50)
    intercept = accs.mean() - fit["slope_yr"] * ages.mean()
    ax.plot(xs, fit["slope_yr"] * xs + intercept, "r-",
            label=f"OLS: {fit['slope_decade']:+.3f}/decade (p={fit['p']:.3f})")
    ax.set_xlabel("actor age (years)"); ax.set_ylabel("per-actor accuracy")
    ax.set_title("Experiment 1: baseline emotion accuracy vs actor age (CREMA-D)")
    ax.legend(); fig.tight_layout(); fig.savefig(out_png, dpi=130)
    print(f"[plot] -> {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Experiment 1 age-gap analysis")
    ap.add_argument("--pred-glob", default="data/crema_d/exp1_qwen0.5b/fold*/predictions.csv")
    ap.add_argument("--out", default="data/crema_d/exp1_qwen0.5b/analysis")
    ap.add_argument("--model", default="Qwen2.5-0.5B", help="label for the report header")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files, rows = load_predictions(args.pred_glob)
    table = per_actor_table(rows)
    acc_fit = ols([d["age"] for d in table], [d["accuracy"] for d in table])
    f1_fit = ols([d["age"] for d in table], [d["macro_f1"] for d in table])
    clip = per_clip_age_effect(rows)
    overall = overall_metrics(rows)

    ages = [d["age"] for d in table]
    n_60plus = sum(a >= 60 for a in ages)

    # per_actor.csv
    with open(out / "per_actor.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["actor_id", "age", "sex", "n", "accuracy", "macro_f1"])
        w.writeheader(); w.writerows(table)

    # regression.txt (experiment3 style)
    sig = lambda p: "SIGNIFICANT" if p < 0.05 else "flat/ns"
    L = [
        f"Experiment 1: baseline emotion recognition vs actor age (CREMA-D, {args.model})",
        "=" * 78,
        f"folds combined: {len(files)}   actors: {len(table)}   "
        f"age range: {min(ages)}-{max(ages)}   60+ actors: {n_60plus}   clips: {overall['n']}",
        f"overall: accuracy={overall['accuracy']:.4f}  macro-F1={overall['macro_f1']:.4f}  "
        f"weighted-F1={overall['weighted_f1']:.4f}",
        "",
        "PER-ACTOR regression on age (n=actors); H1/Exp1 expects a NEGATIVE slope:",
        f"  accuracy   slope/decade={acc_fit['slope_decade']:+.4f}  "
        f"95%CI/yr=[{acc_fit['ci_lo']:+.5f},{acc_fit['ci_hi']:+.5f}]  "
        f"r={acc_fit['r']:+.3f}  p={acc_fit['p']:.5f}  -> {sig(acc_fit['p'])}",
        f"  macro_f1   slope/decade={f1_fit['slope_decade']:+.4f}  "
        f"95%CI/yr=[{f1_fit['ci_lo']:+.5f},{f1_fit['ci_hi']:+.5f}]  "
        f"r={f1_fit['r']:+.3f}  p={f1_fit['p']:.5f}  -> {sig(f1_fit['p'])}",
        "",
        "PER-CLIP age effect (n=clips, support):",
        f"  correctness~age  point-biserial r={clip['pb_r']:+.3f}  p={clip['pb_p']:.5f}  "
        f"-> {sig(clip['pb_p'])}",
        f"  logistic odds ratio per decade = {clip['or_decade']:.3f}  "
        f"(base accuracy {clip['base_acc']:.4f})",
        "",
        "VERDICT",
        "-------",
    ]
    neg_sig = acc_fit["p"] < 0.05 and acc_fit["slope_decade"] < 0
    if neg_sig:
        L.append("Baseline accuracy declines significantly with actor age: the within-CREMA-D")
        L.append("age gap is real and age-attributable. This is the gap Experiments 4-6 must close.")
    elif acc_fit["p"] < 0.05:
        L.append("Accuracy varies significantly with age but NOT in the expected direction;")
        L.append("inspect per_actor.csv / age_gap.png before interpreting.")
    else:
        L.append("No significant baseline age gap on CREMA-D (slope flat). Consistent with the")
        L.append("Exp-3 ceiling finding: the encoders' emotion signal is age-robust on clean")
        L.append("studio data. The age story is carried by in-the-wild ElderReact (Phase 6).")
    text = "\n".join(L) + "\n"
    (out / "regression.txt").write_text(text)
    print("\n" + text)
    plot(table, acc_fit, out / "age_gap.png")


if __name__ == "__main__":
    main()
