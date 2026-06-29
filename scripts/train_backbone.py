#!/usr/bin/env python3
"""
train_backbone.py -- Task 3.2 / Experiment 1: projector + LoRA training loop.
=============================================================================

Trains the Emotion-LLaMA-style backbone validated in Task 1.5, on the frozen
features cached by Task 3.1 (scripts/extract_features.py). Everything upstream
(HuBERT audio, OpenFace visual) is frozen and pre-extracted; everything here is
the thin trainable layer:

    cached features            TRAINABLE                       FROZEN
    audio  [T_a, 1024] ─┐
                        ├─> per-modality projector ─> pseudo-tokens ─┐
    visual [T_v, 17]  ─┘    (attention-pool to n_query tokens each)   │
                                                                      ├─> LLM (+LoRA) ─> summary
                            learned [SUMMARY] token ───────────────────┘    hidden ─> head ─> 6 logits

Only the projectors + summary token + classification head + the LoRA adapter on
the LLM are trained. The LLM body stays frozen (the ~2.6M-trainable-param
MSE-Adapter efficiency recipe). Experiment 1 is BIMODAL (audio+video); the text
branch arrives in Experiment 2.

DESIGN DECISIONS (documented so they're easy to revisit):
  * Prediction = a classification HEAD on the LLM's summary-token hidden state,
    not generative decoding. Cleaner/faster/calibrated F1 for a baseline; the
    generative Emotion-LLaMA variant is an alternative, not required to measure
    the age gap. The frozen-encoder -> projector -> frozen-LLM+LoRA invariant
    (what the contributions act on) is preserved either way.
  * Pseudo-tokens: each modality's variable-length sequence is attention-pooled
    to a fixed n_query tokens (MSE-Adapter-style), so batching is trivial and
    temporal info is kept (unlike mean-pooling).
  * The [SUMMARY] token is placed LAST: the LLM is causal, so only a final
    position attends to all modality tokens. Its last-layer hidden state feeds
    the head.
  * Default --llm is a small OPEN model (Qwen2.5-0.5B) so this runs out-of-the-
    box for local dev; switch to meta-llama/Llama-2-7b-hf on Atlas for the
    faithful/published comparison (Task 3.5). Backbone-agnostic via AutoModel.
  * SPEAKER-INDEPENDENT k-fold by actor (no actor in both train and test, or
    results leak). One fold per invocation (--n-folds / --fold); run all folds
    to get held-out predictions for all 91 actors, which the [M4] age-covariate
    analysis (the actual Experiment 1 result) regresses emotion F1 on age.

OUTPUTS (per fold, under --out):
    fold<f>/best.pt              trainable weights (projectors+head+summary+LoRA)
    fold<f>/predictions.csv      per test clip: stem, actor, age, sex, true, pred, probs
    fold<f>/metrics.json         acc / macro-F1 / weighted-F1 on this fold's test
  Concatenate all folds' predictions.csv for the across-all-actors age analysis.

USAGE (in the `elder-mer` conda env; Atlas for real runs -- reserve a GPU):
    # quick local sanity (tiny LLM, few steps, handful of clips):
    python scripts/train_backbone.py --fold 0 --n-folds 5 --limit 200 --max-steps 5
    # one real fold:
    python scripts/train_backbone.py --fold 0 --n-folds 5 --epochs 15
    # all folds (bash loop):
    for f in 0 1 2 3 4; do python scripts/train_backbone.py --fold $f --n-folds 5 --epochs 15; done
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]   # label index 0..5
NUM_CLASSES = len(EMOTIONS)


# ============================================================================
# Data: read the 3.1 cache (index.csv + per-clip .pt feature files)
# ============================================================================
class FeatureDataset:
    """Holds clip metadata; loads cached audio/visual feature tensors on access.
    Stores no module/tensor handles so it stays picklable for DataLoader workers."""
    def __init__(self, rows, features_dir):
        self.rows = rows
        self.features_dir = Path(features_dir)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch
        r = self.rows[i]
        a = torch.load(self.features_dir / r["audio_path"], map_location="cpu",
                       weights_only=False)["feats"].float()      # [T_a, 1024]
        v = torch.load(self.features_dir / r["visual_path"], map_location="cpu",
                       weights_only=False)["feats"].float()      # [T_v, 17]
        return {
            "audio": a, "visual": v,
            "label": int(r["label_idx"]),
            "stem": r["stem"], "actor_id": r["actor_id"], "age": int(r["age"]),
            "sex": r["sex"], "emotion_code": r["emotion_code"],
        }


def collate_fn(batch):
    """Pad each modality to the batch max; return feats, pad-masks (True=pad),
    labels, meta. Top-level (captures no module handle) so DataLoader worker
    processes can serialize it."""
    import torch

    def pad(key, dim):
        seqs = [b[key] for b in batch]
        T = max(s.shape[0] for s in seqs)
        out = torch.zeros(len(seqs), T, dim)
        mask = torch.ones(len(seqs), T, dtype=torch.bool)   # True = padding
        for i, s in enumerate(seqs):
            out[i, :s.shape[0]] = s
            mask[i, :s.shape[0]] = False
        return out, mask
    audio, a_mask = pad("audio", batch[0]["audio"].shape[1])
    visual, v_mask = pad("visual", batch[0]["visual"].shape[1])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    meta = [{k: b[k] for k in ("stem", "actor_id", "age", "sex", "emotion_code")} for b in batch]
    return {"audio": audio, "a_mask": a_mask, "visual": visual, "v_mask": v_mask,
            "labels": labels, "meta": meta}


# ============================================================================
# Model: per-modality projector + frozen LLM(+LoRA) + classification head
# ============================================================================
def build_modules(torch):
    import torch.nn as nn

    class ModalityProjector(nn.Module):
        """Variable-length [B,T,d_in] -> fixed [B,n_query,d_llm] pseudo-tokens via
        learned-query attention pooling (ignores padded frames via key mask)."""
        def __init__(self, d_in, d_model, d_llm, n_query, n_heads):
            super().__init__()
            self.in_proj = nn.Linear(d_in, d_model)
            self.query = nn.Parameter(torch.randn(n_query, d_model) * 0.02)
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            self.out_proj = nn.Linear(d_model, d_llm)

        def forward(self, x, pad_mask):
            B = x.shape[0]
            h = self.in_proj(x)                                  # [B,T,d_model]
            q = self.query.unsqueeze(0).expand(B, -1, -1)        # [B,n_query,d_model]
            # if a whole row is padding (shouldn't happen), unmask to avoid NaN
            safe = pad_mask.clone()
            safe[safe.all(dim=1)] = False
            a, _ = self.attn(q, h, h, key_padding_mask=safe)     # [B,n_query,d_model]
            return self.out_proj(self.norm(a))                   # [B,n_query,d_llm]

    class TrimodalLLM(nn.Module):
        def __init__(self, llm_name, d_model, n_query, n_heads, lora_r, lora_alpha, dtype):
            super().__init__()
            from transformers import AutoModelForCausalLM
            from peft import LoraConfig, get_peft_model
            llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=dtype)
            llm.config.use_cache = False
            # LoRA on all linear layers -> portable across LLM families
            llm = get_peft_model(llm, LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                target_modules="all-linear", task_type="CAUSAL_LM"))
            self.llm = llm
            d_llm = llm.config.hidden_size
            self.d_llm = d_llm
            self.audio_proj = ModalityProjector(1024, d_model, d_llm, n_query, n_heads)
            self.visual_proj = ModalityProjector(17, d_model, d_llm, n_query, n_heads)
            self.summary = nn.Parameter(torch.randn(1, 1, d_llm) * 0.02)
            self.head = nn.Sequential(nn.LayerNorm(d_llm), nn.Linear(d_llm, NUM_CLASSES))

        def forward(self, audio, a_mask, visual, v_mask):
            B = audio.shape[0]
            at = self.audio_proj(audio, a_mask)                  # [B,n_q,d_llm]
            vt = self.visual_proj(visual, v_mask)                # [B,n_q,d_llm]
            summ = self.summary.expand(B, -1, -1)                # [B,1,d_llm]
            embeds = torch.cat([at, vt, summ], dim=1)            # [B, 2n_q+1, d_llm]
            embeds = embeds.to(next(self.llm.parameters()).dtype)
            attn_mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
            out = self.llm(inputs_embeds=embeds, attention_mask=attn_mask,
                           output_hidden_states=True)
            last = out.hidden_states[-1][:, -1, :]               # summary-token state (causal)
            return self.head(last.float())

    return TrimodalLLM


def trainable_report(model):
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return t, tot


# ============================================================================
# Speaker-independent k-fold split (by actor)
# ============================================================================
def fold_split(rows, n_folds, fold, seed):
    actors = sorted({r["actor_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(actors)
    buckets = [actors[i::n_folds] for i in range(n_folds)]
    test_actors = set(buckets[fold])
    val_actors = set(buckets[(fold + 1) % n_folds])
    train_actors = set(actors) - test_actors - val_actors
    pick = lambda S: [r for r in rows if r["actor_id"] in S]
    return pick(train_actors), pick(val_actors), pick(test_actors)


# ============================================================================
# Metrics
# ============================================================================
def metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n": len(y_true),
    }


# ============================================================================
# Train / eval
# ============================================================================
def evaluate(model, loader, device, torch):
    model.train(False)
    ys, ps, probs, metas = [], [], [], []
    with torch.no_grad():
        for b in loader:
            logits = model(b["audio"].to(device), b["a_mask"].to(device),
                           b["visual"].to(device), b["v_mask"].to(device))
            pr = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = pr.argmax(1)
            ys += b["labels"].tolist(); ps += pred.tolist()
            probs += pr.tolist(); metas += b["meta"]
    return ys, ps, probs, metas


def run(args):
    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    feats_dir = Path(args.features)
    rows = [r for r in csv.DictReader(open(feats_dir / "index.csv"))
            if r["audio_path"] and r["visual_path"]]
    if args.limit and args.limit < len(rows):
        # sample ACROSS actors (index.csv is sorted by clip, so a head-slice
        # would cover only a few actors and collapse the speaker-fold split)
        rows = random.Random(args.seed).sample(rows, args.limit)
    if not rows:
        raise SystemExit(f"[error] no cached clips with both modalities in {feats_dir}/index.csv. "
                         "Run scripts/extract_features.py first.")

    device = (args.device if args.device != "auto" else
              "mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if (args.fp16 and device == "cuda") else torch.float32

    train_rows, val_rows, test_rows = fold_split(rows, args.n_folds, args.fold, args.seed)
    print(f"[3.2] fold {args.fold}/{args.n_folds} | device={device} | llm={args.llm}")
    print(f"      clips: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
          f"(speaker-independent: "
          f"{len({r['actor_id'] for r in train_rows})}/"
          f"{len({r['actor_id'] for r in val_rows})}/"
          f"{len({r['actor_id'] for r in test_rows})} actors)")

    mk = lambda rws, sh: DataLoader(FeatureDataset(rws, feats_dir), batch_size=args.batch_size,
                                    shuffle=sh, collate_fn=collate_fn, num_workers=args.workers)
    train_loader, val_loader, test_loader = mk(train_rows, True), mk(val_rows, False), mk(test_rows, False)

    TrimodalLLM = build_modules(torch)
    model = TrimodalLLM(args.llm, args.d_model, args.n_query, args.n_heads,
                        args.lora_r, args.lora_alpha, dtype).to(device)
    tr, tot = trainable_report(model)
    print(f"      trainable params: {tr:,} / {tot:,} ({100*tr/tot:.2f}%)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    out_dir = Path(args.out) / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # save only the trainable weights (projectors+head+summary+LoRA) -- not the
    # frozen LLM body (which would be GBs per fold).
    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
    def save_trainable(path):
        sd = model.state_dict()
        torch.save({k: sd[k].cpu() for k in trainable_keys if k in sd}, path)

    best_val, step = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train(True)
        running, nb = 0.0, 0
        for b in train_loader:
            logits = model(b["audio"].to(device), b["a_mask"].to(device),
                           b["visual"].to(device), b["v_mask"].to(device))
            loss = loss_fn(logits, b["labels"].to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            running += loss.item(); step += 1; nb += 1
            if args.max_steps and step >= args.max_steps:
                break
        yv, pv, _, _ = evaluate(model, val_loader, device, torch) if len(val_rows) else ([], [], [], [])
        vmf1 = metrics(yv, pv)["macro_f1"] if yv else 0.0
        print(f"  epoch {epoch:2d}  train_loss={running/max(1,nb):.4f}  val_macroF1={vmf1:.4f}")
        if vmf1 >= best_val:
            best_val = vmf1
            save_trainable(out_dir / "best.pt")
        if args.max_steps and step >= args.max_steps:
            print("  [max-steps reached: stopping early (smoke test)]")
            break

    # ---- final test-set predictions (the input to the age-gap analysis) ----
    yt, pt, probs, metas = evaluate(model, test_loader, device, torch)
    m = metrics(yt, pt) if yt else {"note": "empty test fold"}
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "actor_id", "age", "sex", "emotion_code",
                    "true_idx", "pred_idx", "true", "pred"] + [f"p_{e}" for e in EMOTIONS])
        for meta, yt_i, pt_i, pr in zip(metas, yt, pt, probs):
            w.writerow([meta["stem"], meta["actor_id"], meta["age"], meta["sex"],
                        meta["emotion_code"], yt_i, pt_i, EMOTIONS[yt_i], EMOTIONS[pt_i]]
                       + [f"{x:.4f}" for x in pr])
    json.dump({**m, "fold": args.fold, "n_folds": args.n_folds, "llm": args.llm,
               "best_val_macro_f1": best_val}, open(out_dir / "metrics.json", "w"), indent=2)
    print(f"[3.2] fold {args.fold} TEST: {m}")
    print(f"      wrote {out_dir}/predictions.csv + metrics.json + best.pt")


def main():
    ap = argparse.ArgumentParser(description="Task 3.2 / Experiment 1 backbone training")
    ap.add_argument("--features", default="data/crema_d/features")
    ap.add_argument("--out", default="data/crema_d/exp1")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-0.5B",
                    help="HF CausalLM (default small/open for dev; meta-llama/Llama-2-7b-hf for the faithful run)")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--n-query", type=int, default=4, help="pseudo-tokens per modality")
    ap.add_argument("--d-model", type=int, default=256, help="projector inner width")
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fp16", action="store_true", help="fp16 LLM (CUDA only)")
    ap.add_argument("--limit", type=int, default=0, help="subset clips (smoke test)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke test)")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers (0 is safest on Mac/MPS; >0 fine on CUDA)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
