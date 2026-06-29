# Project Tasks — Multimodal Emotion Analysis for Older Adults

Derived from `Multimodal_Emotion_Older_Adults_Proposal_LLM_v3.md`. Tasks are in
execution order. The ordering follows the proposal's own staging logic
(§3.7): **backbone first, then the confidence gate, then calibration** — and
the decision-gate experiment (Experiment 3) runs *before* any modeling, since
it tests the premise (H1) that Contribution 2 acts on.

Phases 1–2 are the critical path: everything in them either unblocks later
work (access requests, codebase porting) or de-risks the thesis premise
(the visual decision gate).

## Where each task runs

| Tag | Machine | Used for |
|---|---|---|
| `[M4]` | M4 MacBook Pro (local, MPS) | coding, debugging, OpenFace + frozen-encoder feature extraction, Whisper inference, statistical analysis, plots, write-up, admin |
| `[Atlas]` | Atlas GPU server (CUDA 12.8, Blackwell sm_120) | projector + LoRA training runs, ablation sweeps, published-baseline training (CASP/READ), large-VLM inference (Qwen2.5-VL) |
| `[M4→Atlas]` | both | develop/smoke-test locally on a small subset, then run for real on Atlas |

**Rule of thumb:** the encoders and LLM body stay frozen, so everything
*upstream* of training (feature extraction, transcription, confidence
signals) and *downstream* of it (regressions, significance tests, plots,
write-up) is local M4 work. Only steps that backprop through the projector
+ LoRA — or run a large model the M4 can't hold — need Atlas. Cache all
extracted features to disk once, then ship the feature tensors (not raw
media) to Atlas.

---

## Phase 0 — Environment & groundwork ✅ (mostly done)

- [x] **0.1 Environment setup scripts** for both machines: M4 MacBook (MPS —
  coding, debugging, Whisper inference, feature extraction) and Atlas server
  (CUDA 12.8 / Blackwell sm_120 — real training). `scripts/setup/setup_m4.sh`,
  `setup_atlas.sh`, shared `requirements-common.txt`.
- [x] **0.2 CREMA-D audio + demographics downloaded.** `[M4]` 7,442 WAVs via
  git-LFS sparse clone, `VideoDemographics.csv` (per-actor age 20–74),
  `metadata.csv` with ground-truth transcripts per clip
  (`scripts/setup/download_cremad.py`).
- [x] **0.3 CREMA-D demographics check.** `[M4]` Confirmed only 6 of 91
  actors are 60+; consequence (already folded into the proposal): age is
  modeled as a **continuous covariate** across all actors, never as a 60+
  subgroup split.
- [x] **0.4 ASR pilot (retired experiment).** `[M4]` Whisper WER-vs-age on
  CREMA-D (`scripts/wer_vs_age.py`) confirmed the scripted clips do not
  stress ASR enough to anchor H1's ASR form there. The CREMA-D WER-vs-age
  regression is **no longer a core experiment** — the ASR-text gate is
  validated on spontaneous elderly speech instead (Phase 6).
- [ ] **0.5 Salvage the reusable piece of the retired script.** `[M4]`
  Extract the `transcribe()` function (faster-whisper with per-word
  probabilities) into a shared utility — it is exactly the Whisper
  confidence-signal extraction the ASR-text gate needs later (task 6.4).
  Retire the rest of `wer_vs_age.py` or mark it clearly as superseded.

## Phase 1 — Week-1 critical path (parallel; everything here unblocks later phases)

- [ ] **1.1 Send dataset access requests** `[M4]` (admin/email; long lead
  times — send first, nothing downstream waits on them):
  - [ ] USOMS-e (academic EULA, University of Augsburg) — German elderly
    spontaneous narratives with transcripts, ComParE 2020 Elderly Emotion
    Sub-Challenge.
  - [ ] DementiaBank-Emotion (via TalkBank) — English spontaneous
    picture-description speech with CHAT transcripts + emotion layer.
  - [ ] MPDD-Elderly (ACM MM Grand Challenge corpus, 564 samples, released
    fusion baseline 0.6675) — needed for Experiments 2 and 6.
  - Note: **no core result depends on USOMS-e / DementiaBank** — they only
    upgrade the ASR-gate validation from "downstream F1 effect" to "direct
    WER evidence."
  - [x] MER2023 (for Experiment 7 only): gated access **GRANTED**, confirmed
    2026-06-26 (the 2026-06-12 attempt never registered a request —
    Gated Repos Status showed none on record — so it was re-submitted and
    approved). Test sets + password README (~280 MB) downloaded to
    `data/mer2023/` on the M4 (`hf download` command in
    `scripts/download_mer2023_mac.sh`, dest updated); the 140 GB 7-part
    train archive goes **directly to Atlas** (doesn't fit the M4; split zip
    needs all parts to extract). Unzip passwords are in
    `data/mer2023/README_AFTER_APPROVAL.md`.
- [x] **1.2 Confirm ElderReact raw-video access works end-to-end.** `[M4]`
  Done 2026-06-10: all 1,323 raw videos (h264 1280×720) at
  `data/elderreact/ElderReact_{train,dev,test}` (615/355/353), labels +
  pre-extracted features (COVAREP audio, OpenFace video) cloned to
  `data/elderreact/_elderreact_repo`. Label↔video cross-check: 0 mismatches
  in every split. Label format: file, 6 binary emotions, gender, valence.
- [x] **1.3 Download CREMA-D video.** `[M4]` Done 2026-06-10: all 7,442
  FLVs (VP6F 480×360, 2.3 GiB) pulled via git-LFS sparse checkout into
  `data/crema_d/_cremad_repo/VideoFlash`, symlinked at
  `data/crema_d/VideoFlash` (same pattern as `AudioWAV`). Verified with
  ffprobe. (Mirror to Atlas's data volume later only if extraction is rerun
  there; otherwise only extracted features get shipped.)
- [x] **1.4 Install OpenFace.** `[M4]` Done 2026-06-10: built from source at
  `~/tools/OpenFace` (binaries in `~/tools/OpenFace/build/bin`), CEN
  patch-expert models downloaded. Verified on a CREMA-D FLV: per-frame
  `confidence` + `success` columns, 17 AU intensity (`AU*_r`) + 18 AU
  presence (`AU*_c`) columns, and PDM shape params (`p_scale`, `p_rx`…,
  `p_0`–`p_33`) for the landmark-fit fallback signal. Three macOS-arm64
  build fixes were needed (documented in the root `README.md`):
  `OpenBLAS_HOME` env var, `stdc++fs` link guard, prefer `std::filesystem`
  over Boost.
- [x] **1.5 Codebase porting pass (Blackwell fit) — RESULT: GO.** `[Atlas]`
  Done 2026-06-27 (`scripts/setup/port_codebases_atlas.sh`). Emotion-LLaMA
  and TiCAL pin older PyTorch (2.0.0 / 2.3.1) lacking sm_120 kernels; rebuilt
  each in an isolated conda env against cu128 torch + their own pinned deps,
  and **a LoRA training step ran end-to-end on the RTX PRO 6000 Blackwell
  (sm_120)** for both. Verdict: reuse is viable (no need to reimplement the
  projector+LoRA recipe). Per-repo gotchas the script now handles: install
  via each env's python by abs path (conda activate doesn't survive a `tee`
  subshell); backfill `setuptools<81` for old `accelerate`'s `pkg_resources`;
  and for peft 0.2.0 set `lora_dropout` explicitly + move model to GPU after
  adapter injection. NOT yet run: each repo's real `train.py` on weights/data
  (needs LLaMA-2-7B + MiniGPT-v2 ckpt / MOSEI feats) — that's Phase 3.
  Atlas env built by `setup_atlas.sh`: conda env `elder-mer` (py3.11),
  torch 2.11.0+cu128, repo at `~/elder-mer`, data on `~/data` (-> /data1).
- [ ] **1.6 Set up experiment tracking.** `[M4→Atlas]` `wandb login` on
  both machines; on Atlas, set the cache env vars so HF/Whisper downloads
  land on the 750 GB data volume.

## Phase 2 — Experiment 3: the decision gate (run before any modeling)

Tests H1, the premise of Contribution 2: *upstream encoder reliability
declines with age.* **Entirely local — no training involved.**

- [x] **2.1 Run OpenFace over all CREMA-D videos.** `[M4]` Done 2026-06-11:
  all 7,442 clips extracted (`scripts/experiment3_visual_gate.py --full`),
  zero failures. Per-frame CSVs kept at
  `data/crema_d/experiment3_visual/per_frame/` for Experiment 4 reuse.
  Ops notes: per-worker threading must be clamped
  (`OPENCV_FOR_THREADS_NUM=1` etc., now in the script — ~400× slowdown
  otherwise) and overnight runs need `caffeinate` (first attempt died to
  Mac sleep).
- [x] **2.2 Regress each visual reliability signal on actor age.** `[M4]`
  Done 2026-06-11: continuous regression across all 91 actors, five
  signals (conf mean/p10/min, success rate, landmark jitter). Results in
  `data/crema_d/experiment3_visual/regression.txt`.
- [x] **2.3 Decision gate — RESULT: H1 visual NOT supported on CREMA-D.**
  All five signals are age-flat (|r| ≤ 0.13, all p ≥ 0.22). Cause is a
  ceiling effect, not a weak trend: OpenFace confidence saturates at its
  0.98 cap on studio video (mean 0.9775, 99% of clips > 0.95, only 0.1%
  with any tracking failure; 60+ actors identical to ≤30). CREMA-D's
  clean frontal studio conditions don't stress the visual encoder — the
  same pattern as ASR on its scripted speech. Per the proposal's fallback:
  escalate the visual-gate validation to in-the-wild elderly video
  (ElderReact), and report H1-visual as not demonstrable on clean studio
  data. **Decision needed with supervisor:** reframe the gate's premise
  from "encoder reliability declines with age" to "encoder reliability
  degrades on realistic elderly recordings, and the gate acts on the
  reliability signal directly" (ElderReact has no per-person ages, so an
  age regression is not possible there).
- [x] **2.3b ElderReact follow-up — gate signal CONFIRMED in-the-wild.**
  `[M4]` Done 2026-06-11 (`scripts/elderreact_visual_reliability.py`; all
  1,323 clips, zero failures). On in-the-wild elderly video the saturation
  vanishes: conf_mean 0.9445 vs 0.9775, p10 0.893 vs 0.973; clips below
  0.95 confidence: **35.4% vs 1.0%**; below 0.90: **11.6% vs 0.0%**; clips
  with ≥1 tracking failure: **21.8% vs 0.1%**. The visual gate has a fat
  low-confidence tail to act on exactly where the system is meant to work.
  Experiment 4's visual-gate validation therefore targets ElderReact;
  per-frame CSVs (gate inputs) already extracted at
  `data/elderreact/visual_reliability/per_frame/`. Full comparison in
  `data/elderreact/visual_reliability/summary.txt`.
- [ ] **2.4 Write up the H1 result** `[M4]` — now a two-part story:
  (a) on clean studio video the encoder saturates and no age trend is
  measurable (CREMA-D null, ceiling effect); (b) on in-the-wild elderly
  video reliability degrades massively (ElderReact comparison). Frame the
  gate's premise as reliability-driven rather than age-regression-driven;
  discuss the reframing with the supervisor. Still a standalone deliverable
  for Task 2.

## Phase 3 — Backbone: Experiments 1 & 2 (Contribution 1, baselines)

Build the Emotion-LLaMA-style backbone: frozen audio encoder (HuBERT or
wav2vec 2.0) + frozen visual encoder (MAE-family or OpenFace features) +
LLM text embedder; trainable projector mapping non-text features into token
space; LoRA on the LLM. Encoders and LLM body stay frozen throughout.

- [x] **3.1 Feature extraction pipeline.** `[M4]` Done 2026-06-29
  (`scripts/extract_features.py`). Frozen HuBERT-large audio ([T,1024]) +
  aggregated OpenFace AU-intensity visual ([T,17], with per-frame
  confidence/success bundled as the Exp-4 gate inputs), all 7,442 CREMA-D
  clips cached to `data/crema_d/features/` (3.7 GB) + `index.csv`. Verified
  HuBERT features are deterministic across loads (the pos_conv "newly
  initialized" warning is benign). Visual encoder is swappable (OpenFace now,
  VideoMAE a documented drop-in for the absolute-number/published-baseline
  runs). Raw CREMA-D already rsync'd to Atlas; ship the feature cache up
  before the 7B run.
- [x] **3.2 Projector + LoRA training loop.** `[M4→Atlas]` Done 2026-06-29
  (`scripts/train_backbone.py`). Per-modality attention-pool -> pseudo-tokens
  -> frozen LLM(+LoRA) -> summary-token classification head (6 emotions).
  Speaker-independent k-fold by actor (no leakage). 10M/504M trainable =
  2.00% (the lightweight adapter recipe). LLM is configurable: Qwen2.5-0.5B
  (open, fast, the "now" dev/iteration backbone) -> LLaMA-2-7B on Atlas (the
  faithful/published-comparison "later" backbone).
- [x] **3.3 Experiment 1 — bimodal baseline (audio + video) on CREMA-D —
  RESULT: age gap CONFIRMED.** `[M4]` Done 2026-06-29 (Qwen2.5-0.5B, 5-fold,
  `scripts/analyze_exp1_age_gap.py`). Held-out predictions for all 91 actors;
  overall acc=0.640, macro-F1=0.638. **Continuous per-actor regression on
  age: accuracy -0.0225/decade (r=-0.264, p=0.011) and macro-F1
  -0.0252/decade (r=-0.265, p=0.011), both significant**; per-clip
  correctness~age point-biserial r=-0.062 (p<1e-5). Baseline emotion
  recognition declines with actor age -- the within-dataset gap is real and
  age-attributable. NOTE the contrast with Exp 3: OpenFace *tracking
  confidence* was age-flat, so on clean studio data the gap is at the
  *expression-readability* level, not encoder tracking (aged-expression
  failure mode), and the encoder-reliability gate (Contribution 2) needs
  in-the-wild ElderReact degradation to act on. Outputs:
  `data/crema_d/exp1_qwen0.5b/` (per-fold predictions/metrics + analysis/).
  TODO: re-run on LLaMA-2-7B (Atlas) to confirm the gap holds at scale.
- [ ] **3.4 Experiment 2 — naive trimodal (+ ASR text, no gating).**
  `[Atlas]` training; `[M4]` Whisper transcription of the text branch.
  All modalities trusted equally; quantifies ungated fusion. Note:
  text-channel *gains* are assessed on ElderReact / MPDD-Elderly later —
  CREMA-D's 12 fixed sentences are emotionally uninformative by
  construction, so no text benefit is expected there.
- [ ] **3.5 Benchmark sanity check** `[Atlas]` against published
  Emotion-LLaMA-style numbers to confirm the backbone is implemented
  correctly before any contribution is layered on.

## Phase 4 — Experiment 4: confidence-gated fusion (Contribution 2)

- [ ] **4.1 Implement the confidence-gated projector.** `[M4]` (pure code +
  unit tests; the gate signals were already extracted in Phases 2–3.)
  Before each modality's pseudo-tokens enter the LLM, scale them by that
  modality's upstream native reliability signal: the Phase-2-selected
  OpenFace signal (visual), voice-activity/signal-quality estimate (audio),
  Whisper token confidence (text). One unified gate mechanism across all
  three channels.
- [ ] **4.2 Train gated vs. naive fusion** `[Atlas]` under identical budgets
  on CREMA-D (visual + audio gates active; the ASR gate is exercised in
  Phase 6 where ASR actually struggles).
- [ ] **4.3 Test H2.** `[M4]` (statistical analysis of Atlas outputs.)
  Target: a significant **age-by-method interaction** in a continuous
  regression across all 91 actors; no significant loss at any age (high
  confidence ⇒ tokens essentially unscaled). Run the never-hurts check
  explicitly.
- [ ] **4.4 Ablations:** `[Atlas]` per-channel gates on/off; gate signal
  choice (detection conf vs. AU conf vs. landmark error); scaling function.

## Phase 5 — Experiment 5: per-person test-time calibration (Contribution 3)

- [ ] **5.1 Implement per-person calibration.** `[M4]` (implementation —
  it's an inference-time mechanism, no backprop.) From a ~10-second
  calibration window per speaker, estimate per-modality reliability
  (per-modality prediction entropy + the Contribution-2 gate signals;
  optionally TiCAL's typicality aggregated over the person's calibration
  clips) and reweight that speaker's modality tokens at inference.
  Contribution 2 sets per-input gates; this sets each person's baseline
  for those gates.
- [ ] **5.2 Leakage protocol:** `[M4]` calibration clips are excluded from
  that person's test metrics; enforce via speaker-independent splits.
- [ ] **5.3 Set up the published baselines:** `[Atlas]` CASP (AAAI-25) as
  the global TTA baseline; READ (ICLR 2024) as the reliability-driven TTA
  reference. (Both adapt model state — GPU runs.)
- [ ] **5.4 Test H3.** `[Atlas]` inference runs over the trained backbone;
  `[M4]` per-subject variance / worst-individual analysis. Per-person
  calibration vs. global TTA: reduced worst-individual F1 and per-subject
  F1 variance; flat-affect subgroup +≥5 absolute F1 points. Fallback
  framing (per proposal): if it only matches global TTA on average, report
  it as a lightweight complementary plug-in, not a standalone gain.

## Phase 6 — Experiment 6: transfer to elderly corpora + subgroup analysis

- [ ] **6.1 Run the full pipeline on ElderReact** (and MPDD-Elderly if
  access landed). `[M4]` feature extraction (OpenFace, Whisper, frozen
  encoders) on the raw videos; `[Atlas]` fine-tuning/evaluation runs:
  trimodal, gated, with per-person calibration. Handle ElderReact's
  multi-label, imbalanced setup (happiness 56% vs. fear 11%).
- [ ] **6.2 Test H4.** `[Atlas]` runs; `[M4]` analysis. Gains transfer:
  ≥5 absolute F1 points over ungated fusion (H2 target on ElderReact);
  worst-subgroup F1 +≥5 points over a generic-trained baseline.
- [ ] **6.3 Subgroup analysis** `[M4]` (the ethical core of the evaluation):
  per-subgroup F1 by gender × prosodic variance (flat-affect subgroup
  explicitly), worst-group reported alongside averages.
- [ ] **6.4 Validate the ASR-text gate where ASR genuinely struggles.**
  `[M4]` Whisper transcription + confidence extraction on ElderReact
  spontaneous speech (no reference transcripts ⇒ measure the downstream F1
  effect of gating low-confidence words, using the salvaged
  Whisper-confidence utility from 0.5); `[Atlas]` the gated-vs-ungated
  training comparison. If USOMS-e / DementiaBank access landed:
  additionally show WER rises with age (H1 secondary form, `[M4]` —
  pure Whisper inference + regression) and the gate's effect with direct
  WER evidence.
- [ ] **6.5 Compare against published ElderReact baselines** `[M4]`
  (Sreevidya 2022, Jothimani 2023) and the MPDD-Elderly released baseline
  (0.6675 avg) — numbers come from the papers; no reruns needed.

## Phase 7 — Experiment 7: cross-population non-degradation (supporting)

Visual/audio channels only (DFEW is video-centric; MER2023 is Mandarin).

- [ ] **7.1 Validate Qwen2.5-VL as a binary age-group labeller** `[Atlas]`
  against CREMA-D ground-truth ages (VLM inference over video frames — too
  heavy for the M4); `[M4]` accuracy analysis, including degradation near
  the young/old boundary.
- [ ] **7.2 Label DFEW/MER clips** `[Atlas]` with a margin:
  clearly-younger vs. clearly-older, ambiguous middle band excluded.
  `[M4]` compare resolution / image-quality distributions of the two
  buckets (age must not be confounded with recording quality); report both.
- [ ] **7.3 Test H5.** `[Atlas]` projector trained on young+older vs.
  young-only, evaluated on a mixed-age held-out set; `[M4]` analysis.
  Target: younger-subject F1 within 1 point of the young-only baseline
  while older-subject F1 improves. Fallback: report the trade-off curve
  across young/older mixing ratios and identify the operating point.

## Phase 8 — Write-up & stretch goals

All `[M4]` except where noted.

- [ ] **8.1 Final report**, mapped to the four project tasks:
  Task 1 (lit survey — delivered, extended to the 2024-26 LLM-backbone
  frontier), Task 2 (data audit + H1 age-gap quantification),
  Task 3 (Experiments 1–3 baselines), Task 4 (Experiments 4–6 adaptations,
  subgroup evaluation). State the acted-vs-spontaneous scope limitation;
  note MECO / CINTHeA data as future spontaneous validation.
- [ ] **8.2 Position each claim against its nearest neighbours** in the
  write-up (Santoso 2021/22, COLD-Fusion lineage, READ, CASP, TiCAL,
  EGMF) — the novelty audit in the proposal already scopes this; keep the
  citations explicit.
- [ ] Stretch (only if time permits):
  - [ ] **8.3 Class-balanced focal loss** `[Atlas]` for ElderReact tail
    classes (fear, disgust) — small change, meaningful tail-class F1
    (Gap C). (Loss change ⇒ retraining runs.)
  - [ ] **8.4 SHAP / Integrated-Gradients interpretability** `[M4→Atlas]` —
    which features drive predictions for elderly vs. younger speakers
    (attribution passes need the trained model; develop locally, run on
    Atlas).
  - [ ] **8.5 Whisper fine-tuning on a small senior-speech sample**
    `[Atlas]` — fixes ASR at the source rather than downstream
    (fine-tuning large-v3 is a GPU job).

---

## Dependency summary

```
Phase 1 (access requests, video data, OpenFace, porting)
   └─> Phase 2 (H1 decision gate — picks the visual gate signal)   [all local]
          └─> Phase 3 (backbone, Exp 1–2 baselines)                [extract local, train Atlas]
                 └─> Phase 4 (confidence gate, Exp 4 — H2)         [code local, train Atlas]
                        └─> Phase 5 (calibration, Exp 5 — H3)      [code local, runs Atlas]
                               └─> Phase 6 (transfer, Exp 6 — H4)  [extract local, train Atlas]
Phase 7 (Exp 7 — H5) needs only Phase 3's projector training; can run in
parallel with Phases 5–6 once the backbone is stable.                [mostly Atlas]
Phase 8 consumes everything.                                         [local]
```

Practical consequence of the split: **Phases 0–2 need no Atlas time at
all** — the H1 decision gate can be fully resolved locally while the
Phase-1.5 porting pass is the only thing touching the server. Atlas becomes
load-bearing from Phase 3 onward, and the M4↔Atlas interface is always the
same artifact: cached feature tensors going up, prediction/metric files
coming back down.

## Hypothesis → task map

| Hypothesis | Tested in | Fallback (per proposal) |
|---|---|---|
| H1 — encoder reliability declines with age | 2.2–2.4 (visual, primary); 6.4 (ASR, secondary) | flat detection conf ⇒ reroute gate to AU conf / landmark error |
| H2 — gated fusion beats naive, gains concentrate on older | 4.3; ElderReact form in 6.2 | — (core claim) |
| H3 — per-person calibration beats global TTA | 5.4 | report as complementary plug-in if only matching on average |
| H4 — gains transfer to ElderReact, reach worst subgroup | 6.2–6.3 | — (core claim) |
| H5 — older-adaptation doesn't cost younger performance | 7.3 | report mixing-ratio trade-off curve as operating-point result |
