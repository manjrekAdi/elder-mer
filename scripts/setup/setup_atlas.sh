#!/usr/bin/env bash
# =============================================================================
# setup_atlas.sh -- One-shot bootstrap of the Atlas GPU server (CUDA 12.8,
#                   Blackwell sm_120) for the elder-mer project.
#
# Sets up, in order:
#   1. ~/data dirs + HF/Torch/Whisper cache env vars (big volume, not home)
#   2. Miniconda (installed if missing) + conda env 'elder-mer' (Python 3.11)
#   3. PyTorch cu128 + shared requirements + whisper + peft
#   4. The repo (clone or pull)  -- push your latest commits first!
#   5. Datasets onto ~/data:
#        - CREMA-D audio+video (downloaded fresh via git-LFS)
#        - ElderReact          (cannot be downloaded; prints the rsync to
#                               run from the Mac, skips if already present)
#        - MER2023 train       (140 GB; only if HF access is approved)
#   6. GPU smoke tests: sm_120 matmul + one LoRA training step (task 1.5 gate)
#
# Usage (on Atlas):
#   bash setup_atlas.sh          # or: bash ~/elder-mer/scripts/setup/setup_atlas.sh
#
# Idempotent: every step checks before acting; rerun freely after a failure.
# No sudo required (ffmpeg, git-lfs come from conda-forge).
# =============================================================================
set -uo pipefail   # NOT -e: optional steps (MER2023) may fail without aborting

ENV_NAME="elder-mer"
PY_VERSION="3.11"
REPO_URL="https://github.com/manjrekAdi/elder-mer.git"
# Atlas storage (this account, per admin Hamidreza 2026-05-27): the /data0
# allocation IS the home dir -- /home/manjrek5 is bind-mounted to
# /data0/home/manjrek5, and there is NO /data0/users/manjrek5. So $HOME already
# lives on the /data0 volume (~200 GB): code + conda envs go in $HOME, while
# datasets/checkpoints go on ~/data (-> /data1/users, 750 GB).
REPO_DIR="$HOME/elder-mer"
DATA_ROOT="$HOME/data/elder-mer-data"

step()  { echo; echo "==> $*"; }
ok()    { echo "    [ok] $*"; }
warn()  { echo "    [warn] $*"; }

# --- 0. sanity: Linux box with an NVIDIA GPU --------------------------------
step "[0/6] Sanity checks"
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: this script targets the Atlas Linux server (got $(uname -s))." >&2
  echo "       On the Mac, use setup_m4.sh instead." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found -- is this really Atlas?" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/    GPU: /'

# --- 1. data volume + cache env vars ----------------------------------------
step "[1/6] Data dirs + cache env vars (HF/Whisper/Torch downloads -> ~/data)"
mkdir -p "$DATA_ROOT" "$HOME/data/hf_cache" "$HOME/data/torch_cache" "$HOME/data/whisper_cache"
BASHRC_TAG="# >>> elder-mer caches >>>"
if ! grep -q "$BASHRC_TAG" "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" <<EOF

$BASHRC_TAG
export HF_HOME="\$HOME/data/hf_cache"
export TORCH_HOME="\$HOME/data/torch_cache"
export WHISPER_CACHE="\$HOME/data/whisper_cache"
export XDG_CACHE_HOME="\$HOME/data/.cache"
# <<< elder-mer caches <<<
EOF
  ok "cache exports appended to ~/.bashrc"
else
  ok "cache exports already in ~/.bashrc"
fi
export HF_HOME="$HOME/data/hf_cache" TORCH_HOME="$HOME/data/torch_cache"
export WHISPER_CACHE="$HOME/data/whisper_cache" XDG_CACHE_HOME="$HOME/data/.cache"

# --- 2. conda + env ----------------------------------------------------------
step "[2/6] Miniconda + '$ENV_NAME' env (Python $PY_VERSION)"
if ! command -v conda >/dev/null 2>&1; then
  if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    ok "found existing miniconda3"
  else
    echo "    installing Miniconda to ~/miniconda3 ..."
    curl -fsSL -o /tmp/miniconda.sh \
      "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh"
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  fi
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
  conda init bash >/dev/null
else
  eval "$(conda shell.bash hook)"
fi
# Miniconda base + envs live under ~/miniconda3 (i.e. on /data0/home for this
# account), which is the correct volume -- no envs_dirs redirection needed.
if ! conda env list | grep -qE "(^|/)$ENV_NAME\s"; then
  conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi
conda activate "$ENV_NAME"
ok "active env: $CONDA_DEFAULT_ENV (python $(python -V 2>&1 | cut -d' ' -f2))"

# ffmpeg (whisper) + git-lfs (CREMA-D) without sudo
command -v ffmpeg  >/dev/null 2>&1 || conda install -y -c conda-forge ffmpeg
command -v git-lfs >/dev/null 2>&1 || conda install -y -c conda-forge git-lfs
git lfs install --skip-repo >/dev/null
ok "ffmpeg + git-lfs available"

# --- 3. python stack ---------------------------------------------------------
step "[3/6] PyTorch cu128 + shared requirements"
python - <<'EOF' 2>/dev/null
import torch, sys
sys.exit(0 if torch.version.cuda and torch.version.cuda.startswith("12.8") else 1)
EOF
if [[ $? -ne 0 ]]; then
  pip install --upgrade pip setuptools wheel
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
else
  ok "torch cu128 already installed"
fi

REQ="$REPO_DIR/scripts/setup/requirements-common.txt"
if [[ -f "$REQ" ]]; then
  pip install -q -r "$REQ"
else
  warn "requirements-common.txt not found yet (repo not cloned) -- installed after step 4"
fi
# whisper needs setuptools present at build time; peft for LoRA; faster-whisper
# for the confidence-extraction utility; hf CLI for gated datasets
pip install -q --no-build-isolation openai-whisper
pip install -q peft "huggingface_hub[cli]" faster-whisper
ok "python stack installed"

# --- 4. repo -----------------------------------------------------------------
step "[4/6] Repo at $REPO_DIR  (in \$HOME, on the /data0 volume)"
if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" pull --ff-only || warn "pull failed (local changes?) -- continuing"
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
# late-install requirements if step 3 skipped them
[[ -f "$REQ" ]] && pip install -q -r "$REQ"
# datasets live on the big volume; repo scripts use ./data/... paths
if [[ ! -e "$REPO_DIR/data" ]]; then
  ln -s "$DATA_ROOT" "$REPO_DIR/data"
  ok "symlinked $REPO_DIR/data -> $DATA_ROOT"
fi

# --- 5. datasets -------------------------------------------------------------
step "[5/6] Datasets -> $DATA_ROOT"

# 5a. CREMA-D (fully open: audio via download script, video via sparse LFS add)
CREMA="$DATA_ROOT/crema_d"
if [[ -f "$CREMA/metadata.csv" && -d "$CREMA/_cremad_repo/AudioWAV" ]]; then
  ok "CREMA-D audio + metadata present"
else
  python "$REPO_DIR/scripts/setup/download_cremad.py" --method git --out "$CREMA"
fi
if [[ -d "$CREMA/_cremad_repo/VideoFlash" ]] && \
   [[ $(ls "$CREMA/_cremad_repo/VideoFlash" 2>/dev/null | wc -l) -ge 7442 ]]; then
  ok "CREMA-D video present"
else
  echo "    fetching VideoFlash (2.3 GB via git-LFS) ..."
  git -C "$CREMA/_cremad_repo" sparse-checkout add VideoFlash
  git -C "$CREMA/_cremad_repo" lfs pull --include "VideoFlash/*"
  [[ -e "$CREMA/VideoFlash" ]] || ln -s "$CREMA/_cremad_repo/VideoFlash" "$CREMA/VideoFlash"
fi

# 5b. ElderReact (access-restricted; must come from the Mac)
ER="$DATA_ROOT/elderreact"
if [[ -d "$ER/ElderReact_train" ]]; then
  ok "ElderReact present"
else
  warn "ElderReact cannot be downloaded publicly. From the MAC, run:"
  echo "      rsync -avP ~/Documents/elder-mer/data/elderreact/ atlas:~/data/elder-mer-data/elderreact/"
  echo "    (also ships labels + features + the OpenFace reliability CSVs)"
fi
# OpenFace per-frame gate inputs for CREMA-D (computed on the Mac; rsync >> rebuild)
if [[ ! -d "$DATA_ROOT/crema_d/experiment3_visual/per_frame" ]]; then
  warn "OpenFace gate inputs for CREMA-D not present. From the MAC, run:"
  echo "      rsync -avP ~/Documents/elder-mer/data/crema_d/experiment3_visual/ atlas:~/data/elder-mer-data/crema_d/experiment3_visual/"
fi

# 5c. MER2023 train set (gated, 140 GB; Experiment 7 only -- optional)
MER="$DATA_ROOT/mer2023"
if [[ -d "$MER" ]] && ls "$MER"/mer2023train.z* >/dev/null 2>&1; then
  ok "MER2023 archives present"
elif hf auth whoami >/dev/null 2>&1 || huggingface-cli whoami >/dev/null 2>&1; then
  echo "    attempting MER2023 download (gated; skips cleanly if not approved) ..."
  HFC=$(command -v hf || command -v huggingface-cli)
  if "$HFC" download MERChallenge/MER2023 --repo-type dataset --local-dir "$MER"; then
    ok "MER2023 downloaded (unzip with password from README_AFTER_APPROVAL.md)"
  else
    warn "MER2023 denied/failed -- approval pending? Rerun this script later; it resumes."
  fi
else
  warn "Not logged into Hugging Face; skipping MER2023."
  echo "      To enable: hf auth login   (then rerun this script)"
fi

# --- 6. GPU smoke tests (the task-1.5 go/no-go) ------------------------------
step "[6/6] GPU smoke tests: sm_120 matmul + one LoRA training step"
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available!"
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f"    GPU: {name}  capability sm_{cap[0]}{cap[1]}  torch {torch.__version__} cuda {torch.version.cuda}")
x = torch.randn(2048, 2048, device="cuda")
y = (x @ x).sum().item()
print(f"    matmul on GPU: ok (checksum {y:.3e})")
if cap < (12, 0):
    print("    [warn] capability below sm_120 -- is this the Blackwell card?")
EOF
if [[ $? -ne 0 ]]; then
  echo "ERROR: basic CUDA test failed. Check driver/torch-cu128 pairing." >&2
  exit 1
fi

python - <<'EOF'
# One real LoRA training step on a small causal LM: validates the full
# training stack (transformers + peft + cu128 backward pass) end-to-end.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").cuda()
model = get_peft_model(model, LoraConfig(r=8, target_modules=["c_attn"]))
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
batch = tok(["Emotion recognition for older adults."] * 2, return_tensors="pt").to("cuda")
out = model(**batch, labels=batch["input_ids"])
out.loss.backward()
torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4).step()
print(f"    LoRA step: ok (loss {out.loss.item():.3f}, {n:,} trainable params)")
print("    TASK 1.5 SMOKE TEST PASSED: cu128 training stack works on this GPU.")
EOF
if [[ $? -ne 0 ]]; then
  echo "ERROR: LoRA smoke test failed -- the cu128 wheel may lag this GPU." >&2
  echo "       Check pytorch.org for a newer build and flag to Hamidreza." >&2
  exit 1
fi

# --- done --------------------------------------------------------------------
echo
echo "============================================================"
echo "ATLAS SETUP COMPLETE"
echo "  env        : conda activate $ENV_NAME"
echo "  repo       : $REPO_DIR   (in \$HOME, on the /data0 volume)"
echo "  data       : $DATA_ROOT (= ~/data -> /data1; symlinked as repo data/)"
echo
echo "Remaining manual steps:"
echo "  1. wandb login                  (once, before tracked runs)"
echo "  2. rsync ElderReact + OpenFace features from the Mac (commands above)"
echo "  3. Port Emotion-LLaMA / TiCAL codebases (task 1.5 proper):"
echo "       bash $REPO_DIR/scripts/setup/port_codebases_atlas.sh"
echo "     The LoRA smoke test passing means the underlying stack is sound."
echo
echo "  ! Atlas is a SHARED 4-GPU server. Before any heavy/long run, check the"
echo "    GPU Monitoring & Reservation Sheet and reserve a GPU; don't grab all"
echo "    four. (Setup smoke tests are tiny, so they're fine to run unreserved.)"
echo "============================================================"
