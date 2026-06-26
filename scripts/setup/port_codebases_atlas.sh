#!/usr/bin/env bash
# =============================================================================
# port_codebases_atlas.sh -- Task 1.5 "proper": port the two reference
#   codebases (Emotion-LLaMA, TiCAL) onto the Atlas Blackwell GPU
#   (sm_120, CUDA 12.8) and confirm a LoRA training step runs end-to-end
#   with EACH repo's own dependency stack.
#
# WHY THIS EXISTS
#   setup_atlas.sh step 6 already proves the *generic* stack works (cu128
#   torch + transformers + peft + one LoRA step on distilgpt2). That is
#   Layer A. This script is Layer B: the two repos we want to REUSE pin old,
#   sm_120-incompatible torch (Emotion-LLaMA: torch==2.0.0; TiCAL:
#   torch==2.3.1 / cu11.8). The go/no-go is whether their *other* pinned
#   dependencies (notably transformers/peft) can coexist with a cu128 torch
#   that actually has Blackwell kernels. If yes -> reuse their code. If their
#   old pins can't be reconciled with cu128 -> reimplement the projector+LoRA
#   recipe ourselves.
#
# WHAT IT DOES, per codebase:
#   1. Clone the repo (idempotent).
#   2. Create an ISOLATED conda env (their pins conflict with each other and
#      with the main `elder-mer` env, so never share).
#   3. Install cu128 torch FIRST, then the repo's torch-adjacent pinned deps
#      with the torch/cuda pins STRIPPED so pip can't downgrade torch.
#   4. Verify the GPU is seen as sm_120 and a matmul runs.
#   5. Run a real LoRA forward+backward+optimizer step using THAT env's
#      transformers+peft on the GPU  -> the decisive "one training step" gate.
#   6. Print PASS / PARTIAL / FAIL and an overall reuse-vs-reimplement verdict.
#
# OUT OF SCOPE (deliberately): downloading LLaMA-2 7B / MiniGPT-v2 / MOSEI
#   features and running each repo's full train.py on real data. The proposal
#   scopes 1.5 to validating the GPU/kernel + dependency stack "before
#   committing to the full pipeline", not a full data run. The script PRINTS
#   each repo's real training command so you can do that wiring afterward.
#
# Dependency lists below are transcribed from each repo's pinned env file
# (Emotion-LLaMA environment.yaml; TiCAL requirements) as of 2026-06-26.
# Re-verify if upstream changed.
#
# Usage (ON ATLAS):
#   bash scripts/setup/port_codebases_atlas.sh                 # both repos
#   bash scripts/setup/port_codebases_atlas.sh --only tical    # one repo
#   bash scripts/setup/port_codebases_atlas.sh --workdir ~/ports
#
# Idempotent: re-run freely. No sudo.
# =============================================================================
set -uo pipefail   # NOT -e: we want to capture failures per-step, not abort

# ---- args -------------------------------------------------------------------
# Atlas storage (this account): $HOME is on the /data0 volume
# (/home/manjrek5 -> /data0/home/manjrek5, admin-confirmed), so repos + the
# isolated port conda-envs live under $HOME -- the correct volume.
ONLY=""
WORKDIR="$HOME/elder-mer-ports"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)    ONLY="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,55p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$WORKDIR"
LOGDIR="$WORKDIR/_logs"; mkdir -p "$LOGDIR"

CU128_INDEX="https://download.pytorch.org/whl/cu128"

step()  { echo; echo "==> $*"; }
ok()    { echo "    [ok] $*"; }
warn()  { echo "    [warn] $*"; }
fail()  { echo "    [FAIL] $*"; }

# verdict accumulator: "name=STATUS" lines
declare -a VERDICTS=()

# ---- sanity: Linux + NVIDIA -------------------------------------------------
step "[0] Sanity: Atlas Linux box with an NVIDIA GPU"
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: run this ON ATLAS (got $(uname -s)). It validates the target GPU." >&2
  exit 1
fi
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not found." >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/    GPU: /'

# ---- conda ------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
  else
    echo "ERROR: conda not found. Run setup_atlas.sh first (it installs Miniconda)." >&2
    exit 1
  fi
else
  eval "$(conda shell.bash hook)"
fi
# Port envs are created under ~/miniconda3/envs, i.e. on /data0/home for this
# account -- the correct volume, no redirection needed.

# =============================================================================
# Shared helpers
# =============================================================================

# make_env <env_name> <python_version>  -- create if missing, then activate
make_env() {
  local env="$1" py="$2"
  if ! conda env list | grep -qE "^$env\s"; then
    echo "    creating conda env '$env' (python $py) ..."
    conda create -y -n "$env" "python=$py" >/dev/null 2>&1 \
      || { fail "could not create env $env"; return 1; }
  fi
  conda activate "$env" || { fail "could not activate $env"; return 1; }
  ok "env '$env' active (python $(python -V 2>&1 | cut -d' ' -f2))"
}

# install_cu128_torch  -- cu128 torch, only if not already the active build
install_cu128_torch() {
  if python - <<'PY' 2>/dev/null
import torch,sys
sys.exit(0 if (torch.version.cuda or "").startswith("12.8") else 1)
PY
  then ok "cu128 torch already present"; return 0; fi
  python -m pip install -q --upgrade pip setuptools wheel >/dev/null 2>&1
  echo "    installing cu128 torch (this is the Blackwell-capable build) ..."
  python -m pip install -q torch torchvision torchaudio --index-url "$CU128_INDEX" \
    || { fail "cu128 torch install failed"; return 1; }
  ok "cu128 torch installed"
}

# gpu_smoke  -- assert sm_120 visible + matmul runs in the ACTIVE env
gpu_smoke() {
  python - <<'PY'
import torch, sys
assert torch.cuda.is_available(), "CUDA not available in this env"
cap = torch.cuda.get_device_capability(0)
print(f"    torch {torch.__version__}  cuda {torch.version.cuda}  device sm_{cap[0]}{cap[1]}")
x = torch.randn(1024, 1024, device="cuda"); _ = (x @ x).sum().item()
print("    matmul on GPU: ok")
if cap < (12, 0):
    print("    [warn] capability below sm_120 -- not the Blackwell card?")
    sys.exit(3)
PY
}

# lora_step  -- the decisive "one LoRA training step" gate, run with the
#   ACTIVE env's transformers+peft+cu128 torch. distilgpt2 keeps it tiny and
#   network-light; what we're really testing is the dependency+kernel stack.
lora_step() {
  python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import transformers, peft
print(f"    transformers {transformers.__version__}  peft {peft.__version__}")
tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").cuda()
model = get_peft_model(model, LoraConfig(r=8, target_modules=["c_attn"]))
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
batch = tok(["Emotion recognition for older adults."]*2, return_tensors="pt").to("cuda")
out = model(**batch, labels=batch["input_ids"]); out.loss.backward()
torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4).step()
print(f"    LoRA step ok: loss {out.loss.item():.3f}, {n:,} trainable params")
PY
}

# clone <url> <dir>
clone() {
  local url="$1" dir="$2"
  if [[ -d "$dir/.git" ]]; then ok "repo present: $dir"; git -C "$dir" pull --ff-only >/dev/null 2>&1 || true
  else echo "    cloning $url ..."; git clone --depth 1 "$url" "$dir" >/dev/null 2>&1 \
         || { fail "clone failed: $url"; return 1; }; ok "cloned -> $dir"; fi
}

# run_port  -- generic driver. args:
#   $1 name  $2 env  $3 python  $4 clone_url  $5 dir  $6 deps(space-sep, NO torch)
#   $7 real_train_cmd (printed, not run)
run_port() {
  local name="$1" env="$2" py="$3" url="$4" dir="$5" deps="$6" traincmd="$7"
  local log="$LOGDIR/$name.log"
  step "[$name] porting to cu128 / sm_120   (full log: $log)"
  : > "$log"

  clone "$url" "$dir"        2>&1 | tee -a "$log"
  make_env "$env" "$py"      2>&1 | tee -a "$log" || { VERDICTS+=("$name=FAIL(env)"); return; }
  install_cu128_torch        2>&1 | tee -a "$log" || { VERDICTS+=("$name=FAIL(torch)"); return; }

  # Install the repo's torch-adjacent pins on TOP of cu128 torch. We pass them
  # explicitly (torch/torchvision/torchaudio/cudatoolkit deliberately omitted)
  # so pip cannot pull the repo's old sm_120-less torch back in.
  echo "    installing repo deps (torch pins stripped): $deps" | tee -a "$log"
  if python -m pip install -q $deps 2>&1 | tee -a "$log"; then
    ok "repo deps installed"
  else
    fail "repo deps could not be installed alongside cu128 torch (pin conflict)"
    fail "-> this is itself a go/no-go signal: their stack resists cu128"
    VERDICTS+=("$name=FAIL(deps)");
    echo "    train cmd (for reference): $traincmd"
    return
  fi

  # confirm cu128 torch survived the repo-dep install (old pins sometimes
  # silently downgrade it)
  install_cu128_torch >/dev/null 2>&1
  local g=0 l=0
  gpu_smoke 2>&1 | tee -a "$log"; g=${PIPESTATUS[0]}
  lora_step 2>&1 | tee -a "$log"; l=${PIPESTATUS[0]}

  if [[ $g -eq 0 && $l -eq 0 ]]; then
    ok "$name: LoRA training step runs end-to-end on sm_120 with cu128 torch"
    VERDICTS+=("$name=PASS")
  elif [[ $l -eq 0 ]]; then
    warn "$name: LoRA step ran but GPU was not sm_120 (wrong card?)"
    VERDICTS+=("$name=PARTIAL(not-sm120)")
  else
    fail "$name: LoRA step failed under this repo's dependency stack"
    VERDICTS+=("$name=FAIL(lora)")
  fi
  echo "    NEXT (manual, needs weights/data): $traincmd" | tee -a "$log"
}

# =============================================================================
# Codebase 1: Emotion-LLaMA  (MiniGPT-v2 based; torch==2.0.0 in environment.yaml)
# =============================================================================
port_emotion_llama() {
  # torch-adjacent pins from environment.yaml, torch/torchvision/torchaudio/
  # cudatoolkit removed. peft==0.2.0 & transformers==4.30.0 are the real risk
  # against a modern cu128 torch -- that risk IS what we're measuring.
  local deps="transformers==4.30.0 peft==0.2.0 timm==0.6.13 omegaconf==2.3.0 \
iopath sentencepiece==0.1.99 accelerate==0.20.3 webdataset"
  run_port "emotion-llama" "port-emollama" "3.9" \
    "https://github.com/ZebangCheng/Emotion-LLaMA.git" \
    "$WORKDIR/Emotion-LLaMA" \
    "$deps" \
    "CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 train.py --cfg-path train_configs/Emotion-LLaMA_finetune.yaml  (needs Llama-2-7b-chat-hf + minigptv2_checkpoint.pth)"
}

# =============================================================================
# Codebase 2: TiCAL  (torch==2.3.1 / cu11.8 in requirements)
# =============================================================================
port_tical() {
  local deps="transformers==4.51.1 networkx==3.4.2 numpy==1.25.0 pandas==2.2.3 \
peft scikit-learn"
  run_port "tical" "port-tical" "3.10" \
    "https://github.com/yinwen2019/TiCAL.git" \
    "$WORKDIR/TiCAL" \
    "$deps" \
    "python train.py   (configure ./config/config.json + pick dataset in train.py; MOSI/MOSEI features into ./dataset)"
}

# =============================================================================
# Drive
# =============================================================================
case "$ONLY" in
  ""|all)            port_emotion_llama; port_tical ;;
  emotion-llama|emollama|emotionllama) port_emotion_llama ;;
  tical)             port_tical ;;
  *) echo "unknown --only '$ONLY' (use: emotion-llama | tical | all)" >&2; exit 2 ;;
esac

# =============================================================================
# Summary / go-no-go
# =============================================================================
echo
echo "============================================================"
echo "TASK 1.5 PORTING SUMMARY"
echo "============================================================"
for v in "${VERDICTS[@]}"; do printf "  %-14s %s\n" "${v%%=*}" "${v#*=}"; done
echo "------------------------------------------------------------"
allpass=1
for v in "${VERDICTS[@]}"; do [[ "${v#*=}" == "PASS" ]] || allpass=0; done
if [[ $allpass -eq 1 && ${#VERDICTS[@]} -gt 0 ]]; then
  echo "  GO: every ported codebase ran a LoRA step on sm_120 with cu128 torch."
  echo "      Reuse is viable. Next: wire weights/data and run each repo's real"
  echo "      train.py (commands printed above) on a tiny config / max_steps=1."
  echo "      ! Reserve a GPU on the Atlas sheet before that real run (shared 4-GPU box)."
else
  echo "  MIXED/NO-GO: at least one codebase's pinned stack did not reconcile"
  echo "      with cu128 torch on this GPU. Options per failing repo:"
  echo "        - bump its transformers/peft to a cu128-compatible version and"
  echo "          re-test (risks their code's API assumptions), OR"
  echo "        - reimplement the projector+LoRA recipe in the elder-mer env"
  echo "          (which already passed setup_atlas.sh's generic LoRA gate)."
  echo "      See per-codebase logs in: $LOGDIR"
fi
echo "============================================================"
