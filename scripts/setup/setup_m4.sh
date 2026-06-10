#!/usr/bin/env bash
# =============================================================================
# setup_m4.sh  --  Development environment on the M4 MacBook Pro (Apple Silicon)
#
# Installs the MPS (Metal) build of PyTorch plus the shared requirements.
# Use this machine for: coding, debugging, small tests, Whisper inference,
# feature extraction. Real training happens on Atlas (see setup_atlas.sh).
#
# Usage:
#   bash scripts/setup/setup_m4.sh
#
# Prereqs: Miniconda/Anaconda installed, and Homebrew (for ffmpeg).
# =============================================================================
set -euo pipefail

ENV_NAME="elder-mer"
PY_VERSION="3.11"

echo "==> [M4] Setting up '${ENV_NAME}' (Apple Silicon / MPS)"

# --- sanity: are we actually on Apple Silicon? ---
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "WARNING: uname -m is not arm64. This script targets Apple Silicon (M-series)." >&2
  echo "         If you are on Atlas or an Intel Mac, use the correct setup script." >&2
fi

# --- conda must be available ---
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda first, then re-run." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- ffmpeg (required by Whisper) via Homebrew ---
if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "==> Installing ffmpeg via Homebrew"
    brew install ffmpeg
  else
    echo "WARNING: ffmpeg not found and Homebrew not installed." >&2
    echo "         Install ffmpeg manually or Whisper will fail at runtime." >&2
  fi
fi

# --- create / reuse the env ---
if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "==> Env '${ENV_NAME}' already exists; reusing it."
else
  echo "==> Creating conda env '${ENV_NAME}' (python ${PY_VERSION})"
  conda create -n "${ENV_NAME}" "python=${PY_VERSION}" -y
fi
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel

# --- PyTorch: default wheels include MPS support on Apple Silicon ---
echo "==> Installing PyTorch (MPS build for Apple Silicon)"
pip install torch torchvision torchaudio

# --- shared requirements ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "==> Installing common requirements"
pip install -r "${SCRIPT_DIR}/requirements-common.txt"

# --- Whisper (separate: its build expects setuptools already present) ---
echo "==> Installing openai-whisper"
pip install --no-build-isolation openai-whisper==20240930

# --- verify MPS is actually available ---
echo "==> Verifying PyTorch / MPS"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
if not torch.backends.mps.is_available():
    print("WARNING: MPS not available; training-test ops will fall back to CPU.")
PY

cat <<'EOF'

==> [M4] Done.
    Activate with:  conda activate elder-mer
    Reminders:
      - OpenFace (video action units) is a separate C++ toolkit, not pip.
        On macOS build from source: https://github.com/TadasBaltrusaitis/OpenFace
        (feature extraction can also be run on Atlas instead.)
      - Run `wandb login` once before your first tracked experiment.
EOF
