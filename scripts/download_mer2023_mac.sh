#!/bin/bash
# ============================================================
# MER2023 dataset downloader (macOS)
# Downloads the gated MERChallenge/MER2023 dataset from
# Hugging Face into ~/Desktop/MER2023_Data
#
# Prerequisites:
#   1. Your HF access request must already be APPROVED
#      (check: https://huggingface.co/settings/gated-repos)
#   2. You need a HF access token (read scope is enough):
#      https://huggingface.co/settings/tokens
#
# Usage:
#   chmod +x download_mer2023_mac.sh
#   ./download_mer2023_mac.sh
#
# The download is resumable: if it gets interrupted, just
# run the script again and it picks up where it left off.
# ============================================================

set -euo pipefail

REPO_ID="MERChallenge/MER2023"
# Project data layout (gitignored), consistent with crema_d / elderreact
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/mer2023"

echo "=== MER2023 Downloader ==="
echo "Destination: $DEST_DIR"
echo ""

# --- 1. Check Python ---
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install it first (brew install python)."
    exit 1
fi

# --- 2. Install/upgrade the Hugging Face CLI ---
echo "[1/4] Installing/updating huggingface_hub..."
python3 -m pip install -q -U "huggingface_hub[cli]"

# Resolve the CLI entry point (hf is current, huggingface-cli is legacy)
if command -v hf &>/dev/null; then
    HF_CMD="hf"
elif command -v huggingface-cli &>/dev/null; then
    HF_CMD="huggingface-cli"
else
    # Fall back to module invocation if scripts dir isn't on PATH
    HF_CMD="python3 -m huggingface_hub.commands.huggingface_cli"
fi

# --- 3. Authenticate if needed ---
echo "[2/4] Checking Hugging Face login..."
if ! $HF_CMD auth whoami &>/dev/null; then
    echo "Not logged in. Paste your HF token when prompted"
    echo "(create one at https://huggingface.co/settings/tokens):"
    $HF_CMD auth login
else
    echo "Already logged in as: $($HF_CMD auth whoami 2>/dev/null | head -n 1)"
fi

# --- 4. Download (caffeinate keeps the Mac awake) ---
echo "[3/4] Downloading $REPO_ID ..."
echo "This is large; the Mac will stay awake until it finishes."
mkdir -p "$DEST_DIR"

# Mac gets ONLY the small files (~280 MB): test sets, labels, and the
# password README. The train set is a 140 GB 7-part split zip (all parts
# required to extract) and must be downloaded directly on Atlas instead:
#   hf download MERChallenge/MER2023 --repo-type dataset \
#       --local-dir ~/data/MER2023
caffeinate -i $HF_CMD download "$REPO_ID" \
    --repo-type dataset \
    --local-dir "$DEST_DIR" \
    --include "mer2023test1&2.zip" "test-labels.zip" \
              "README_AFTER_APPROVAL.md" "README.md"

# --- 5. Post-download instructions ---
echo ""
echo "[4/4] Download complete. Files are in: $DEST_DIR"
echo ""
echo "NEXT STEPS:"
echo "  1. Open $DEST_DIR/README_AFTER_APPROVAL.md"
echo "     It contains the password for the compressed archives."
echo "  2. Unzip with the password, e.g.:"
echo "       cd \"$DEST_DIR\""
echo "       unzip -P 'PASSWORD_HERE' archive_name.zip"
echo "     (If the archives are .rar or 7z-encrypted, install p7zip:"
echo "       brew install p7zip"
echo "       7z x archive_name.zip -p'PASSWORD_HERE')"
echo ""
echo "  3. The 140 GB train set is NOT downloaded here. Get it directly on Atlas:"
echo "       hf download MERChallenge/MER2023 --repo-type dataset --local-dir ~/data/MER2023"
echo ""
echo "Done."
