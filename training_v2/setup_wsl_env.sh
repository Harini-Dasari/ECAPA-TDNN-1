#!/bin/bash
# =============================================================================
# setup_wsl_env.sh — One-time environment setup for training in WSL
#
# Run this ONCE to set up your Python/conda environment in WSL:
#   bash training_v2/setup_wsl_env.sh
# =============================================================================

set -e

echo "=============================================="
echo "  ECAPA-TDNN WSL Environment Setup"
echo "=============================================="

# ── Step 1: Check conda is available ─────────────────────────────────────────
if ! command -v conda &> /dev/null; then
    echo ""
    echo "conda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    echo "Miniconda installed. Please restart your terminal and run this script again."
    exit 0
fi

echo "✓ conda found: $(conda --version)"

# ── Step 2: Create ECAPA environment ─────────────────────────────────────────
ENV_NAME="ECAPA"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "✓ Environment '$ENV_NAME' already exists — skipping creation"
else
    echo ""
    echo "Creating conda environment: $ENV_NAME (Python 3.7.9)..."
    conda create -n "$ENV_NAME" python=3.7.9 -y
    echo "✓ Environment created"
fi

# ── Step 3: Activate and install packages ────────────────────────────────────
echo ""
echo "Installing packages..."

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Install PyTorch (CUDA 11.0 build — matches original requirements.txt)
# If you have a different CUDA version, adjust the URL below
pip install torch==1.7.1+cu110 torchaudio==0.7.2 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install other requirements
pip install numpy scipy scikit-learn tqdm soundfile

# ── Step 4: Check ffmpeg for audio conversion ─────────────────────────────────
echo ""
if command -v ffmpeg &> /dev/null; then
    echo "✓ ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "Installing ffmpeg (needed for .m4a → .wav conversion)..."
    sudo apt-get update -qq && sudo apt-get install -y ffmpeg
    echo "✓ ffmpeg installed"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Setup complete!"
echo ""
echo "  To activate environment:"
echo "    conda activate ECAPA"
echo ""
echo "  Then go to project root:"
echo "    cd /mnt/c/Users/Harini/Documents/ECAPA-TDNN-1"
echo ""
echo "  When dataset is ready, run:"
echo "    bash training_v2/run_eval_wsl.sh      ← Stage 1 (eval only, ~10 min)"
echo "    bash training_v2/run_finetune_wsl.sh  ← Stage 2 (fine-tuning, ~6-10h)"
echo "=============================================="
