#!/bin/bash
# =============================================================================
# Stage 1 — Direct Evaluation of Method 1 (Temporal Aggregation)
# Loads pretrain.model into ECAPA_TDNN_A and evaluates on VoxCeleb1-O
# No training needed. Runtime: ~10 minutes on GPU.
#
# Run from project root:
#   cd /mnt/c/Users/Harini/Documents/ECAPA-TDNN-1
#   bash training_v2/run_eval_wsl.sh
# =============================================================================

set -e  # Exit on any error

# ── Project root (WSL path) ───────────────────────────────────────────────────
PROJECT_ROOT="/mnt/c/Users/Harini/Documents/ECAPA-TDNN-1"
cd "$PROJECT_ROOT"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ecapa

echo "=============================================="
echo "  ECAPA-TDNN Method 1 — Stage 1: Direct Eval"
echo "  Model: exps/pretrain.model"
echo "  Eval:  Voxceleb/veri_test2.txt"
echo "  Expected baseline EER: ~0.96%"
echo "=============================================="

python training_v2/trainECAPAModel_v2.py \
    --eval \
    --initial_model exps/pretrain.model \
    --eval_list     Voxceleb/veri_test2.txt \
    --eval_path     Voxceleb/

echo ""
echo "Stage 1 complete. Compare EER to baseline 0.96%."
