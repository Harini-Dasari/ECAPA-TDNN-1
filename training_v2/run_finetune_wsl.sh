#!/bin/bash
# =============================================================================
# Stage 2 — Fine-Tuning Method 1 (Temporal Aggregation) on VoxCeleb2
#
# BEFORE RUNNING: Edit the DATA_ROOT variable below to point to your
#                 downloaded VoxCeleb2 dataset folder.
#
# Run from project root:
#   cd /mnt/c/Users/Harini/Documents/ECAPA-TDNN-1
#   bash training_v2/run_finetune_wsl.sh
#
# Expected runtime: ~6-10 hours for 15 epochs on a single GPU
# =============================================================================

set -e

# ── EDIT THIS → path to your downloaded VoxCeleb2 dataset ────────────────────
# After download, the structure should be:
#   DATA_ROOT/
#     train_list.txt
#     aac/idXXXXX/videoID/utterance.m4a   (or .wav after conversion)
#
# If your data is on Windows, use the WSL mount path:
#   e.g. /mnt/d/datasets/voxceleb2
DATA_ROOT="/mnt/c/Users/Harini/Documents/ECAPA-TDNN-1/Voxceleb-2"

# Augmentation data (optional — comment out --musan_path/--rir_path if not available)
# MUSAN_PATH="/mnt/d/datasets/musan_split"
# RIR_PATH="/mnt/d/datasets/RIRS_NOISES/simulated_rirs"

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT="/mnt/c/Users/Harini/Documents/ECAPA-TDNN-1"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "  ECAPA-TDNN Method 1 — Stage 2: Fine-Tuning"
echo "  Initial: exps/pretrain.model"
echo "  Data:    $DATA_ROOT"
echo "  Save:    training_v2/exps_modelA/"
echo "  LR:      0.0001 (10x lower than original)"
echo "  Epochs:  15"
echo "=============================================="

# Check dataset exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: DATA_ROOT not found: $DATA_ROOT"
    echo "Please edit DATA_ROOT in this script to point to your VoxCeleb2 folder."
    exit 1
fi

# Check train_list.txt exists
TRAIN_LIST="$DATA_ROOT/train_list.txt"
if [ ! -f "$TRAIN_LIST" ]; then
    echo "WARNING: train_list.txt not found at $TRAIN_LIST"
    echo "You may need to generate it. See: training_v2/prepare_train_list.py"
fi

python training_v2/trainECAPAModel_v2.py \
    --initial_model exps/pretrain.model \
    --save_path     training_v2/exps_modelA \
    --train_list    "$TRAIN_LIST" \
    --train_path    "$DATA_ROOT/wav" \
    --eval_list     Voxceleb/veri_test2.txt \
    --eval_path     Voxceleb/ \
    --lr            0.0001 \
    --lr_decay      0.90 \
    --max_epoch     15 \
    --test_step     1 \
    --batch_size    200 \
    --n_cpu         4
    # --musan_path  "$MUSAN_PATH" \
    # --rir_path    "$RIR_PATH"

echo ""
echo "Fine-tuning complete!"
echo "Best model is in: training_v2/exps_modelA/model/"
echo "Training log:     training_v2/exps_modelA/score.txt"
