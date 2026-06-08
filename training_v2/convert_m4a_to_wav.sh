#!/bin/bash
# =============================================================================
# convert_m4a_to_wav.sh — Convert VoxCeleb2 .m4a files to .wav (16kHz mono)
#
# Needed if you downloaded the Kaggle version (e1lephant/voxceleb2) which
# comes as .m4a files. The dataLoader expects .wav format.
#
# Prerequisites (install once in WSL):
#   sudo apt-get install -y ffmpeg
#
# Usage:
#   bash training_v2/convert_m4a_to_wav.sh /path/to/voxceleb2/aac
#
# This will:
#   - Walk through all idXXXXX/videoID/*.m4a files
#   - Convert each to 16kHz mono .wav alongside the .m4a
#   - Keep originals (safe to delete afterwards to save space)
# =============================================================================

set -e

AAC_DIR="${1:-/mnt/c/Users/Harini/Documents/ECAPA-TDNN-1/Voxceleb-2/aac}"

if [ ! -d "$AAC_DIR" ]; then
    echo "ERROR: Directory not found: $AAC_DIR"
    echo "Usage: bash training_v2/convert_m4a_to_wav.sh /path/to/aac"
    exit 1
fi

echo "Converting .m4a → .wav (16kHz mono)"
echo "Source: $AAC_DIR"

TOTAL=$(find "$AAC_DIR" -name "*.m4a" | wc -l)
echo "Found $TOTAL .m4a files to convert..."
echo ""

COUNT=0
find "$AAC_DIR" -name "*.m4a" | while read -r m4a_file; do
    wav_file="${m4a_file%.m4a}.wav"
    if [ ! -f "$wav_file" ]; then
        ffmpeg -y -i "$m4a_file" -ac 1 -ar 16000 "$wav_file" -loglevel error
    fi
    COUNT=$((COUNT + 1))
    if [ $((COUNT % 1000)) -eq 0 ]; then
        echo "  Converted $COUNT / $TOTAL ..."
    fi
done

echo ""
echo "✓ Conversion complete!"
echo "Now run: python training_v2/prepare_train_list.py --wav_subdir aac"
