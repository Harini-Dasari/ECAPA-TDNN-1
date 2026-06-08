"""
prepare_train_list.py — Generate VoxCeleb2 train_list.txt from your dataset folder.

This script scans your VoxCeleb2 directory for .wav (or .m4a) audio files and
produces a train_list.txt in the exact format expected by dataLoader.py:

    id00012/21Uxsk56VDQ/00001.wav
    id00012/21Uxsk56VDQ/00002.wav
    ...

Usage (from project root in WSL):
    python training_v2/prepare_train_list.py \\
        --data_path /mnt/c/Users/Harini/Documents/ECAPA-TDNN-1/Voxceleb-2 \\
        --ext wav

If your files are .m4a (Kaggle download), convert first:
    bash training_v2/convert_m4a_to_wav.sh /path/to/voxceleb2

Output: <data_path>/train_list.txt
"""

import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Generate VoxCeleb2 train_list.txt')
parser.add_argument('--data_path', type=str,
                    default='/mnt/c/Users/Harini/Documents/ECAPA-TDNN-1/Voxceleb-2',
                    help='Root path of VoxCeleb2 dataset')
parser.add_argument('--ext',       type=str, default='wav',
                    help='Audio file extension: wav or m4a')
parser.add_argument('--wav_subdir', type=str, default='wav',
                    help='Subdirectory under data_path containing speaker folders (e.g. "wav" or "aac")')
args = parser.parse_args()

audio_root = Path(args.data_path) / args.wav_subdir
out_path   = Path(args.data_path) / 'train_list.txt'

print(f"Scanning: {audio_root}")
print(f"Looking for: .{args.ext} files")

entries = []
for speaker_dir in sorted(audio_root.iterdir()):
    if not speaker_dir.is_dir():
        continue
    speaker_id = speaker_dir.name
    for video_dir in sorted(speaker_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        video_id = video_dir.name
        for wav_file in sorted(video_dir.glob(f'*.{args.ext}')):
            rel_path = f"{speaker_id}/{video_id}/{wav_file.name}"
            entries.append(f"{speaker_id} {rel_path}")

if not entries:
    print(f"\nERROR: No .{args.ext} files found under {audio_root}")
    print("Please check --data_path and --wav_subdir arguments.")
    print("If files are .m4a, run: bash training_v2/convert_m4a_to_wav.sh")
    exit(1)

with open(out_path, 'w') as f:
    f.write('\n'.join(entries))

# Count unique speakers
speakers = set(e.split('/')[0] for e in entries)
print(f"\n✓ Found {len(speakers)} speakers, {len(entries)} utterances")
print(f"✓ Written to: {out_path}")
print(f"\nNext step: bash training_v2/run_finetune_wsl.sh")
