"""
regen_recs245.py
================
Regenerates only recordings 2, 4, and 5 for m0004 / my_voice_is_my_password
using the improved two-threshold VAD that skips the DC artifact at t=0.
"""
import os, sys, json, csv, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from xai_reddots.scripts.plot_individual_recordings import (
    detect_speech_span, remap_timeline, aggregate_single,
    SPEAKER, PHRASE_KEY, PHRASE_IDX, PHRASE_DEF, CSV_META, MFA_TIMELINE, OUTPUT_DIR
)
from xai_reddots.scripts.final_figure import load_model, build_figure, SR, HOP
import soundfile as sf

# Only regenerate these recording indices (1-based)
REGEN = {2, 4, 5}

def main():
    with open(MFA_TIMELINE) as f:
        mfa_tdata = json.load(f)

    ph0 = mfa_tdata['phonemes'][0]
    phn = mfa_tdata['phonemes'][-1]
    print(f"MFA timeline: {len(mfa_tdata['phonemes'])} phonemes  "
          f"span={ph0['start']:.3f}s → {phn['end']:.3f}s")

    print("Loading model...")
    model = load_model()

    recs = []
    with open(CSV_META) as f:
        for row in csv.DictReader(f):
            if row['speaker_id'] == SPEAKER and os.path.exists(row['pcm_path']):
                recs.append(row['pcm_path'])

    n = len(recs)
    for idx, pcm_path in enumerate(recs, 1):
        if idx not in REGEN:
            print(f"Skipping Rec {idx}")
            continue

        print(f"\n── Rec {idx}/{n}: {os.path.basename(pcm_path)}")
        audio, _ = sf.read(pcm_path, channels=1, samplerate=SR,
                           subtype='PCM_16', format='RAW')
        dur = len(audio) / SR
        print(f"   Duration: {dur:.3f}s")

        speech_start, speech_end = detect_speech_span(audio, sr=SR, hop=HOP)
        print(f"   Speech span detected: {speech_start:.3f}s → {speech_end:.3f}s")

        tdata = remap_timeline(mfa_tdata, speech_start, speech_end)
        tdata['avg_duration_sec'] = dur

        first_ph = tdata['phonemes'][0]
        last_ph  = tdata['phonemes'][-1]
        print(f"   Remapped: {first_ph['phoneme']} at {first_ph['start']:.3f}s → "
              f"{last_ph['phoneme']} at {last_ph['end']:.3f}s")

        d = aggregate_single(pcm_path, model, tdata)

        phrase_title = PHRASE_DEF['display'] + f" (Rec {idx}/{n})"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fig = build_figure(SPEAKER, d, phrase_title)

        fig_path = os.path.join(OUTPUT_DIR, f"{SPEAKER}_rec{idx:02d}_final_figure.png")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        print(f"   ✓ Saved: {fig_path}")

    print("\nAll done!")

if __name__ == '__main__':
    main()
