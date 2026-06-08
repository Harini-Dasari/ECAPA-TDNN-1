import os
import sys
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

sys.path.append(os.getcwd())

# Import robust VAD and remapper from our previous pipeline
from xai_reddots.scripts.plot_individual_recordings import detect_speech_span, remap_timeline

SR = 16000
HOP = 160

def plot_temporal_importance(speaker, phrase_key, recording_id, pcm_path, output_path):
    """
    Plots the waveform, phoneme boundaries, and temporal alpha_hat curve.
    """
    # 1. Load the temporal attention curve
    alpha_hat = []
    time_axis = []
    
    with open('xai_reddots_temporal/outputs/temporal_alpha.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['recording_id'] == recording_id:
                alpha_hat.append(float(row['alpha_hat']))
                time_axis.append(float(row['time']))
                
    if not alpha_hat:
        print(f"No temporal data found for {recording_id}")
        return
        
    alpha_hat = np.array(alpha_hat)
    time_axis = np.array(time_axis)
    
    # 2. Load Audio
    audio, _ = sf.read(pcm_path, channels=1, samplerate=SR, subtype='PCM_16', format='RAW')
    audio_time = np.linspace(0, len(audio) / SR, len(audio))
    
    # 3. Load MFA timeline and remap to audio
    mfa_path = f'xai_reddots/metadata/{speaker}_{phrase_key}_timeline.json'
    if os.path.exists(mfa_path):
        with open(mfa_path) as f:
            mfa_tdata = json.load(f)
            
        # Remap using our exact logic from the entropy pipeline
        speech_start, speech_end = detect_speech_span(audio, sr=SR, hop=HOP)
        tdata = remap_timeline(mfa_tdata, speech_start, speech_end)
        phonemes = tdata.get('phonemes', [])
    else:
        print(f"Warning: No MFA timeline found at {mfa_path}")
        phonemes = []

    # 4. Create Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1.5]})
    
    # --- Top Plot: Waveform & Phonemes ---
    ax1.plot(audio_time, audio, color='gray', alpha=0.7)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Temporal XAI Analysis: {recording_id} ({phrase_key})")
    ax1.margins(x=0)
    
    # Draw phonemes
    for ph in phonemes:
        s, e = float(ph['start']), float(ph['end'])
        label = ph['phoneme']
        # Draw boundaries
        ax1.axvline(s, color='blue', linestyle='--', alpha=0.4)
        ax2.axvline(s, color='blue', linestyle='--', alpha=0.4)
        
        # Draw label in the center of the phoneme span
        mid_t = (s + e) / 2
        ax1.text(mid_t, ax1.get_ylim()[1]*0.8, label, color='blue', 
                 ha='center', va='center', fontsize=10, rotation=0, 
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    # --- Bottom Plot: Temporal Importance ---
    ax2.plot(time_axis, alpha_hat, color='red', linewidth=2, label="α̂(t)")
    ax2.fill_between(time_axis, 0, alpha_hat, color='red', alpha=0.2)
    ax2.set_ylabel("Temporal Importance")
    ax2.set_xlabel("Time (seconds)")
    ax2.legend(loc='upper right')
    ax2.margins(x=0)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")

def main():
    metadata_csv = 'xai_reddots/metadata/phrase_groups.csv'
    
    if not os.path.exists(metadata_csv):
        print("Metadata not found. Please run batch_temporal.py first, or ensure paths are correct.")
        return
        
    records = []
    with open(metadata_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['speaker_id'] == 'm0002':
                records.append(row)
                
    # Visualize first 3
    for idx, row in enumerate(records[:3]):
        rec_id = row['recording_id']
        pcm_path = row['pcm_path']
        output_path = f"xai_reddots_temporal/plots/{rec_id}_temporal.png"
        
        plot_temporal_importance(
            speaker='m0002',
            phrase_key='my_voice_is_my_password',
            recording_id=rec_id,
            pcm_path=pcm_path,
            output_path=output_path
        )

if __name__ == "__main__":
    main()
