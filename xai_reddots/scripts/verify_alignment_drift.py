import os
import sys
import json
import csv
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuration
SPEAKER = 'm0004'
PHRASE_KEY = 'my_voice_is_my_password'
SR = 16000
HOP = 160
WIN = 400

def pcm_duration(path):
    try:
        audio, _ = sf.read(path, channels=1, samplerate=SR, subtype='PCM_16', format='RAW')
        return len(audio) / SR, audio
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None, None

def rms_envelope(audio):
    hw = WIN // 2
    n = len(audio) // HOP
    out = np.zeros(n)
    for i in range(n):
        c = i * HOP
        seg = audio[max(0, c - hw):min(len(audio), c + hw)]
        out[i] = np.sqrt(np.mean(seg ** 2)) if len(seg) > 0 else 0.0
    return out

def detect_speech_boundaries(audio, sr=16000, threshold=0.04):
    hw = WIN // 2
    hop = HOP
    n = len(audio) // hop
    rms = np.zeros(n)
    for i in range(n):
        c = i * hop
        seg = audio[max(0, c - hw):min(len(audio), c + hw)]
        rms[i] = np.sqrt(np.mean(seg ** 2)) if len(seg) > 0 else 0.0
    rms_norm = rms / (np.max(rms) + 1e-9)
    rms_t = np.linspace(0, len(audio)/sr, len(rms_norm))
    
    speech_indices = np.where(rms_norm > threshold)[0]
    if len(speech_indices) == 0:
        return 0.0, len(audio)/sr
        
    start_time = rms_t[speech_indices[0]]
    end_time = rms_t[speech_indices[-1]]
    return start_time, end_time

def main():
    metadata_csv = f'xai_reddots/metadata/separated_phrases/{PHRASE_KEY}.csv'
    timeline_json = f'xai_reddots/metadata/{SPEAKER}_{PHRASE_KEY}_timeline.json'
    
    if not os.path.exists(metadata_csv):
        print(f"Metadata CSV not found: {metadata_csv}")
        return
        
    if not os.path.exists(timeline_json):
        print(f"Timeline JSON not found: {timeline_json}")
        return
        
    # Load template timeline
    with open(timeline_json, 'r') as f:
        tdata = json.load(f)
    template_phonemes = tdata.get('phonemes', [])
    orig_dur = float(tdata.get('avg_duration_sec', 3.791333))
    
    # Load recordings
    recs = []
    with open(metadata_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['speaker_id'] == SPEAKER and os.path.exists(row['pcm_path']):
                recs.append(row['pcm_path'])
                
    if not recs:
        print(f"No recordings found for speaker {SPEAKER}!")
        return
        
    print(f"Found {len(recs)} recordings. Processing envelopes and alignments...")
    
    # Process each recording
    processed_data = []
    for idx, path in enumerate(recs, 1):
        dur, audio = pcm_duration(path)
        if audio is None:
            continue
            
        # Compute RMS envelope
        rms = rms_envelope(audio)
        rms_norm = rms / (np.max(rms) + 1e-9)
        rms_t = np.linspace(0, dur, len(rms_norm))
        
        # Detect actual speech start and end times in the file
        t_start_file, t_end_file = detect_speech_boundaries(audio, threshold=0.04)
        
        t_start_template = template_phonemes[0]['start']
        t_end_template = template_phonemes[-1]['end']
        template_speech_dur = t_end_template - t_start_template
        file_speech_dur = t_end_file - t_start_file
        
        scaled_phonemes = []
        for ph in template_phonemes:
            # Normalize relative to template speech segment
            norm_start = (ph['start'] - t_start_template) / template_speech_dur
            norm_end = (ph['end'] - t_start_template) / template_speech_dur
            
            # Scale to fit actual speech segment in the recording
            scaled_phonemes.append({
                'phoneme': ph['phoneme'],
                'word': ph['word'],
                'start': t_start_file + norm_start * file_speech_dur,
                'end': t_start_file + norm_end * file_speech_dur,
            })
            
        processed_data.append({
            'recording_id': os.path.basename(path).replace('.pcm', ''),
            'duration': dur,
            'rms_norm': rms_norm,
            'rms_t': rms_t,
            'phonemes': scaled_phonemes
        })
        
    # Plotting
    fig, axes = plt.subplots(len(processed_data), 2, figsize=(18, 12), facecolor='#f8f9fa')
    fig.suptitle(f"Alignment Drift Analysis • Speaker {SPEAKER} • \"My voice is my password\"", 
                 fontsize=14, fontweight='bold', y=0.98)
                 
    word_colors = ['#d1e7dd', '#fff3cd', '#f8d7da', '#cff4fc', '#eaddf5']
    
    for i, data in enumerate(processed_data):
        rec_name = data['recording_id']
        dur = data['duration']
        rms_norm = data['rms_norm']
        rms_t = data['rms_t']
        ph_list = data['phonemes']
        
        # ---------------- COLUMN 1: ABSOLUTE TIME ----------------
        ax_abs = axes[i, 0]
        ax_abs.set_facecolor('white')
        ax_abs.plot(rms_t, rms_norm, color='#0072BD', lw=1.5, zorder=3)
        ax_abs.fill_between(rms_t, 0, rms_norm, color='#e2f0f9', alpha=0.6, zorder=2)
        
        # Draw word boundaries
        word_bounds = {}
        for ph in ph_list:
            w = ph['word']
            if w not in word_bounds:
                word_bounds[w] = [ph['start'], ph['end']]
            else:
                word_bounds[w][0] = min(word_bounds[w][0], ph['start'])
                word_bounds[w][1] = max(word_bounds[w][1], ph['end'])
                
        for wi, (word, (ws, we)) in enumerate(sorted(word_bounds.items(), key=lambda x: x[1][0])):
            rect = patches.Rectangle((ws, 0), we - ws, 1.0, 
                                     facecolor=word_colors[wi % len(word_colors)], alpha=0.3, zorder=1)
            ax_abs.add_patch(rect)
            ax_abs.text((ws + we)/2, 0.9, word, ha='center', va='center', fontsize=9, fontweight='bold', color='#444')
            
        # Draw phoneme boundary lines
        for ph in ph_list:
            ax_abs.axvline(x=ph['end'], color='#c0392b', linestyle='--', linewidth=0.8, alpha=0.7, zorder=4)
            cx = (ph['start'] + ph['end']) / 2
            ax_abs.text(cx, 0.15, ph['phoneme'], color='#c0392b', ha='center', va='bottom', fontsize=8, fontweight='bold', zorder=5)
            
        ax_abs.set_xlim(0, 4.5)  # Align x-axis to see absolute shifts
        ax_abs.set_ylim(0, 1.05)
        ax_abs.set_ylabel(f"Rec {i+1}\n(dur: {dur:.2f}s)", fontsize=9, fontweight='bold', rotation=0, labelpad=30, va='center')
        ax_abs.grid(True, linestyle='--', alpha=0.5, color='#dddddd')
        
        if i == 0:
            ax_abs.set_title("Absolute Time Alignment (RMS Energy vs. Scaled Boundaries)", fontsize=11, fontweight='bold', pad=10)
        if i == len(processed_data) - 1:
            ax_abs.set_xlabel("Time (seconds)", fontsize=10)
            
        # ---------------- COLUMN 2: NORMALIZED TIME ----------------
        ax_norm = axes[i, 1]
        ax_norm.set_facecolor('white')
        
        # Draw normalized phoneme timelines
        # Normalized start/end is ph['start'] / dur
        for wi, (word, (ws, we)) in enumerate(sorted(word_bounds.items(), key=lambda x: x[1][0])):
            n_ws, n_we = ws / dur, we / dur
            rect = patches.Rectangle((n_ws, 0.1), n_we - n_ws, 0.8, 
                                     facecolor=word_colors[wi % len(word_colors)], edgecolor='gray', alpha=0.7, zorder=1)
            ax_norm.add_patch(rect)
            ax_norm.text((n_ws + n_we)/2, 0.5, f"{word}\n({word.upper()})", ha='center', va='center', fontsize=8, fontweight='bold')
            
        for ph in ph_list:
            n_end = ph['end'] / dur
            ax_norm.axvline(x=n_end, color='#444444', linestyle='-', linewidth=0.6, zorder=2)
            n_cx = (ph['start'] / dur + ph['end'] / dur) / 2
            ax_norm.text(n_cx, 0.85, ph['phoneme'], color='#c0392b', ha='center', va='top', fontsize=8, fontweight='bold')
            
        ax_norm.set_xlim(0, 1.0)
        ax_norm.set_ylim(0, 1.0)
        ax_norm.set_yticks([])
        
        if i == 0:
            ax_norm.set_title("Normalized Time Timeline (Proving 100% Identical Relative Spacing)", fontsize=11, fontweight='bold', pad=10)
        if i == len(processed_data) - 1:
            ax_norm.set_xlabel("Normalized Time (0.0 to 1.0)", fontsize=10)
            
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the output figures
    out_dir = 'xai_reddots/plots'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'alignment_drift_verification.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save to plots_individual
    indiv_dir = f'xai_reddots/results/phrase1_{PHRASE_KEY}/plots_individual'
    os.makedirs(indiv_dir, exist_ok=True)
    indiv_path = os.path.join(indiv_dir, 'alignment_drift_verification.png')
    import shutil
    shutil.copy2(out_path, indiv_path)
    
    print(f"Verification plot successfully saved to:\n  - {out_path}\n  - {indiv_path}")

if __name__ == '__main__':
    main()
