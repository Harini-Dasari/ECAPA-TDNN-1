import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import matplotlib.patches as patches

def main():
    timeline_json = 'xai_reddots/metadata/timeline.json'
    phrase_csv = 'xai_reddots/entropy/phrase_entropy.csv'
    groups_csv = 'xai_reddots/metadata/phrase_groups.csv'
    
    # Read speaker ID and phrase dynamically to construct paths
    with open(groups_csv, 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)
        speaker_id = row['speaker_id']
        phrase = row['phrase']
    phrase_clean = phrase.lower().replace(" ", "_").replace('"', '')
    phoneme_csv = f'xai_reddots/entropy/{speaker_id}_{phrase_clean}_ecapa_phoneme_ranked.csv'
    output_png = f'xai_reddots/plots/{speaker_id}_entropy_dashboard.png'
    
    # 1. Load the representative recording (first in phrase_groups)
    with open(groups_csv, 'r') as f:
        recs = list(csv.DictReader(f))
        rep_rec = recs[0]
        total_recs = len(recs)
    
    pcm_path = rep_rec['pcm_path']
    audio, sr = sf.read(pcm_path, channels=1, samplerate=16000, subtype='PCM_16', format='RAW')
    audio = audio / np.max(np.abs(audio))
    audio_time = np.linspace(0, len(audio)/sr, len(audio))
    
    # 2. Load Timeline JSON
    with open(timeline_json, 'r') as f:
        timeline_data = json.load(f)
    phonemes = timeline_data.get('phonemes', [])
    
    # Extract unique word boundaries
    word_bounds = {}
    for ph in phonemes:
        w_idx = ph['word_index']
        if w_idx not in word_bounds:
            word_bounds[w_idx] = {'word': ph['word'], 'start': float(ph['word_start']), 'end': float(ph['word_end'])}
    word_bounds_list = [word_bounds[i] for i in sorted(word_bounds.keys())]
    
    # 3. Load Phoneme Entropy Stats
    phoneme_stats = []
    with open(phoneme_csv, 'r') as f:
        for row in csv.DictReader(f):
            phoneme_stats.append({
                'phoneme': row['phoneme'],
                'word': row['word'],
                'start': float(row['start']),
                'end': float(row['end']),
                'mean_entropy': float(row['mean_attention']),
                'max_entropy': float(row['max_attention'])
            })
            
    # 4. Load Phrase Entropy Profile
    times, means, stds = [], [], []
    with open(phrase_csv, 'r') as f:
        for row in csv.DictReader(f):
            times.append(float(row['time']))
            means.append(float(row['mean_entropy']))
            stds.append(float(row['std_entropy']))
    times, means, stds = np.array(times), np.array(means), np.array(stds)
    
    # STYLING SETUP
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    bg_color = '#f8f9fa'
    purple_line = '#7030a0'
    purple_fill = '#eaddf5'
    grid_color = '#e0e0e0'
    
    fig = plt.figure(figsize=(16, 24), facecolor=bg_color)
    fig.suptitle(f"Phase-2 ECAPA Analysis • Speaker {rep_rec['speaker_id']} • \"{rep_rec['phrase']}\" • {total_recs} recordings", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 6 rows
    gs = fig.add_gridspec(6, 1, height_ratios=[1.5, 1.5, 2, 2, 2, 3.5], hspace=0.6)
    max_time = times[-1] if len(times) > 0 else 2.0
    
    pastel_colors = ['#d1e7dd', '#fff3cd', '#f8d7da', '#cff4fc', '#e2e3e5']
    phoneme_colors = ['#f8d7da', '#cff4fc', '#d1e7dd', '#fff3cd', '#eaddf5']
    
    # ================= ROW 1: WORD TIMELINE =================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(bg_color)
    ax1.set_xlim(0, max_time)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title("Word Timeline (WhisperX alignment)", fontsize=10, fontweight='bold', loc='left', pad=10)
    
    for i, wb in enumerate(word_bounds_list):
        rect_color = pastel_colors[i % len(pastel_colors)]
        rect = patches.Rectangle((wb['start'], 0.1), wb['end'] - wb['start'], 0.8, 
                                 linewidth=0.5, edgecolor='gray', facecolor=rect_color)
        ax1.add_patch(rect)
        
        center_x = (wb['start'] + wb['end']) / 2
        ax1.text(center_x, 0.65, wb['word'], ha='center', va='center', fontsize=10, fontweight='bold')
        ax1.text(center_x, 0.35, f"{wb['start']:.2f}s - {wb['end']:.2f}s", 
                 ha='center', va='center', fontsize=8, color='#444')
                 
    # ================= ROW 2: PHONEME TIMELINE =================
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.set_facecolor(bg_color)
    ax2.set_xlim(0, max_time)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title("Phoneme Timeline (WhisperX -> CMU phonemes)", fontsize=10, fontweight='bold', loc='left', pad=10)
    
    for i, ph in enumerate(phoneme_stats):
        rect_color = phoneme_colors[i % len(phoneme_colors)]
        rect = patches.Rectangle((ph['start'], 0.1), ph['end'] - ph['start'], 0.8, 
                                 linewidth=0.5, edgecolor='gray', facecolor=rect_color)
        ax2.add_patch(rect)
        
        center_x = (ph['start'] + ph['end']) / 2
        ax2.text(center_x, 0.65, ph['phoneme'], ha='center', va='center', fontsize=9, fontweight='bold')
        ax2.text(center_x, 0.35, f"{ph['start']:.2f}s\n{ph['end']:.2f}s", 
                 ha='center', va='center', fontsize=7, color='#444')

    def format_ax(ax, title, ylabel):
        ax.set_facecolor('white')
        ax.grid(True, linestyle='--', color=grid_color, alpha=0.7)
        ax.set_xlim(0, max_time)
        ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        for spine in ax.spines.values():
            spine.set_color('#cccccc')

    def add_phoneme_boundaries(ax):
        for ph in phoneme_stats:
            ax.axvline(x=ph['end'], color='red', linestyle='--', linewidth=0.8, alpha=0.6)
            center_x = (ph['start'] + ph['end']) / 2
            ax.text(center_x, ax.get_ylim()[1]*0.95, ph['phoneme'], color='red', 
                    ha='center', va='top', fontsize=7, alpha=0.7)
        # Also add start of first phoneme
        if phoneme_stats:
            ax.axvline(x=phoneme_stats[0]['start'], color='red', linestyle='--', linewidth=0.8, alpha=0.6)

    # ================= ROW 3: MEL SPECTROGRAM =================
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.set_facecolor('black')
    Pxx, freqs, bins, im = ax3.specgram(audio, NFFT=512, Fs=sr, noverlap=256, cmap='magma')
    format_ax(ax3, "Spectrogram + Phoneme Boundaries", "Frequency (Hz)")
    add_phoneme_boundaries(ax3)
    # No legend needed for specgram

    # ================= ROW 4: ENTROPY-DERIVED ATTENTION CURVE =================
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    ax4.plot(times, means, color=purple_line, linewidth=2.5, label='ECAPA Attention')
    ax4.fill_between(times, 0, means, color=purple_fill, alpha=0.6)
    format_ax(ax4, "ECAPA Entropy-Derived Attention", "Attention Weight")
    ax4.set_ylim(0, np.max(means) * 1.2 if len(means) > 0 else 1.05)
    ax4.legend(loc='upper right', fontsize=8)

    # ================= ROW 5: ATTENTION CURVE + BOUNDARIES =================
    ax5 = fig.add_subplot(gs[4, 0], sharex=ax1)
    ax5.plot(times, means, color=purple_line, linewidth=2.5, label='ECAPA Attention')
    ax5.fill_between(times, 0, means, color=purple_fill, alpha=0.6)
    format_ax(ax5, "ECAPA Attention + Phoneme Boundaries", "Attention Weight")
    ax5.set_ylim(0, np.max(means) * 1.2 if len(means) > 0 else 1.05)
    add_phoneme_boundaries(ax5)
    ax5.legend(loc='upper right', fontsize=8)
    
    # ================= ROW 6: TABLE RANKING =================
    ax6 = fig.add_subplot(gs[5, 0])
    ax6.axis('off')
    ax6.set_title("ECAPA Phoneme Ranking (sorted by mean attention)", fontsize=10, fontweight='bold', loc='left', pad=30)
    
    sorted_stats = sorted(phoneme_stats, key=lambda x: x['mean_entropy'], reverse=True)
    
    table_data = []
    columns = ["Rank", "Phoneme", "Word", "Start (s)", "End (s)", "Duration (s)", "Mean Attn", "Max Attn"]
    for i, s in enumerate(sorted_stats):
        duration = s['end'] - s['start']
        table_data.append([
            i+1, 
            s['phoneme'], 
            s['word'], 
            f"{s['start']:.3f}", 
            f"{s['end']:.3f}", 
            f"{duration:.3f}", 
            f"{s['mean_entropy']:.6f}", 
            f"{s['max_entropy']:.6f}"
        ])
        
    # Use bbox to position the table cleanly below the title and prevent overlapping
    table = ax6.table(cellText=table_data, colLabels=columns, loc='bottom', cellLoc='center', bbox=[0, 0, 1, 0.78])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#d9d9d9')
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(purple_line)
        else:
            if row % 2 == 0:
                cell.set_facecolor('#f2f2f2')
            else:
                cell.set_facecolor('white')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"6-Row Phoneme Dashboard saved to {output_png}")

if __name__ == "__main__":
    main()
