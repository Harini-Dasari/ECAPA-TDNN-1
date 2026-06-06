import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import matplotlib.patches as patches
import torch
import sys
import math
import matplotlib.mlab as mlab
from matplotlib.colors import LinearSegmentedColormap

sys.path.append(os.getcwd())
from ECAPAModel import ECAPAModel

def load_model():
    print("Loading pretrained ECAPA-TDNN model...")
    args = type('Args', (), {})()
    args.C = 1024
    args.m = 0.2
    args.s = 30
    args.n_class = 5994
    args.lr = 0.001
    args.lr_decay = 0.97
    args.test_step = 1
    
    model = ECAPAModel(**vars(args))
    model.load_parameters("exps/pretrain.model")
    model.speaker_encoder.eval()
    return model

def extract_attention(model, pcm_path):
    audio, sr = sf.read(pcm_path, channels=1, samplerate=16000, subtype='PCM_16', format='RAW')
    if len(audio) < 400:
        audio = np.pad(audio, (0, 400 - len(audio)), 'constant')
    data = torch.FloatTensor(np.stack([audio], axis=0)).cuda()
    
    with torch.no_grad():
        x = model.speaker_encoder.torchfbank(data) + 1e-6
        x = x.log()
        x = x - torch.mean(x, dim=-1, keepdim=True)
        
        x = model.speaker_encoder.conv1(x)
        x = model.speaker_encoder.relu(x)
        x = model.speaker_encoder.bn1(x)
        
        x1 = model.speaker_encoder.layer1(x)
        x2 = model.speaker_encoder.layer2(x+x1)
        x3 = model.speaker_encoder.layer3(x+x1+x2)
        
        x = model.speaker_encoder.layer4(torch.cat((x1, x2, x3), dim=1))
        x = model.speaker_encoder.relu(x)
        
        t = x.size()[-1]
        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t), 
                             torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)
        
        w_logits = global_x
        for i in range(len(model.speaker_encoder.attention) - 1):
            w_logits = model.speaker_encoder.attention[i](w_logits)
            
        a = torch.softmax(w_logits, dim=1)
        H = -torch.sum(a * torch.log(a + 1e-9), dim=1)
        
        C_channels = a.shape[1]
        confidence = 1.0 - H / math.log(C_channels)
        alpha_hat = confidence / torch.sum(confidence, dim=1, keepdim=True)
        
        profile = alpha_hat.squeeze().cpu().numpy()
        
    return profile, audio

def check_timing_and_correlation(recordings, sr, profile, profile_len):
    hop_length = 160
    win_length = 400
    half_win = win_length // 2
    
    all_rms = []
    
    for path in recordings:
        try:
            audio, _ = sf.read(path, channels=1, samplerate=16000, subtype='PCM_16', format='RAW')
            num_frames = len(audio) // hop_length
            rms = []
            for j in range(num_frames):
                center = j * hop_length
                start = max(0, center - half_win)
                end = min(len(audio), center + half_win)
                frame_audio = audio[start:end]
                if len(frame_audio) > 0:
                    val = np.sqrt(np.mean(frame_audio ** 2))
                else:
                    val = 0.0
                rms.append(val)
            rms = np.array(rms)
            
            orig_x = np.linspace(0, 1, len(rms))
            target_x = np.linspace(0, 1, profile_len)
            interp_rms = np.interp(target_x, orig_x, rms)
            all_rms.append(interp_rms)
        except Exception as e:
            print(f"Error reading/computing RMS for {path}: {e}")
            
    if all_rms:
        avg_rms = np.mean(all_rms, axis=0)
        correlation = np.corrcoef(avg_rms, profile)[0, 1]
    else:
        avg_rms = np.zeros(profile_len)
        correlation = 0.0
        
    print("\n" + "="*60)
    print("TIMING AND ALIGNMENT CHECK AUDIT (AVERAGED OVER ALL RECORDINGS):")
    print(f"Number of recordings:  {len(recordings)}")
    print(f"Attention Frames:      {profile_len}")
    print(f"Attention Step Size:   0.01 seconds")
    print(f"RMS Energy vs. Entropy Attention Pearson Correlation: {correlation:.6f}")
    print("="*60 + "\n")
    
    return correlation, avg_rms

def main():
    speaker_id = sys.argv[1] if len(sys.argv) > 1 else "m0004"
    phrase_clean = "my_voice_is_my_password"
    
    # 1. Paths Setup
    separated_csv = f'xai_reddots/metadata/separated_phrases/{phrase_clean}.csv'
    
    # Try multiple paths for timeline
    timeline_paths_to_try = [
        f'xai_reddots/timelines/{speaker_id}_{phrase_clean}_timeline.json',
        f'xai_reddots/metadata/{speaker_id}_{phrase_clean}_timeline.json'
    ]
    if speaker_id == "m0001":
        timeline_paths_to_try.append('xai_reddots/metadata/timeline.json')
        
    timeline_json_path = None
    for path in timeline_paths_to_try:
        if os.path.exists(path):
            timeline_json_path = path
            break
            
    if timeline_json_path is None:
        print(f"Error: Timeline JSON not found in tried paths: {timeline_paths_to_try}")
        sys.exit(1)
            
    # Load timeline data
    with open(timeline_json_path, 'r') as f:
        timeline_data = json.load(f)
    
    # Get speaker's recordings
    recordings = []
    with open(separated_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['speaker_id'] == speaker_id and os.path.exists(row['pcm_path']):
                recordings.append(row['pcm_path'])
                
    if not recordings:
        print(f"Error: No recordings found for speaker {speaker_id}")
        sys.exit(1)
        
    # Choose first recording as representative
    pcm_path = recordings[0]
    print(f"Using representative recording (Reference): {pcm_path}")
    print(f"Found {len(recordings)} total recordings for speaker {speaker_id}. Averaging profiles, waveforms, and spectrograms...")
    
    # Get average duration from timeline JSON
    avg_duration = float(timeline_data.get('avg_duration_sec', 3.0))
    print(f"Aligning and warping all recordings to average speaker timeline of {avg_duration:.3f} seconds.")
    
    # 2. Extract Attention for all recordings and warp them to the average timeline
    model = load_model()
    
    # Use the representative recording's profile length as standard resolution
    rep_profile, rep_audio = extract_attention(model, pcm_path)
    profile_len = len(rep_profile)
    profile_times = np.linspace(0, avg_duration, profile_len)
    
    # We will interpolate all audio envelopes to a standard time axis of length 60,000 samples
    num_audio_samples = int(16000 * avg_duration)
    audio_time = np.linspace(0, avg_duration, num_audio_samples)
    
    # Setup standard spectrogram bins using a dummy signal of average duration
    dummy_audio = np.zeros(num_audio_samples)
    Pxx_ref, freqs, bins = mlab.specgram(dummy_audio, NFFT=512, Fs=16000, noverlap=256)
    
    aligned_profiles = []
    audio_envelopes = []
    all_Pxx = []
    
    for idx, path in enumerate(recordings):
        try:
            prof, audio_i = extract_attention(model, path)
            orig_duration = len(audio_i) / 16000.0
            
            # Interpolate attention profile to target profile_times
            orig_profile_x = np.linspace(0, orig_duration, len(prof))
            interp_prof = np.interp(profile_times, orig_profile_x, prof)
            aligned_profiles.append(interp_prof)
            
            # Normalize and interpolate absolute waveform envelope
            audio_i_norm = audio_i / np.max(np.abs(audio_i))
            orig_audio_x = np.linspace(0, orig_duration, len(audio_i_norm))
            interp_audio_env = np.interp(audio_time, orig_audio_x, np.abs(audio_i_norm))
            audio_envelopes.append(interp_audio_env)
            
            # Compute and interpolate spectrogram
            Pxx_i, _, bins_i = mlab.specgram(audio_i_norm, NFFT=512, Fs=16000, noverlap=256)
            orig_spec_x = bins_i
            Pxx_i_interp = np.zeros((Pxx_i.shape[0], len(bins)))
            for f_idx in range(Pxx_i.shape[0]):
                Pxx_i_interp[f_idx, :] = np.interp(bins, orig_spec_x, Pxx_i[f_idx, :])
            all_Pxx.append(Pxx_i_interp)
            
            print(f"Aligned and averaged recording {idx+1}/{len(recordings)}: {os.path.basename(path)}")
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    aligned_profiles = np.vstack(aligned_profiles)
    profile = np.mean(aligned_profiles, axis=0)
    profile_std = np.std(aligned_profiles, axis=0)
    
    audio_envelopes = np.vstack(audio_envelopes)
    avg_audio_envelope = np.mean(audio_envelopes, axis=0)
    
    avg_Pxx = np.mean(all_Pxx, axis=0)
    
    # Timing audit and correlation check (on averaged RMS energy vs mean profile)
    correlation, rms_energy = check_timing_and_correlation(recordings, 16000, profile, profile_len)
    
    # 3. Compute Phoneme Attention and Output CSV
    phonemes = timeline_data.get('phonemes', [])
    csv_results = []
    
    for ph in phonemes:
        start = float(ph['start'])
        end = float(ph['end'])
        
        # Mask frames falling in phoneme boundary
        mask = (profile_times >= start) & (profile_times <= end)
        if not np.any(mask):
            # Fallback to closest frame if segment is too narrow
            closest_idx = np.argmin(np.abs(profile_times - (start + end)/2))
            mask = np.zeros_like(profile_times, dtype=bool)
            mask[closest_idx] = True
            
        # Calculate mean and standard deviation of attention across recordings within boundaries
        ph_attn_per_rec = []
        for r_idx in range(aligned_profiles.shape[0]):
            ph_attn_per_rec.append(np.mean(aligned_profiles[r_idx, mask]))
            
        ph_attn_per_rec = np.array(ph_attn_per_rec)
        mean_attn = np.mean(ph_attn_per_rec)
        std_attn = np.std(ph_attn_per_rec)
        max_attn = np.max(ph_attn_per_rec)
        
        csv_results.append({
            'phoneme_id': ph['phoneme_id'],
            'phoneme': ph['phoneme'],
            'word': ph['word'],
            'start': start,
            'end': end,
            'mean_attention': mean_attn,
            'std_attention': std_attn,
            'max_attention': max_attn
        })
        
    # Save a copy of the timeline data in the structured timelines folder
    out_timeline_dir = 'xai_reddots/timelines'
    os.makedirs(out_timeline_dir, exist_ok=True)
    out_timeline_path = os.path.join(out_timeline_dir, f'{speaker_id}_{phrase_clean}_timeline.json')
    with open(out_timeline_path, 'w') as f:
        json.dump(timeline_data, f, indent=2)
    print(f"Timeline JSON saved to {out_timeline_path}")

    # Save phoneme ranked CSV to xai_reddots/csv/
    output_csv = f"xai_reddots/csv/{speaker_id}_{phrase_clean}_ecapa_entropy_phenome_attention.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['phoneme_id', 'phoneme', 'word', 'start', 'end', 'mean_attention', 'std_attention', 'max_attention'])
        writer.writeheader()
        writer.writerows(csv_results)
    print(f"Phoneme attention saved to {output_csv}")
    
    # Save frame-by-frame attention profile CSV to xai_reddots/attention/
    attention_csv = f"xai_reddots/attention/{speaker_id}_{phrase_clean}_attention.csv"
    os.makedirs(os.path.dirname(attention_csv), exist_ok=True)
    with open(attention_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'mean_attention', 'std_attention'])
        for t, m, s in zip(profile_times, profile, profile_std):
            writer.writerow([f"{t:.4f}", f"{m:.6f}", f"{s:.6f}"])
    print(f"Attention profile saved to {attention_csv}")
    
    # 4. Generate the 7-Panel Dashboard Graph
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    bg_color = '#f8f9fa'
    purple_line = '#7030a0'
    purple_fill = '#eaddf5'
    grid_color = '#e0e0e0'
    
    # Figure size for 7 rows is 16 x 28
    fig = plt.figure(figsize=(16, 28), facecolor=bg_color)
    fig.suptitle(f"ECAPA-TDNN Speaker XAI Dashboard • Speaker: {speaker_id}\nPhrase: \"{timeline_data['phrase']}\" • Averaged over {len(recordings)} recordings", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(7, 1, height_ratios=[1.2, 1.2, 2.5, 2.5, 2.5, 2.5, 3.5], hspace=0.6)
    
    # Extract unique word boundaries
    word_bounds = {}
    for ph in csv_results:
        w_name = ph['word']
        start_t = float(ph['start'])
        end_t = float(ph['end'])
        
        if w_name not in word_bounds:
            word_bounds[w_name] = {'start': start_t, 'end': end_t}
        else:
            word_bounds[w_name]['start'] = min(word_bounds[w_name]['start'], start_t)
            word_bounds[w_name]['end'] = max(word_bounds[w_name]['end'], end_t)
            
    word_list = [{'word': w, 'start': data['start'], 'end': data['end']} for w, data in word_bounds.items()]
    # Ensure ordered by start time
    word_list.sort(key=lambda x: x['start'])
    
    pastel_colors = ['#d1e7dd', '#fff3cd', '#f8d7da', '#cff4fc', '#e2e3e5']
    phoneme_colors = ['#f8d7da', '#cff4fc', '#d1e7dd', '#fff3cd', '#eaddf5']
    
    # ---------------- PANEL 1: Word Timeline (WhisperX alignment) ----------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(bg_color)
    ax1.set_xlim(0, avg_duration)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title("1. Word Timeline (WhisperX alignment)", fontsize=11, fontweight='bold', loc='left', pad=10)
    
    for i, w in enumerate(word_list):
        rect_color = pastel_colors[i % len(pastel_colors)]
        rect = patches.Rectangle((w['start'], 0.1), w['end'] - w['start'], 0.8, 
                                 linewidth=0.5, edgecolor='gray', facecolor=rect_color)
        ax1.add_patch(rect)
        center_x = (w['start'] + w['end']) / 2
        ax1.text(center_x, 0.5, w['word'], ha='center', va='center', fontsize=10, fontweight='bold')
        ax1.text(center_x, 0.2, f"{w['start']:.2f}s - {w['end']:.2f}s", ha='center', va='center', fontsize=8, color='#444')
        
    # ---------------- PANEL 2: Phoneme Timeline ----------------
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.set_facecolor(bg_color)
    ax2.set_xlim(0, avg_duration)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title("2. Phoneme Timeline", fontsize=11, fontweight='bold', loc='left', pad=10)
    
    for i, ph in enumerate(csv_results):
        rect_color = phoneme_colors[i % len(phoneme_colors)]
        rect = patches.Rectangle((ph['start'], 0.1), ph['end'] - ph['start'], 0.8, 
                                 linewidth=0.5, edgecolor='gray', facecolor=rect_color)
        ax2.add_patch(rect)
        center_x = (ph['start'] + ph['end']) / 2
        ax2.text(center_x, 0.5, ph['phoneme'], ha='center', va='center', fontsize=9, fontweight='bold')
        ax2.text(center_x, 0.2, f"{ph['start']:.2f}s", ha='center', va='center', fontsize=7, color='#444')
        
    # Helper to style subplots
    def format_subplot(ax, title, ylabel):
        ax.set_facecolor('white')
        ax.grid(True, linestyle='--', color=grid_color, alpha=0.7)
        ax.set_xlim(0, avg_duration)
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Time (seconds)", fontsize=10)
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            
    # Helper to draw vertical lines for phonemes
    def add_phoneme_lines(ax, text_y):
        for ph in csv_results:
            ax.axvline(x=ph['end'], color='red', linestyle='--', linewidth=0.8, alpha=0.5)
            center_x = (ph['start'] + ph['end']) / 2
            ax.text(center_x, text_y, ph['phoneme'], color='red', ha='center', va='top', fontsize=7, alpha=0.7, weight='bold')
        # Also line for start of first phoneme
        if csv_results:
            ax.axvline(x=csv_results[0]['start'], color='red', linestyle='--', linewidth=0.8, alpha=0.5)

    # ---------------- PANEL 3: Average Mel Spectrogram + Boundaries ----------------
    # Compute relative threshold based on averaged signal power percentiles
    avg_Pxx_db = 10 * np.log10(avg_Pxx + 1e-10)
    vmax_db = np.percentile(avg_Pxx_db, 99.9)
    vmin_db = vmax_db - 45
    
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.pcolormesh(bins, freqs, avg_Pxx_db, cmap='gray_r', vmin=vmin_db, vmax=vmax_db, shading='nearest')
    format_subplot(ax3, "3. Average Mel Spectrogram + Boundaries", "Frequency (Mel)")
    ax3.set_ylim(0, 8000)
    ax3.set_yticks([0, 2000, 4000, 6000, 8000])
    ax3.set_yticklabels(['0Hz', '2kHz', '4kHz', '6kHz', '8kHz'])
    add_phoneme_lines(ax3, 7600)
    
    # ---------------- PANEL 4: Average Mel Spectrogram + Boundaries + Overlaid Attention ----------------
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    ax4.pcolormesh(bins, freqs, avg_Pxx_db, cmap='gray_r', vmin=vmin_db, vmax=vmax_db, shading='nearest')
    format_subplot(ax4, "4. Average Mel Spectrogram + Boundaries + Overlaid Attention", "Frequency (Mel)")
    ax4.set_ylim(0, 8000)
    ax4.set_yticks([0, 2000, 4000, 6000, 8000])
    ax4.set_yticklabels(['0Hz', '2kHz', '4kHz', '6kHz', '8kHz'])
    add_phoneme_lines(ax4, 7600)
    
    # Overlay attention using dual y-axis
    ax4_twin = ax4.twinx()
    ax4_twin.plot(profile_times, profile, color='#e67e22', linewidth=2.0, alpha=0.95, label='Mean ECAPA Attention')
    ax4_twin.set_ylabel("Attention Weight", color='#e67e22', fontsize=10)
    ax4_twin.tick_params(axis='y', labelcolor='#e67e22')
    ax4_twin.set_ylim(0, np.max(profile) * 1.25)
    ax4_twin.spines['top'].set_visible(False)
    ax4_twin.spines['left'].set_visible(False)
    ax4_twin.spines['bottom'].set_visible(False)
    
    # ---------------- PANEL 5: ECAPA Entropy-Derived Attention (Averaged) ----------------
    ax5 = fig.add_subplot(gs[4, 0], sharex=ax1)
    ax5.plot(profile_times, profile, color=purple_line, linewidth=2.5, label='Mean Attention')
    ax5.fill_between(profile_times, np.maximum(0, profile - profile_std), profile + profile_std, color=purple_fill, alpha=0.4, label='Std Dev (Variability)')
    format_subplot(ax5, "5. ECAPA Entropy-Derived Attention (Averaged)", "Attention Weight")
    ax5.set_ylim(0, np.max(profile) * 1.25)
    ax5.legend(loc='upper right', fontsize=8)
    
    # ---------------- PANEL 6: ECAPA Attention + Phoneme Boundaries (Averaged) ----------------
    ax6 = fig.add_subplot(gs[5, 0], sharex=ax1)
    ax6.plot(profile_times, profile, color=purple_line, linewidth=2.5, label='Mean Attention')
    ax6.fill_between(profile_times, np.maximum(0, profile - profile_std), profile + profile_std, color=purple_fill, alpha=0.4, label='Std Dev (Variability)')
    format_subplot(ax6, "6. ECAPA Attention + Phoneme Boundaries (Averaged)", "Attention Weight")
    ax6.set_ylim(0, np.max(profile) * 1.25)
    add_phoneme_lines(ax6, np.max(profile) * 1.15)
    ax6.legend(loc='upper right', fontsize=8)
    
    # ---------------- PANEL 7: ECAPA Entropy Attention Phoneme Ranking (Averaged) ----------------
    ax7 = fig.add_subplot(gs[6, 0])
    ax7.set_facecolor('white')
    ax7.grid(True, linestyle='--', color=grid_color, alpha=0.7)
    
    # Sort phonemes by mean attention
    sorted_phonemes = sorted(csv_results, key=lambda x: x['mean_attention'], reverse=False)
    
    labels = [f"{ph['word']}: /{ph['phoneme']}/" for ph in sorted_phonemes]
    means = [ph['mean_attention'] for ph in sorted_phonemes]
    stds = [ph['std_attention'] for ph in sorted_phonemes]
    
    bars = ax7.barh(labels, means, xerr=stds, color=purple_line, alpha=0.85, edgecolor='gray', height=0.6,
                    error_kw=dict(ecolor='#333333', lw=1.2, capsize=3))
    
    # Add values on the bars (mean ± std)
    for bar, std_val in zip(bars, stds):
        width = bar.get_width()
        ax7.text(width + std_val + max(means)*0.015, bar.get_y() + bar.get_height()/2, f"{width:.5f} ± {std_val:.5f}", 
                 va='center', ha='left', fontsize=8, color='#333333', fontweight='semibold')
                 
    ax7.set_title("7. ECAPA Entropy Attention Phoneme Ranking (Averaged)", fontsize=11, fontweight='bold', loc='left')
    ax7.set_xlabel("Mean Attention Weight", fontsize=10)
    max_x_limit = max([m + s for m, s in zip(means, stds)]) * 1.35
    ax7.set_xlim(0, max_x_limit)
    for spine in ax7.spines.values():
        spine.set_color('#cccccc')
        
    plt.tight_layout()
    plot_out_dir = f"xai_reddots/plots"
    os.makedirs(plot_out_dir, exist_ok=True)
    plot_out_png = os.path.join(plot_out_dir, f'{speaker_id}_{phrase_clean}_entropy_dashboard.png')
    plt.savefig(plot_out_png, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print(f"Dashboard plot saved to {plot_out_png}")
    print("Execution successfully finished.")

if __name__ == '__main__':
    main()
