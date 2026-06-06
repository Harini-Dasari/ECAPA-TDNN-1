import os
import csv
import math
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

sys.path.append(os.getcwd())
from ECAPAModel import ECAPAModel

# Dictionary of representative recordings for each of the 10 target phrases
REPRESENTATIVE_RECORDINGS = {
    'ok_google': 'Reddots/pcm/m0002/20150727093434536_m0002_32.pcm',
    'birthday_parties_have_cupcakes_and_ice_cream': 'Reddots/pcm/m0010/20150224244213291_m0010_35.pcm',
    'my_voice_is_my_password': 'Reddots/pcm/m0002/20150803094831202_m0002_31.pcm',
    'a_watched_pot_never_boils': 'Reddots/pcm/m0002/20150727093248799_m0002_38.pcm',
    'only_lawyers_love_millionaires': 'Reddots/pcm/m0007/20150318162630467_m0007_33.pcm',
    'actions_speak_louder_than_words': 'Reddots/pcm/m0043/20150401162237970_m0043_36.pcm',
    'artificial_intelligence_is_for_real': 'Reddots/pcm/m0002/20150727093843169_m0002_34.pcm',
    'there_s_no_such_thing_as_a_free_lunch': 'Reddots/pcm/m0007/20150603214800167_m0007_37.pcm',
    'jealousy_has_twenty_twenty_vision': 'Reddots/pcm/m0002/20150406083444737_m0002_39.pcm',
    'necessity_is_the_mother_of_invention': 'Reddots/pcm/m0028/20150225052907815_m0028_40.pcm'
}

# Static representative word segmentations in seconds for the reference files
WORD_SEGMENTATIONS = {
    'ok_google': [
        {'word': 'OK', 'start': 0.20, 'end': 0.60},
        {'word': 'Google', 'start': 0.70, 'end': 1.30}
    ],
    'birthday_parties_have_cupcakes_and_ice_cream': [
        {'word': 'Birthday', 'start': 0.15, 'end': 0.60},
        {'word': 'parties', 'start': 0.60, 'end': 0.95},
        {'word': 'have', 'start': 0.95, 'end': 1.15},
        {'word': 'cupcakes', 'start': 1.15, 'end': 1.65},
        {'word': 'and', 'start': 1.65, 'end': 1.80},
        {'word': 'ice', 'start': 1.80, 'end': 2.05},
        {'word': 'cream', 'start': 2.05, 'end': 2.30}
    ],
    'my_voice_is_my_password': [
        {'word': 'My', 'start': 0.358, 'end': 0.493},
        {'word': 'voice', 'start': 0.556, 'end': 0.873},
        {'word': 'is', 'start': 0.983, 'end': 1.079},
        {'word': 'my', 'start': 1.134, 'end': 1.277},
        {'word': 'password', 'start': 1.346, 'end': 1.887}
    ],
    'a_watched_pot_never_boils': [
        {'word': 'A', 'start': 0.15, 'end': 0.30},
        {'word': 'watched', 'start': 0.30, 'end': 0.65},
        {'word': 'pot', 'start': 0.65, 'end': 0.95},
        {'word': 'never', 'start': 0.95, 'end': 1.35},
        {'word': 'boils', 'start': 1.35, 'end': 1.95}
    ],
    'only_lawyers_love_millionaires': [
        {'word': 'Only', 'start': 0.25, 'end': 0.80},
        {'word': 'lawyers', 'start': 0.80, 'end': 1.45},
        {'word': 'love', 'start': 1.45, 'end': 1.85},
        {'word': 'millionaires', 'start': 1.85, 'end': 2.95}
    ],
    'actions_speak_louder_than_words': [
        {'word': 'Actions', 'start': 0.35, 'end': 1.10},
        {'word': 'speak', 'start': 1.10, 'end': 1.65},
        {'word': 'louder', 'start': 1.65, 'end': 2.30},
        {'word': 'than', 'start': 2.30, 'end': 2.65},
        {'word': 'words', 'start': 2.65, 'end': 3.45}
    ],
    'artificial_intelligence_is_for_real': [
        {'word': 'Artificial', 'start': 0.15, 'end': 0.85},
        {'word': 'intelligence', 'start': 0.85, 'end': 1.45},
        {'word': 'is', 'start': 1.45, 'end': 1.65},
        {'word': 'for', 'start': 1.65, 'end': 1.85},
        {'word': 'real', 'start': 1.85, 'end': 2.40}
    ],
    'there_s_no_such_thing_as_a_free_lunch': [
        {'word': "There's", 'start': 0.20, 'end': 0.60},
        {'word': 'no', 'start': 0.60, 'end': 0.85},
        {'word': 'such', 'start': 0.85, 'end': 1.15},
        {'word': 'thing', 'start': 1.15, 'end': 1.45},
        {'word': 'as', 'start': 1.45, 'end': 1.60},
        {'word': 'a', 'start': 1.60, 'end': 1.70},
        {'word': 'free', 'start': 1.70, 'end': 2.10},
        {'word': 'lunch', 'start': 2.10, 'end': 2.80}
    ],
    'jealousy_has_twenty_twenty_vision': [
        {'word': 'Jealousy', 'start': 0.15, 'end': 0.70},
        {'word': 'has', 'start': 0.70, 'end': 0.95},
        {'word': 'twenty-twenty', 'start': 0.95, 'end': 1.55},
        {'word': 'vision', 'start': 1.55, 'end': 2.00}
    ],
    'necessity_is_the_mother_of_invention': [
        {'word': 'Necessity', 'start': 0.20, 'end': 0.95},
        {'word': 'is', 'start': 0.95, 'end': 1.15},
        {'word': 'the', 'start': 1.15, 'end': 1.25},
        {'word': 'mother', 'start': 1.25, 'end': 1.65},
        {'word': 'of', 'start': 1.65, 'end': 1.75},
        {'word': 'invention', 'start': 1.75, 'end': 2.65}
    ]
}

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
    audio, _ = sf.read(pcm_path, channels=1, samplerate=16000, subtype='PCM_16', format='RAW')
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

def process_phrase(model, phrase_clean, phrase_display):
    csv_path = f"xai_reddots/metadata/separated_phrases/{phrase_clean}.csv"
    rep_pcm = REPRESENTATIVE_RECORDINGS[phrase_clean]
    words = WORD_SEGMENTATIONS[phrase_clean]
    
    print(f"\n================ Processing Phrase: \"{phrase_display}\" ================")
    
    # 1. Read files
    recordings = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if os.path.exists(row['pcm_path']):
                recordings.append(row['pcm_path'])
                
    print(f"Found {len(recordings)} existing recordings in the dataset.")
    
    # 2. Extract attention for representative recording first
    rep_profile, rep_audio = extract_attention(model, rep_pcm)
    rep_len = len(rep_profile)
    rep_duration = len(rep_audio) / 16000.0
    rep_times = np.linspace(0, rep_duration, rep_len)
    
    # 3. Extract and interpolate all other profiles
    aligned_profiles = []
    aligned_profiles.append(rep_profile)
    
    for idx, path in enumerate(recordings):
        if path == rep_pcm:
            continue
        try:
            prof, _ = extract_attention(model, path)
            orig_x = np.linspace(0, rep_duration, len(prof))
            interp_prof = np.interp(rep_times, orig_x, prof)
            aligned_profiles.append(interp_prof)
        except Exception as e:
            pass
            
    aligned_profiles = np.vstack(aligned_profiles)
    mean_profile = np.mean(aligned_profiles, axis=0)
    std_profile = np.std(aligned_profiles, axis=0)
    
    # 4. Save averaged entropy curve
    entropy_out_csv = f"xai_reddots/entropy/average_attention_{phrase_clean}.csv"
    os.makedirs(os.path.dirname(entropy_out_csv), exist_ok=True)
    with open(entropy_out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'mean_attention', 'std_attention'])
        for t, m, s in zip(rep_times, mean_profile, std_profile):
            writer.writerow([f"{t:.4f}", f"{m:.6f}", f"{s:.6f}"])
            
    # 5. Compute word statistics
    word_stats = []
    for wb in words:
        w_name = wb['word']
        start_t = wb['start']
        end_t = wb['end']
        
        mask = (rep_times >= start_t) & (rep_times <= end_t)
        if np.any(mask):
            mean_attn = np.mean(mean_profile[mask])
            max_attn = np.max(mean_profile[mask])
        else:
            mean_attn = 0.0
            max_attn = 0.0
            
        word_stats.append({
            'word': w_name,
            'start': start_t,
            'end': end_t,
            'mean_attention': mean_attn,
            'max_attention': max_attn
        })
        
    # Sort words by attention descending
    sorted_words = sorted(word_stats, key=lambda x: x['mean_attention'], reverse=True)
    
    # Save ranked words
    ranked_words_csv = f"xai_reddots/reports/{phrase_clean}_word_ranked.csv"
    os.makedirs(os.path.dirname(ranked_words_csv), exist_ok=True)
    with open(ranked_words_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['word', 'start', 'end', 'mean_attention', 'max_attention'])
        writer.writeheader()
        writer.writerows(sorted_words)
        
    # 6. Generate simplified dashboard plot
    print("Generating simplified dashboard...")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    bg_color = '#f8f9fa'
    purple_line = '#7030a0'
    purple_fill = '#eaddf5'
    grid_color = '#e0e0e0'
    
    fig = plt.figure(figsize=(14, 18), facecolor=bg_color)
    fig.suptitle(f"ECAPA-TDNN XAI Dashboard • \"{phrase_display}\"\n{len(recordings)} recordings • Representative Speaker: {rep_pcm.split('/')[-2]}", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 4 rows
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 2, 2, 2.5], hspace=0.45)
    
    # ROW 1: Word Segmentation Timeline
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(bg_color)
    ax1.set_xlim(0, rep_duration)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title("Representative Word Segmentation", fontsize=11, fontweight='bold', loc='left')
    
    pastel_colors = ['#d1e7dd', '#fff3cd', '#f8d7da', '#cff4fc', '#e2e3e5']
    for i, wb in enumerate(words):
        rect_color = pastel_colors[i % len(pastel_colors)]
        rect = patches.Rectangle((wb['start'], 0.1), wb['end'] - wb['start'], 0.8, 
                                 linewidth=0.5, edgecolor='gray', facecolor=rect_color)
        ax1.add_patch(rect)
        
        center_x = (wb['start'] + wb['end']) / 2
        ax1.text(center_x, 0.5, wb['word'], ha='center', va='center', fontsize=11, fontweight='bold')
        ax1.text(center_x, 0.2, f"{wb['start']:.2f}s - {wb['end']:.2f}s", 
                 ha='center', va='center', fontsize=8, color='#555')
                 
    # ROW 2: Mel Spectrogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('black')
    audio_norm = rep_audio / np.max(np.abs(rep_audio))
    ax2.specgram(audio_norm, NFFT=512, Fs=16000, noverlap=256, cmap='magma')
    ax2.set_xlim(0, rep_duration)
    ax2.set_title("Mel Spectrogram + Word Boundaries", fontsize=11, fontweight='bold', loc='left')
    ax2.set_ylabel("Frequency (Hz)", fontsize=10)
    ax2.set_xlabel("Time (s)", fontsize=10)
    
    # Add vertical lines for word boundaries
    for wb in words:
        ax2.axvline(x=wb['end'], color='white', linestyle='--', linewidth=1.0, alpha=0.7)
        ax2.axvline(x=wb['start'], color='white', linestyle='--', linewidth=1.0, alpha=0.7)
        
    # ROW 3: Averaged Attention Curve
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor('white')
    ax3.plot(rep_times, mean_profile, color=purple_line, linewidth=2.5, label='Mean Attention')
    ax3.fill_between(rep_times, mean_profile - std_profile, mean_profile + std_profile, color=purple_fill, alpha=0.4, label='Standard Deviation')
    ax3.set_xlim(0, rep_duration)
    ax3.set_ylim(0, np.max(mean_profile) * 1.3)
    ax3.grid(True, linestyle='--', color=grid_color)
    ax3.set_title("Averaged Entropy-Derived Attention Curve", fontsize=11, fontweight='bold', loc='left')
    ax3.set_ylabel("Attention Weight", fontsize=10)
    ax3.set_xlabel("Time (s)", fontsize=10)
    ax3.legend(loc='upper right')
    
    # Add vertical lines for word boundaries in attention curve
    for wb in words:
        ax3.axvline(x=wb['end'], color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax3.axvline(x=wb['start'], color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        
    # ROW 4: Word Attention Ranking Table
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.axis('off')
    ax4.set_title("Word Attention Ranking Table (sorted by mean attention)", fontsize=11, fontweight='bold', loc='left', pad=10)
    
    table_data = []
    columns = ["Rank", "Word", "Start (s)", "End (s)", "Duration (s)", "Mean Attention", "Max Attention"]
    for idx, s in enumerate(sorted_words):
        duration = s['end'] - s['start']
        table_data.append([
            idx + 1,
            s['word'],
            f"{s['start']:.3f}",
            f"{s['end']:.3f}",
            f"{duration:.3f}",
            f"{s['mean_attention']:.6f}",
            f"{s['max_attention']:.6f}"
        ])
        
    table = ax4.table(cellText=table_data, colLabels=columns, loc='bottom', cellLoc='center', bbox=[0, 0, 1, 0.88])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor('#d9d9d9')
        if row_idx == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(purple_line)
        else:
            if row_idx % 2 == 0:
                cell.set_facecolor('#f2f2f2')
            else:
                cell.set_facecolor('white')
                
    plt.tight_layout()
    plot_out_dir = f"xai_reddots/separated_phrases/{phrase_clean}"
    os.makedirs(plot_out_dir, exist_ok=True)
    plot_out_png = os.path.join(plot_out_dir, 'dashboard.png')
    plt.savefig(plot_out_png, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print(f"Dashboard saved to {plot_out_png}")
    print(f"Top word for \"{phrase_display}\" is \"{sorted_words[0]['word']}\" with mean attention {sorted_words[0]['mean_attention']:.6f}")

def main():
    phrases = [
        ('ok_google', 'OK Google'),
        ('birthday_parties_have_cupcakes_and_ice_cream', 'Birthday parties have cupcakes and ice cream'),
        ('my_voice_is_my_password', 'My voice is my password'),
        ('a_watched_pot_never_boils', 'A watched pot never boils'),
        ('only_lawyers_love_millionaires', 'Only lawyers love millionaires'),
        ('actions_speak_louder_than_words', 'Actions speak louder than words'),
        ('artificial_intelligence_is_for_real', 'Artificial intelligence is for real'),
        ('there_s_no_such_thing_as_a_free_lunch', "There's no such thing as a free lunch"),
        ('jealousy_has_twenty_twenty_vision', 'Jealousy has twenty-twenty vision'),
        ('necessity_is_the_mother_of_invention', 'Necessity is the mother of invention')
    ]
    
    model = load_model()
    
    for clean_phrase, phrase_display in phrases:
        process_phrase(model, clean_phrase, phrase_display)
        
    print("\n================ Pipeline execution complete! ================")

if __name__ == '__main__':
    main()
