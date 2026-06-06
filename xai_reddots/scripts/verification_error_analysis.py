import os
import csv
import json
import math
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import sys

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
    audio, _ = sf.read(pcm_path, channels=1, samplerate=16000, subtype='PCM_16', format='RAW')
    # Pad audio if it's too short (ECAPA conv layers require a minimum length)
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
        
    return profile

def main():
    predictions_csv = 'exps_reddots/trial_predictions-entropy-attention.csv'
    timeline_json = 'xai_reddots/metadata/timeline.json'
    pcm_base = 'Reddots/pcm'
    output_csv = 'xai_reddots/reports/verification_error_analysis_stats.csv'
    output_png = 'xai_reddots/plots/error_analysis_attention.png'
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    
    # 1. Parse timeline JSON to get the real boundaries for "My voice is my password"
    with open(timeline_json, 'r') as f:
        timeline_data = json.load(f)
    phonemes = timeline_data.get('phonemes', [])
    
    # Extract unique word boundaries
    word_bounds = {}
    for ph in phonemes:
        w_idx = ph['word_index']
        if w_idx not in word_bounds:
            word_bounds[w_idx] = {'word': ph['word'], 'start': float(ph['word_start']), 'end': float(ph['word_end'])}
    words_list = [word_bounds[i] for i in sorted(word_bounds.keys())]
    
    # 2. Read and group trials
    print(f"Reading predictions from {predictions_csv}...")
    groups = {'TA': [], 'FR': [], 'TR': [], 'FA': []}
    
    with open(predictions_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # We filter for trials where both enrollment and test are phrase 31 (My voice is my password)
            if row['Enrollment_ID'].endswith('_31') and row['Test_Audio'].endswith('_31'):
                gt = int(row['GroundTruth'])
                pred = int(row['Prediction_0.6424'])
                rec_path = os.path.join(pcm_base, row['Test_Audio'] + '.pcm')
                
                # Double check that file exists
                if not os.path.exists(rec_path):
                    continue
                    
                if gt == 1 and pred == 1:
                    groups['TA'].append(rec_path)
                elif gt == 1 and pred == 0:
                    groups['FR'].append(rec_path)
                elif gt == 0 and pred == 0:
                    groups['TR'].append(rec_path)
                elif gt == 0 and pred == 1:
                    groups['FA'].append(rec_path)
                    
    for k, v in groups.items():
        print(f"Group {k}: found {len(v)} existing trials.")
        
    # 3. Sample from groups to balance runtime and representation
    random.seed(42)
    sample_size = 100
    sampled_groups = {}
    for k, v in groups.items():
        if len(v) > sample_size:
            sampled_groups[k] = random.sample(v, sample_size)
        else:
            sampled_groups[k] = v
            
    # 4. Load model
    model = load_model()
    
    # 5. Extract and average attention curves
    # Target representative duration = 2.0 seconds (200 frames)
    target_len = 200
    target_times = np.linspace(0, 2.0, target_len)
    
    group_profiles = {}
    
    for k, paths in sampled_groups.items():
        print(f"Processing group {k} ({len(paths)} files)...")
        profiles = []
        for idx, path in enumerate(paths):
            try:
                prof = extract_attention(model, path)
                # Interpolate to match the target timeline length (2.0s)
                orig_x = np.linspace(0, 2.0, len(prof))
                interp_prof = np.interp(target_times, orig_x, prof)
                profiles.append(interp_prof)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                
        if profiles:
            group_profiles[k] = np.mean(np.vstack(profiles), axis=0)
        else:
            group_profiles[k] = np.zeros(target_len)
            
    # 6. Map to word and phoneme boundaries
    print("Computing attention stats per word and phoneme...")
    results = []
    
    # Analyze Words
    for wb in words_list:
        w_name = wb['word']
        start_t = wb['start']
        end_t = wb['end']
        
        mask = (target_times >= start_t) & (target_times <= end_t)
        
        word_attn = {}
        for k in ['TA', 'FR', 'TR', 'FA']:
            if np.any(mask):
                word_attn[k] = np.mean(group_profiles[k][mask])
            else:
                word_attn[k] = 0.0
                
        results.append({
            'Type': 'Word',
            'Name': w_name,
            'Start': start_t,
            'End': end_t,
            'TA_attn': f"{word_attn['TA']:.6f}",
            'FR_attn': f"{word_attn['FR']:.6f}",
            'TR_attn': f"{word_attn['TR']:.6f}",
            'FA_attn': f"{word_attn['FA']:.6f}",
            'Difference_FR_TA': f"{(word_attn['FR'] - word_attn['TA']):.6f}",
            'Difference_FA_TR': f"{(word_attn['FA'] - word_attn['TR']):.6f}"
        })
        
    # Analyze Phonemes
    for ph in phonemes:
        p_name = ph['phoneme']
        w_parent = ph['word']
        start_t = float(ph['start'])
        end_t = float(ph['end'])
        
        mask = (target_times >= start_t) & (target_times <= end_t)
        
        ph_attn = {}
        for k in ['TA', 'FR', 'TR', 'FA']:
            if np.any(mask):
                ph_attn[k] = np.mean(group_profiles[k][mask])
            else:
                ph_attn[k] = 0.0
                
        results.append({
            'Type': 'Phoneme',
            'Name': f"{w_parent} ({p_name})",
            'Start': start_t,
            'End': end_t,
            'TA_attn': f"{ph_attn['TA']:.6f}",
            'FR_attn': f"{ph_attn['FR']:.6f}",
            'TR_attn': f"{ph_attn['TR']:.6f}",
            'FA_attn': f"{ph_attn['FA']:.6f}",
            'Difference_FR_TA': f"{(ph_attn['FR'] - ph_attn['TA']):.6f}",
            'Difference_FA_TR': f"{(ph_attn['FA'] - ph_attn['TR']):.6f}"
        })
        
    # Save stats to CSV
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['Type', 'Name', 'Start', 'End', 'TA_attn', 'FR_attn', 'TR_attn', 'FA_attn', 'Difference_FR_TA', 'Difference_FA_TR']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Stats saved to {output_csv}")
    
    # 7. Generate comparative plot
    print("Generating attention comparison plot...")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#f8f9fa')
    ax.set_facecolor('white')
    
    # Colors
    colors = {
        'TA': '#2ca02c', # Green (Correct Accept)
        'FR': '#d62728', # Red (Incorrect Reject)
        'TR': '#1f77b4', # Blue (Correct Reject)
        'FA': '#ff7f0e'  # Orange (Incorrect Accept)
    }
    
    for k, prof in group_profiles.items():
        label_map = {
            'TA': 'True Accept (Target Correct)',
            'FR': 'False Reject (Target Wrong / FR)',
            'TR': 'True Reject (Impostor Correct)',
            'FA': 'False Accept (Impostor Wrong / FA)'
        }
        ax.plot(target_times, prof, label=label_map[k], color=colors[k], linewidth=2.5)
        
    # Add Word boundaries as shaded regions and text
    pastel_colors = ['#d1e7dd', '#fff3cd', '#f8d7da', '#cff4fc', '#e2e3e5']
    y_min, y_max = ax.get_ylim()
    # Boost y_max slightly for text labels
    ax.set_ylim(0, max(np.max(group_profiles['TA']), np.max(group_profiles['FR'])) * 1.25)
    
    for i, wb in enumerate(words_list):
        rect_color = pastel_colors[i % len(pastel_colors)]
        rect = patches.Rectangle((wb['start'], 0), wb['end'] - wb['start'], ax.get_ylim()[1],
                                 linewidth=0, facecolor=rect_color, alpha=0.3)
        ax.add_patch(rect)
        center_x = (wb['start'] + wb['end']) / 2
        ax.text(center_x, ax.get_ylim()[1]*0.95, wb['word'], ha='center', va='top', fontsize=11, fontweight='bold', color='#333')
        
    ax.grid(True, linestyle='--', color='#e0e0e0', alpha=0.7)
    ax.set_xlim(0, 2.0)
    ax.set_title("Entropy-Derived Attention Profile by Speaker Verification Decision Category\nPhrase: \"My voice is my password\"", 
             fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Attention Weight (Normalized)", fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Comparison plot saved to {output_png}")
    
if __name__ == '__main__':
    main()
