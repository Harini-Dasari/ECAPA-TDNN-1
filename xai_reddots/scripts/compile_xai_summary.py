import os
import csv
import numpy as np
import matplotlib.pyplot as plt

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
    
    summary_data = []
    
    for clean_phrase, phrase_display in phrases:
        sep_csv = f"xai_reddots/metadata/separated_phrases/{clean_phrase}.csv"
        entropy_csv = f"xai_reddots/entropy/average_attention_{clean_phrase}.csv"
        ranked_csv = f"xai_reddots/reports/{clean_phrase}_word_ranked.csv"
        
        # 1. Count recordings
        num_recordings = 0
        if os.path.exists(sep_csv):
            with open(sep_csv, 'r') as f:
                num_recordings = sum(1 for row in csv.DictReader(f) if os.path.exists(row['pcm_path']))
                
        # 2. Read average attention curve
        mean_attn = []
        if os.path.exists(entropy_csv):
            with open(entropy_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mean_attn.append(float(row['mean_attention']))
        mean_attn = np.array(mean_attn)
        
        # Double check normalization for Shannon Entropy computation
        if len(mean_attn) > 0:
            mean_attn = mean_attn / np.sum(mean_attn)
            peak_val = np.max(mean_attn)
            mean_val = np.mean(mean_attn)
            var_val = np.var(mean_attn)
            # Compute Shannon Entropy
            entropy_val = -np.sum(mean_attn * np.log(mean_attn + 1e-9))
        else:
            peak_val, mean_val, var_val, entropy_val = 0.0, 0.0, 0.0, 0.0
            
        # 3. Read top word
        top_word = "Unknown"
        if os.path.exists(ranked_csv):
            with open(ranked_csv, 'r') as f:
                reader = csv.DictReader(f)
                try:
                    first_row = next(reader)
                    top_word = first_row['word']
                except StopIteration:
                    pass
                    
        summary_data.append({
            'phrase': phrase_display,
            'num_recordings': num_recordings,
            'peak_attention': peak_val,
            'mean_attention': mean_val,
            'attention_variance': var_val,
            'attention_entropy': entropy_val,
            'top_word': top_word
        })
        
    # Write to CSV
    output_csv = 'xai_reddots/reports/phrase_summary_xai.csv'
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['phrase', 'num_recordings', 'peak_attention', 'mean_attention', 'attention_variance', 'attention_entropy', 'top_word']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)
        
    print(f"Summary CSV saved to {output_csv}")
    
    # 4. Generate comparison plots
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # Sort phrases by peak attention for clean plotting
    sorted_by_peak = sorted(summary_data, key=lambda x: x['peak_attention'], reverse=True)
    phrases_peak = [x['phrase'] for x in sorted_by_peak]
    peaks = [x['peak_attention'] for x in sorted_by_peak]
    
    # Plot 1: Peak Attention Weight Comparison
    fig1, ax1 = plt.subplots(figsize=(12, 6), facecolor='#f8f9fa')
    ax1.set_facecolor('white')
    bars1 = ax1.barh(phrases_peak, peaks, color='#7030a0', edgecolor='none', height=0.6)
    ax1.invert_yaxis()  # top-down
    ax1.grid(True, linestyle='--', color='#e0e0e0', axis='x')
    ax1.set_title("Peak Temporal Attention Weight by RedDots Passphrase\n(Higher peak indicates localized speaker-discriminative focus)", fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel("Peak Attention Weight (Normalized)", fontsize=10)
    
    # Add values on bars
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + 0.0001, bar.get_y() + bar.get_height()/2, f"{width:.4f}", 
                 ha='left', va='center', fontsize=9, fontweight='bold', color='#333')
                 
    plt.tight_layout()
    plot1_png = 'xai_reddots/plots/phrase_attention_comparison.png'
    plt.savefig(plot1_png, dpi=300, bbox_inches='tight', facecolor=fig1.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Peak attention plot saved to {plot1_png}")
    
    # Sort phrases by attention entropy descending (highest entropy = most spread out, lowest entropy = most concentrated)
    sorted_by_entropy = sorted(summary_data, key=lambda x: x['attention_entropy'])
    phrases_entropy = [x['phrase'] for x in sorted_by_entropy]
    entropies = [x['attention_entropy'] for x in sorted_by_entropy]
    
    # Plot 2: Attention Entropy (Concentration) Comparison
    fig2, ax2 = plt.subplots(figsize=(12, 6), facecolor='#f8f9fa')
    ax2.set_facecolor('white')
    bars2 = ax2.barh(phrases_entropy, entropies, color='#17a2b8', edgecolor='none', height=0.6)
    ax2.invert_yaxis()
    ax2.grid(True, linestyle='--', color='#e0e0e0', axis='x')
    ax2.set_title("Attention Entropy (Concentration) by RedDots Passphrase\n(Lower entropy indicates highly focused attention; higher indicates diffuse attention)", fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel("Shannon Entropy of Attention Curve (H_alpha)", fontsize=10)
    
    # Add values on bars
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 0.05, bar.get_y() + bar.get_height()/2, f"{width:.2f}", 
                 ha='left', va='center', fontsize=9, fontweight='bold', color='#333')
                 
    plt.tight_layout()
    plot2_png = 'xai_reddots/plots/phrase_entropy_concentration.png'
    plt.savefig(plot2_png, dpi=300, bbox_inches='tight', facecolor=fig2.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Attention entropy plot saved to {plot2_png}")

if __name__ == '__main__':
    main()
