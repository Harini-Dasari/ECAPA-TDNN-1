import os
import csv
import numpy as np

def main():
    metadata_csv = 'xai_reddots/metadata/phrase_groups.csv'
    align_csv = 'xai_reddots/metadata/word_alignment.csv'
    frame_csv = 'xai_reddots/entropy/frame_entropy.csv'
    word_csv = 'xai_reddots/entropy/word_entropy.csv'
    phrase_csv = 'xai_reddots/entropy/phrase_entropy.csv'
    
    # 1. Read frame entropies
    entropies = {}
    with open(frame_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec_id = row['recording_id']
            if rec_id not in entropies:
                entropies[rec_id] = {'time': [], 'entropy': []}
            entropies[rec_id]['time'].append(float(row['time']))
            entropies[rec_id]['entropy'].append(float(row['entropy']))
            
    # 2. Read word alignments and compute word stats per recording
    word_stats = {}
    with open(align_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec_id = row['recording_id']
            word = row['word']
            start = float(row['start_time'])
            end = float(row['end_time'])
            
            if rec_id not in entropies:
                continue
                
            times = np.array(entropies[rec_id]['time'])
            ents = np.array(entropies[rec_id]['entropy'])
            
            mask = (times >= start) & (times <= end)
            if np.any(mask):
                val = np.mean(ents[mask])
            else:
                val = 0.0 # fallback if no frames found
                
            if word not in word_stats:
                word_stats[word] = []
            word_stats[word].append(val)
            
    # 3. Aggregate word entropy across all recordings
    os.makedirs(os.path.dirname(word_csv), exist_ok=True)
    with open(word_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['word', 'mean_entropy', 'std_entropy'])
        writer.writeheader()
        
        # We need to preserve original phrase order: My, voice, is, my, password
        # Since dict is ordered in Python 3.7+, this usually works, but let's be safe.
        # Actually dict insertion order is preserved.
        for word, vals in word_stats.items():
            writer.writerow({
                'word': word,
                'mean_entropy': f"{np.mean(vals):.4f}",
                'std_entropy': f"{np.std(vals):.4f}"
            })
            
    # 4. Compute Representative Entropy Profile (Phrase Entropy)
    # Pick the first recording as the representative
    rep_rec = list(entropies.keys())[0]
    rep_times = np.array(entropies[rep_rec]['time'])
    rep_len = len(rep_times)
    
    aligned_entropies = []
    for rec_id, data in entropies.items():
        ents = np.array(data['entropy'])
        times = np.array(data['time'])
        # Interpolate to match the representative's time axis
        # We map their index space [0, len-1] to rep's index space [0, rep_len-1]
        orig_x = np.linspace(0, 1, len(ents))
        target_x = np.linspace(0, 1, rep_len)
        interp_ents = np.interp(target_x, orig_x, ents)
        aligned_entropies.append(interp_ents)
        
    aligned_entropies = np.vstack(aligned_entropies)
    mean_profile = np.mean(aligned_entropies, axis=0)
    std_profile = np.std(aligned_entropies, axis=0)
    
    with open(phrase_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['time', 'mean_entropy', 'std_entropy'])
        writer.writeheader()
        for t, m, s in zip(rep_times, mean_profile, std_profile):
            writer.writerow({
                'time': f"{t:.4f}",
                'mean_entropy': f"{m:.4f}",
                'std_entropy': f"{s:.4f}"
            })
            
    print(f"Aggregation complete. Representative recording used: {rep_rec}")

if __name__ == "__main__":
    main()
