import json
import csv
import os
import numpy as np

def main():
    timeline_json = 'xai_reddots/metadata/timeline.json'
    phrase_csv = 'xai_reddots/entropy/phrase_entropy.csv'
    groups_csv = 'xai_reddots/metadata/phrase_groups.csv'
    
    # Read speaker ID and phrase dynamically to construct output file path
    with open(groups_csv, 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)
        speaker_id = row['speaker_id']
        phrase = row['phrase']
    phrase_clean = phrase.lower().replace(" ", "_").replace('"', '')
    output_csv = f'xai_reddots/entropy/{speaker_id}_{phrase_clean}_ecapa_phoneme_ranked.csv'
    
    with open(timeline_json, 'r') as f:
        data = json.load(f)
        
    phonemes = data.get('phonemes', [])
    
    times, ents = [], []
    with open(phrase_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time']))
            ents.append(float(row['mean_entropy']))
            
    times = np.array(times)
    ents = np.array(ents)
    
    results = []
    
    for ph in phonemes:
        start = float(ph['start'])
        end = float(ph['end'])
        
        mask = (times >= start) & (times <= end)
        if np.any(mask):
            mean_ent = np.mean(ents[mask])
            max_ent = np.max(ents[mask])
        else:
            mean_ent = 0.0
            max_ent = 0.0
            
        results.append({
            'phoneme_id': ph['phoneme_id'],
            'phoneme': ph['phoneme'],
            'word': ph['word'],
            'start': f"{start:.16f}",
            'end': f"{end:.16f}",
            'mean_attention': f"{mean_ent:.16f}",
            'max_attention': f"{max_ent:.16f}"
        })
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['phoneme_id', 'phoneme', 'word', 'start', 'end', 'mean_attention', 'max_attention']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Aggregated {len(results)} phonemes. Saved to {output_csv}")

if __name__ == "__main__":
    main()
