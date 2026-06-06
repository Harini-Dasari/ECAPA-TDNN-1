import csv
import json
import os

def main():
    timeline_in = 'xai_reddots/metadata/timeline.json'
    timeline_out = 'xai_reddots/metadata/m0002_my_voice_is_my_password_timeline.json'
    
    # Count recordings
    count = 0
    with open('xai_reddots/metadata/separated_phrases/my_voice_is_my_password.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['speaker_id'] == 'm0002' and row['file_exists'].lower() == 'true':
                if os.path.exists(row['pcm_path']):
                    count += 1
                
    with open(timeline_in, 'r') as f:
        data = json.load(f)
        
    data['speaker'] = 'm0002'
    data['n_utterances'] = count
    data['n_valid'] = count
    data['n_alignment_sources'] = count
    
    # Word boundaries for m0002 from word_alignment.csv
    # My: 0.37-0.50
    # voice: 0.57-0.88
    # is: 0.99-1.08
    # my: 1.14-1.28
    # password: 1.35-1.88
    word_bounds = [
        (0.37, 0.50),
        (0.57, 0.88),
        (0.99, 1.08),
        (1.14, 1.28),
        (1.35, 1.88)
    ]
    
    # Group phonemes by word index
    word_phonemes = {}
    for ph in data['phonemes']:
        w_idx = ph['word_index']
        if w_idx not in word_phonemes:
            word_phonemes[w_idx] = []
        word_phonemes[w_idx].append(ph)
        
    for w_idx, bounds in enumerate(word_bounds):
        start_w, end_w = bounds
        ph_list = word_phonemes[w_idx]
        n_ph = len(ph_list)
        ph_dur = (end_w - start_w) / n_ph
        
        for i, ph in enumerate(ph_list):
            ph['word_start'] = start_w
            ph['word_end'] = end_w
            ph['start'] = start_w + i * ph_dur
            ph['end'] = start_w + (i + 1) * ph_dur
            
    with open(timeline_out, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"m0002 timeline JSON created successfully with {count} utterances.")

if __name__ == '__main__':
    main()
