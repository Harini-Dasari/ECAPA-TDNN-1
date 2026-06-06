import json

def fix_timeline(speaker, word_bounds):
    tl_path = f'xai_reddots/results/phrase1_my_voice_is_my_password/timelines/{speaker}_timeline.json'
    
    with open(tl_path, 'r') as f:
        data = json.load(f)
        
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
            
    with open(tl_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Fixed {speaker} timeline JSON!")

def main():
    m0005_bounds = [
        (0.50, 0.85),   # my
        (0.88, 1.30),   # voice
        (1.30, 1.55),   # is
        (1.60, 1.95),   # my
        (1.95, 2.60)    # password
    ]
    fix_timeline('m0005', m0005_bounds)

    m0006_bounds = [
        (0.60, 1.05),   # my
        (1.10, 1.55),   # voice
        (1.55, 1.85),   # is
        (1.95, 2.40),   # my
        (2.45, 3.26)    # password
    ]
    fix_timeline('m0006', m0006_bounds)

if __name__ == '__main__':
    main()
