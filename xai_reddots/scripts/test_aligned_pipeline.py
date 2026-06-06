import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PHRASE_DEF = {
    'key'    : 'my_voice_is_my_password',
    'display': 'My voice is my password',
    'words'  : [
        ('my',       ['M', 'AY']),
        ('voice',    ['V', 'OY', 'S']),
        ('is',       ['IH', 'Z']),
        ('my',       ['M', 'AY']),
        ('password', ['P', 'AE', 'S', 'W', 'ER', 'D']),
    ],
    'syl_weights': [1, 1, 1, 1, 2],
}

def decode_alignment(alignments, tokens, blank_id=0):
    path = []
    j = 0
    for idx, label in enumerate(alignments):
        if label == blank_id:
            path.append(None)
        elif j < len(tokens) and label == tokens[j]:
            path.append(j)
        elif j + 1 < len(tokens) and label == tokens[j+1]:
            j += 1
            path.append(j)
        else:
            found = False
            for k in range(j + 2, min(j + 5, len(tokens))):
                if label == tokens[k]:
                    j = k
                    path.append(j)
                    found = True
                    break
            if not found:
                path.append(None)
    return path

def generate_aligned_timeline(pcm_path, phrase_def, speaker_id, wav2vec_model, dictionary, labels):
    with open(pcm_path, 'rb') as f:
        pcm_data = f.read()
    audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.tensor(audio).unsqueeze(0).to(device)
    
    clean_text = phrase_def['display'].upper().replace(" ", "|")
    tokens = [dictionary[c] for c in clean_text if c in dictionary]
    
    with torch.inference_mode():
        emissions, _ = wav2vec_model(waveform)
        emissions = torch.log_softmax(emissions, dim=-1)
        
    targets = torch.tensor([tokens], dtype=torch.long, device=device)
    input_lengths = torch.tensor([emissions.shape[1]], dtype=torch.long, device=device)
    target_lengths = torch.tensor([len(tokens)], dtype=torch.long, device=device)
    
    alignments, scores = F.forced_align(emissions, targets, input_lengths, target_lengths)
    alignments = alignments[0].cpu().numpy()
    
    path = decode_alignment(alignments, tokens)
    
    frames = len(alignments)
    dur = len(audio) / 16000.0
    frame_dur = dur / frames
    
    token_spans = {}
    for frame_idx, t_idx in enumerate(path):
        if t_idx is not None:
            if t_idx not in token_spans:
                token_spans[t_idx] = []
            token_spans[t_idx].append(frame_idx)
            
    words = phrase_def['words']
    
    word_tokens = []
    curr_indices = []
    for idx, c in enumerate(clean_text):
        if c == '|':
            if curr_indices:
                word_tokens.append(curr_indices)
                curr_indices = []
        else:
            curr_indices.append(idx)
    if curr_indices:
        word_tokens.append(curr_indices)
        
    word_boundaries = []
    for wi, w_chars in enumerate(word_tokens):
        w_frames = []
        for char_idx in w_chars:
            if char_idx in token_spans:
                w_frames.extend(token_spans[char_idx])
        if w_frames:
            w_frames = sorted(w_frames)
            word_boundaries.append({
                'word': words[wi][0],
                'start': w_frames[0] * frame_dur,
                'end': (w_frames[-1] + 1) * frame_dur
            })
        else:
            word_boundaries.append({
                'word': words[wi][0],
                'start': None,
                'end': None
            })
            
    n_words = len(word_boundaries)
    for wi in range(n_words):
        if word_boundaries[wi]['start'] is None or word_boundaries[wi]['end'] is None:
            left_end = 0.0
            for k in range(wi - 1, -1, -1):
                if word_boundaries[k]['end'] is not None:
                    left_end = word_boundaries[k]['end']
                    break
            right_start = dur
            for k in range(wi + 1, n_words):
                if word_boundaries[k]['start'] is not None:
                    right_start = word_boundaries[k]['start']
                    break
            word_boundaries[wi]['start'] = left_end
            word_boundaries[wi]['end'] = right_start
            
    phonemes_out = []
    pid = 1
    for wi, ((word_str, phonemes), w_bound) in enumerate(zip(words, word_boundaries)):
        w_start = w_bound['start']
        w_end = w_bound['end']
        w_dur = w_end - w_start
        ph_dur = w_dur / len(phonemes)
        
        for pi, ph in enumerate(phonemes):
            ph_start = w_start + pi * ph_dur
            ph_end = w_start + (pi + 1) * ph_dur
            phonemes_out.append({
                "phoneme_id": f"p{pid:03d}",
                "phoneme": ph,
                "word": word_str,
                "word_index": wi,
                "phoneme_index_in_word": pi,
                "start": ph_start,
                "end": ph_end,
                "word_start": w_start,
                "word_end": w_end
            })
            pid += 1
            
    tdata = {
        "phrase": phrase_def['display'],
        "speaker": speaker_id,
        "avg_duration_sec": dur,
        "phonemes": phonemes_out
    }
    return tdata

def main():
    import csv
    recs = []
    with open('xai_reddots/metadata/separated_phrases/my_voice_is_my_password.csv') as f:
        for row in csv.DictReader(f):
            if row['speaker_id'] == 'm0004' and os.path.exists(row['pcm_path']):
                recs.append(row['pcm_path'])
                
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    dictionary = {c: i for i, c in enumerate(labels)}
    
    print(f"Found {len(recs)} recordings for m0004.")
    for idx, path in enumerate(recs, 1):
        tdata = generate_aligned_timeline(path, PHRASE_DEF, 'm0004', model, dictionary, labels)
        # Find boundaries of the word "is"
        is_phones = [ph for ph in tdata['phonemes'] if ph['word'] == 'is']
        print(f"Rec {idx} (dur: {tdata['avg_duration_sec']:.2f}s):")
        print(f"  word 'is' start: {is_phones[0]['word_start']:.3f}s | end: {is_phones[0]['word_end']:.3f}s")
        print(f"  phoneme IH: {is_phones[0]['start']:.3f} - {is_phones[0]['end']:.3f}s")
        print(f"  phoneme Z : {is_phones[1]['start']:.3f} - {is_phones[1]['end']:.3f}s")

if __name__ == "__main__":
    main()
