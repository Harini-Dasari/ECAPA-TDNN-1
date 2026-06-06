import torch
import torchaudio
import torchaudio.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            # Lookahead up to 3 tokens for skipping
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

def main():
    pcm_path = "Reddots/pcm/m0004/20150630144138886_m0004_31.pcm"
    transcript = "My voice is my password"
    
    with open(pcm_path, 'rb') as f:
        pcm_data = f.read()
    audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.tensor(audio).unsqueeze(0).to(device)
    
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    dictionary = {c: i for i, c in enumerate(labels)}
    
    # Preprocess transcript: replace spaces with | and lowercase to match bundle
    # Wait, Wav2Vec2_ASR_BASE_960H expects uppercase characters
    clean_text = transcript.upper().replace(" ", "|")
    tokens = [dictionary[c] for c in clean_text if c in dictionary]
    
    with torch.inference_mode():
        emissions, _ = model(waveform)
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
    
    # Calculate boundaries for each target index
    token_spans = {}
    for frame_idx, t_idx in enumerate(path):
        if t_idx is not None:
            if t_idx not in token_spans:
                token_spans[t_idx] = []
            token_spans[t_idx].append(frame_idx)
            
    print("Decoded aligned tokens:")
    for t_idx in sorted(token_spans.keys()):
        frame_list = token_spans[t_idx]
        s_f, e_f = frame_list[0], frame_list[-1] + 1
        s_t, e_t = s_f * frame_dur, e_f * frame_dur
        print(f"  Token {t_idx} ({clean_text[t_idx]}): {s_t:.3f} - {e_t:.3f}s")

if __name__ == "__main__":
    main()
