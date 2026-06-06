import torch
import torchaudio
import torchaudio.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    print("Alignments length:", len(alignments))
    print("Alignments raw values:")
    print(alignments.tolist())

if __name__ == "__main__":
    main()
