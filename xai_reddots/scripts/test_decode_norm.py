import torch
import torchaudio
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    pcm_path = "Reddots/pcm/m0004/20150630144138886_m0004_31.pcm"
    
    with open(pcm_path, 'rb') as f:
        pcm_data = f.read()
    audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.tensor(audio).unsqueeze(0).to(device) # [1, T]
    
    # Standard Wav2Vec2 normalization: zero mean, unit variance
    waveform_norm = (waveform - waveform.mean()) / (waveform.std() + 1e-9)
    
    # Test Wav2Vec2 ASR Base
    print("--- Testing WAV2VEC2_ASR_BASE_960H ---")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    
    for name, w in [("Raw", waveform), ("Normalized", waveform_norm)]:
        with torch.inference_mode():
            emissions, _ = model(w)
            emissions = torch.log_softmax(emissions, dim=-1)
        indices = torch.argmax(emissions, dim=-1)[0].cpu().numpy()
        decoded = [labels[idx] for idx in indices if idx != 0] # skip blank
        collapsed = []
        for c in decoded:
            if not collapsed or c != collapsed[-1]:
                collapsed.append(c)
        print(f"  {name} Greedy Decoded:", "".join(collapsed))
        
    # Test MMS_FA
    print("\n--- Testing MMS_FA ---")
    bundle_mms = torchaudio.pipelines.MMS_FA
    model_mms = bundle_mms.get_model().to(device)
    labels_mms = bundle_mms.get_labels()
    
    for name, w in [("Raw", waveform), ("Normalized", waveform_norm)]:
        with torch.inference_mode():
            emissions, _ = model_mms(w)
            emissions = torch.log_softmax(emissions, dim=-1)
        indices = torch.argmax(emissions, dim=-1)[0].cpu().numpy()
        decoded = [labels_mms[idx] for idx in indices if idx != 0] # skip blank
        collapsed = []
        for c in decoded:
            if not collapsed or c != collapsed[-1]:
                collapsed.append(c)
        print(f"  {name} Greedy Decoded:", "".join(collapsed))

if __name__ == "__main__":
    main()
