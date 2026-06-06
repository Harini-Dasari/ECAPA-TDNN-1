import os
import sys
import numpy as np
import soundfile as sf

SR = 16000
HOP = 160
WIN = 400

def rms_envelope(audio):
    hw = WIN // 2
    n = len(audio) // HOP
    out = np.zeros(n)
    for i in range(n):
        c = i * HOP
        seg = audio[max(0, c - hw):min(len(audio), c + hw)]
        out[i] = np.sqrt(np.mean(seg ** 2)) if len(seg) > 0 else 0.0
    return out

def detect_speech_boundaries(audio, threshold=0.03):
    rms = rms_envelope(audio)
    rms_norm = rms / (np.max(rms) + 1e-9)
    rms_t = np.linspace(0, len(audio)/SR, len(rms_norm))
    
    # Find all frames above threshold
    speech_indices = np.where(rms_norm > threshold)[0]
    if len(speech_indices) == 0:
        return 0.0, len(audio)/SR
        
    start_time = rms_t[speech_indices[0]]
    end_time = rms_t[speech_indices[-1]]
    return start_time, end_time

def main():
    import csv
    metadata_csv = 'xai_reddots/metadata/separated_phrases/my_voice_is_my_password.csv'
    recs = []
    with open(metadata_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['speaker_id'] == 'm0004' and os.path.exists(row['pcm_path']):
                recs.append(row['pcm_path'])
                
    print("--- VAD Boundary Detection ---")
    for idx, path in enumerate(recs, 1):
        audio, _ = sf.read(path, channels=1, samplerate=SR, subtype='PCM_16', format='RAW')
        dur = len(audio) / SR
        t_start, t_end = detect_speech_boundaries(audio, threshold=0.04)
        print(f"Rec {idx}: {os.path.basename(path)}")
        print(f"  Total Duration: {dur:.3f}s")
        print(f"  Detected Speech: {t_start:.3f}s to {t_end:.3f}s (dur: {t_end-t_start:.3f}s)")

if __name__ == '__main__':
    main()
