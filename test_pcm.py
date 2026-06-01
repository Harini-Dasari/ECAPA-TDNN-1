import soundfile as sf
import os
import numpy as np

pcm_path = 'Reddots/pcm/m0001/20150129213253016_m0001_36.pcm'

try:
    # Try reading as WAV first, maybe it has a header?
    audio, sr = sf.read(pcm_path)
    print(f"Read normally. SR: {sr}, Audio shape: {audio.shape}")
except Exception as e:
    print(f"Normal read failed: {e}")
    try:
        # Try reading as RAW PCM
        audio, sr = sf.read(pcm_path, channels=1, samplerate=16000, subtype='PCM_16', format='RAW')
        print(f"Read as RAW PCM_16. SR: {sr}, Audio shape: {audio.shape}")
    except Exception as e2:
        print(f"RAW read failed: {e2}")
