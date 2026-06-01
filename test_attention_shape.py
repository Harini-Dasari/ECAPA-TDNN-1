import os
import glob
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from ECAPAModel import ECAPAModel

def main():
    print("Initializing ECAPA-TDNN model...")
    args = type('Args', (), {})()
    args.C = 1024
    args.m = 0.2
    args.s = 30
    args.n_class = 5994
    args.lr = 0.001
    args.lr_decay = 0.97
    args.test_step = 1
    
    model = ECAPAModel(**vars(args))
    model.load_parameters("exps/pretrain.model")
    model.speaker_encoder.eval()

    # Get a few audio files from RedDots
    pcm_base = "Reddots/pcm"
    test_files = []
    
    # Let's just recursively find 10 pcm files
    for root, dirs, files in os.walk(pcm_base):
        for file in files:
            if file.endswith('.pcm'):
                test_files.append(os.path.join(root, file))
                if len(test_files) == 10:
                    break
        if len(test_files) == 10:
            break

    print(f"Found {len(test_files)} files. Running inference...")
    
    for full_path in test_files:
        audio, _ = sf.read(full_path, channels=1, samplerate=16000, subtype='PCM_16', format='RAW')
        data_1 = torch.FloatTensor(np.stack([audio],axis=0)).cuda()
        
        with torch.no_grad():
            _ = model.speaker_encoder.forward(data_1, aug=False)

if __name__ == '__main__':
    main()
