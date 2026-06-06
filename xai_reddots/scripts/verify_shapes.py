import os
import torch
import math
import numpy as np
import soundfile as sf
import sys

sys.path.append(os.getcwd())
from ECAPAModel import ECAPAModel

def main():
    print("Loading model...")
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
    
    # Find multiple pcm files
    metadata_csv = 'xai_reddots/metadata/phrase_groups.csv'
    recordings = []
    if os.path.exists(metadata_csv):
        import csv
        with open(metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                recordings.append(row['pcm_path'])
    else:
        # Fallback to recursive walk
        pcm_base = "Reddots/pcm"
        for root, dirs, files in os.walk(pcm_base):
            for file in files:
                if file.endswith('.pcm'):
                    recordings.append(os.path.join(root, file))
                    if len(recordings) == 5:
                        break
            if len(recordings) == 5:
                break
                
    print("\n--- Channel Discriminability & Shape Verification ---")
    for idx, pcm_path in enumerate(recordings[:5]):
        if not os.path.exists(pcm_path):
            print(f"File not found: {pcm_path}")
            continue
            
        audio, _ = sf.read(pcm_path, channels=1, samplerate=16000, subtype='PCM_16', format='RAW')
        data_1 = torch.FloatTensor(np.stack([audio],axis=0)).cuda()
        
        with torch.no_grad():
            x = model.speaker_encoder.torchfbank(data_1) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            
            x = model.speaker_encoder.conv1(x)
            x = model.speaker_encoder.relu(x)
            x = model.speaker_encoder.bn1(x)
            
            x1 = model.speaker_encoder.layer1(x)
            x2 = model.speaker_encoder.layer2(x+x1)
            x3 = model.speaker_encoder.layer3(x+x1+x2)
            
            x = model.speaker_encoder.layer4(torch.cat((x1,x2,x3),dim=1))
            x = model.speaker_encoder.relu(x)
            
            t = x.size()[-1]
            global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
            
            w_logits = global_x
            for i in range(len(model.speaker_encoder.attention) - 1):
                w_logits = model.speaker_encoder.attention[i](w_logits)
                
            a = torch.softmax(w_logits, dim=1)
            H = -torch.sum(a * torch.log(a + 1e-9), dim=1)
            
            C_channels = a.shape[1]
            confidence = 1.0 - H / math.log(C_channels)
            
            std_val = w_logits.std().item()
            mean_val = confidence.mean().item()
            min_val = confidence.min().item()
            max_val = confidence.max().item()
            
            print(f"Utterance {idx+1}: {os.path.basename(pcm_path)}")
            print(f"  w_logits.shape: {list(w_logits.shape)}")
            print(f"  w_logits.std(): {std_val:.6f}")
            print(f"  confidence:     min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")

if __name__ == "__main__":
    main()
