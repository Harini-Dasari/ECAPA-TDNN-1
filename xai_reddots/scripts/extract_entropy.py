import os
import csv
import math
import torch
import numpy as np
import soundfile as sf
import sys

sys.path.append(os.getcwd())

from ECAPAModel import ECAPAModel

def main():
    metadata_csv = 'xai_reddots/metadata/phrase_groups.csv'
    output_csv = 'xai_reddots/entropy/frame_entropy.csv'
    
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
    
    # Read metadata
    recordings = []
    with open(metadata_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            recordings.append(row)
            
    print(f"Extracting entropy for {len(recordings)} recordings...")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['recording_id', 'frame_id', 'time', 'entropy'])
        writer.writeheader()
        
        for idx, rec in enumerate(recordings):
            pcm_path = rec['pcm_path']
            rec_id = rec['recording_id']
            
            audio, _ = sf.read(pcm_path, channels=1, samplerate=16000, subtype='PCM_16', format='RAW')
            data_1 = torch.FloatTensor(np.stack([audio],axis=0)).cuda()
            
            with torch.no_grad():
                # Replicate the forward pass up to attention to extract entropy
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
                
                # Normalize across frames to get Entropy-Based Attention (alpha_hat)
                alpha_hat = confidence / torch.sum(confidence, dim=1, keepdim=True)
                
                entropy_profile = alpha_hat.squeeze().cpu().numpy()
                
            hop_time = 160 / 16000.0 # 0.01 seconds
            for frame_id, ent in enumerate(entropy_profile):
                time_sec = frame_id * hop_time
                writer.writerow({
                    'recording_id': rec_id,
                    'frame_id': frame_id,
                    'time': f"{time_sec:.2f}",
                    'entropy': f"{ent:.6f}"
                })
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx+1}/{len(recordings)}")
                
    print("Done extracting entropy.")

if __name__ == "__main__":
    main()
