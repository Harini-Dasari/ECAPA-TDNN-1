import os
import csv
import torch
import numpy as np
import soundfile as sf
import sys

sys.path.append(os.getcwd())

# Import the new Model A
from training_v2.ECAPAModel_v2 import ECAPAModel_v2 as ECAPAModel
from xai_reddots_temporal.model_hooks import register_temporal_hook, hook_output

def main():
    metadata_csv = 'xai_reddots/metadata/phrase_groups.csv'
    output_csv = 'xai_reddots_temporal/outputs/temporal_alpha.csv'
    
    print("Loading Temporal Method-1 Model...")
    args = type('Args', (), {})()
    args.C = 1024
    args.m = 0.2
    args.s = 30
    args.n_class = 5994
    args.lr = 0.001
    args.lr_decay = 0.97
    args.test_step = 1
    
    model = ECAPAModel(**vars(args))
    
    # Load the fine-tuned Method 1 model (Epoch 13)
    model.load_parameters("training_v2/exps_modelA/model/model_0013.model")
    model.speaker_encoder.eval()
    
    # Register hook to extract alpha_hat securely
    register_temporal_hook(model.speaker_encoder)
    
    # Read metadata
    recordings = []
    with open(metadata_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            recordings.append(row)
            
    print(f"Extracting temporal alpha_hat for {len(recordings)} recordings...")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['recording_id', 'frame_id', 'time', 'alpha_hat'])
        writer.writeheader()
        
        for idx, rec in enumerate(recordings):
            pcm_path = rec['pcm_path']
            rec_id = rec['recording_id']
            
            # Use same reading mechanism
            audio, _ = sf.read(pcm_path, channels=1, samplerate=16000, subtype='PCM_16', format='RAW')
            data_1 = torch.FloatTensor(np.stack([audio],axis=0)).cuda()
            
            with torch.no_grad():
                # Forward pass will trigger the hook and store alpha_hat
                _ = model.speaker_encoder.forward(data_1, aug=False)
                
                # Retrieve from hook
                alpha_hat = hook_output['alpha_hat'].squeeze().numpy()
                
            hop_time = 160 / 16000.0 # 0.01 seconds
            for frame_id, alpha_val in enumerate(alpha_hat):
                time_sec = frame_id * hop_time
                writer.writerow({
                    'recording_id': rec_id,
                    'frame_id': frame_id,
                    'time': f"{time_sec:.2f}",
                    'alpha_hat': f"{alpha_val:.6f}"
                })
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx+1}/{len(recordings)}")
                
    print("Done extracting temporal alpha_hat.")

if __name__ == "__main__":
    main()
