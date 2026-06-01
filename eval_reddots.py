import os
import glob
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ECAPAModel import ECAPAModel
from tools import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf
import csv
import argparse

def main():
    os.makedirs('exps_reddots', exist_ok=True)
    
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

    print("Parsing RedDots dataset structure...")
    enrollment_file = "Reddots/ndx/m_part_01.trn"
    trial_file = "Reddots/ndx/red_dot_trail.txt"
    pcm_base = "Reddots/pcm"

    enrollment_dict = {} # enrollment_id -> list of paths
    with open(enrollment_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            enroll_id = parts[0]
            audio_paths = parts[1].split(',')
            enrollment_dict[enroll_id] = audio_paths

    labels = []
    trials = []
    with open(trial_file, 'r') as f:
        for line in f:
            label, enroll_id, test_audio = line.strip().split(',')
            labels.append(int(label))
            trials.append((enroll_id, test_audio))

    # Collect all unique audios needed to extract embeddings
    unique_audios = set()
    for paths in enrollment_dict.values():
        unique_audios.update(paths)
    for enroll_id, test_audio in trials:
        unique_audios.add(test_audio)
        
    print(f"Total unique audios to extract: {len(unique_audios)}")

    embeddings = {}
    print("Extracting embeddings for all audios...")
    for audio_rel_path in tqdm(list(unique_audios)):
        full_path = os.path.join(pcm_base, audio_rel_path + ".pcm")
        if not os.path.exists(full_path):
            print(f"WARNING: File not found {full_path}")
            continue
            
        audio, _ = sf.read(full_path, channels=1, samplerate=16000, subtype='PCM_16', format='RAW')
        
        # Exact extraction logic from original eval_network
        data_1 = torch.FloatTensor(np.stack([audio],axis=0)).cuda()
        
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
            
        feats = []
        startframe = np.linspace(0, audio.shape[0]-max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
        feats = np.stack(feats, axis=0).astype(np.float64)
        data_2 = torch.FloatTensor(feats).cuda()
        
        with torch.no_grad():
            embedding_1 = model.speaker_encoder.forward(data_1, aug=False)
            embedding_1 = F.normalize(embedding_1, p=2, dim=1)
            embedding_2 = model.speaker_encoder.forward(data_2, aug=False)
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            
        embeddings[audio_rel_path] = [embedding_1, embedding_2]

    # Pre-compute average enrollment embeddings for efficiency
    enrollment_embeddings = {}
    print("Computing enrollment speaker profiles...")
    for enroll_id, paths in enrollment_dict.items():
        emb1_list = []
        emb2_list = []
        for p in paths:
            if p in embeddings:
                emb1_list.append(embeddings[p][0])
                emb2_list.append(embeddings[p][1])
        if emb1_list:
            avg_emb1 = torch.mean(torch.stack(emb1_list), dim=0)
            avg_emb2 = torch.mean(torch.stack(emb2_list), dim=0)
            avg_emb1 = F.normalize(avg_emb1, p=2, dim=1)
            avg_emb2 = F.normalize(avg_emb2, p=2, dim=1)
            enrollment_embeddings[enroll_id] = [avg_emb1, avg_emb2]

    print("Computing trial scores...")
    scores = []
    final_labels = []
    
    for (enroll_id, test_audio), label in tqdm(zip(trials, labels), total=len(trials)):
        if enroll_id not in enrollment_embeddings or test_audio not in embeddings:
            continue
            
        e11, e12 = enrollment_embeddings[enroll_id]
        e21, e22 = embeddings[test_audio]
        
        score_1 = torch.mean(torch.matmul(e11, e21.T))
        score_2 = torch.mean(torch.matmul(e12, e22.T))
        score = (score_1 + score_2) / 2
        score = score.detach().cpu().numpy()
        
        scores.append(score)
        final_labels.append(label)

    print("Calculating EER and Thresholds...")
    EER = tuneThresholdfromScore(scores, final_labels, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, final_labels)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

    idxE = np.nanargmin(np.absolute((np.array(fnrs) - np.array(fprs))))
    eer_threshold = thresholds[idxE]
    
    print(f"\nFinal Results:")
    print(f"EER: {EER:.4f}%")
    print(f"minDCF: {minDCF:.4f}%")
    print(f"EER Threshold: {eer_threshold:.6f}")

    # Save to CSV
    print("Saving metrics...")
    with open('exps_reddots/eval_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Threshold', 'FAR', 'FRR'])
        for t, far, frr in zip(thresholds, fprs, fnrs):
            writer.writerow([t, far, frr])

    print("Saving trial predictions...")
    with open('exps_reddots/trial_predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Enrollment_ID', 'Test_Audio', 'GroundTruth', 'Score', f'Prediction_{eer_threshold:.4f}', 'IsCorrect'])
        for (enroll_id, test_audio), s, l in zip(trials, scores, final_labels):
            pred = 1 if s >= eer_threshold else 0
            writer.writerow([enroll_id, test_audio, l, s, pred, pred == l])

    print("Evaluation Complete!")

if __name__ == '__main__':
    main()
