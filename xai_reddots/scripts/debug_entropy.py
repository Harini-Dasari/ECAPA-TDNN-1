import os
import torch
import math
import numpy as np
import soundfile as sf
from ECAPAModel import ECAPAModel

def test_entropy():
    # Load model
    model = ECAPAModel(lr=0.001, lr_decay=0.97, C=1024, n_class=322, m=0.2, s=30, test_step=1)
    model.load_state_dict(torch.load("exps/pre_train/model/model_0034.model", map_location='cpu'))
    model.eval()

    # Load audio
    pcm_path = "Reddots/pcm/m0001/q01/m0001_133_q01.pcm"
    audio, _ = sf.read(pcm_path, channels=1, samplerate=16000, subtype='PCM_16', format='RAW')
    data_1 = torch.FloatTensor(audio).unsqueeze(0)

    with torch.no_grad():
        x = model.speaker_encoder.torchfbank(data_1)
        x = x - torch.mean(x, dim=-1, keepdim=True)
        x = model.speaker_encoder.conv1(x)
        x = model.speaker_encoder.relu(x)
        x = model.speaker_encoder.bn1(x)
        x = model.speaker_encoder.layer1(x)
        x = model.speaker_encoder.layer2(x)
        x = model.speaker_encoder.layer3(x)
        x = model.speaker_encoder.layer4(x)
        x = model.speaker_encoder.layer4[0].relu(x)

        # Attention layer logic
        w_logits = model.speaker_encoder.attention.weight_bn(model.speaker_encoder.attention.relu(model.speaker_encoder.attention.tdnn(x)))
        a = torch.softmax(w_logits, dim=1)

        # H calculation
        H = -torch.sum(a * torch.log(a + 1e-9), dim=1)
        C_channels = a.shape[1]
        
        # Confidence calculation
        confidence = 1.0 - H / math.log(C_channels)

    H_np = H.squeeze().cpu().numpy()
    conf_np = confidence.squeeze().cpu().numpy()

    print("Entropy min:", H_np.min())
    print("Entropy max:", H_np.max())
    print("Entropy mean:", H_np.mean())
    print("-" * 20)
    print("Confidence min:", conf_np.min())
    print("Confidence max:", conf_np.max())
    print("Confidence mean:", conf_np.mean())

if __name__ == "__main__":
    test_entropy()
