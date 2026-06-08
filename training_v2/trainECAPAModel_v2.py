'''
trainECAPAModel_v2.py — Fine-tuning launcher for Method 1 (ECAPA_TDNN_A).

Changes vs. trainECAPAModel.py:
  - Imports ECAPAModel_v2 instead of ECAPAModel
  - Default save_path → training_v2/exps_modelA
  - Default initial_model → exps/pretrain.model   (load pretrain weights by default)
  - Default lr → 0.0001  (10× lower for fine-tuning)
  - Default lr_decay → 0.90  (faster decay for fine-tuning)
  - Default max_epoch → 15   (short fine-tuning run)

Usage:
  Stage 1 — Direct evaluation (no training, ~10 min):
    python -m training_v2.trainECAPAModel_v2 --eval

  Stage 2 — Fine-tuning (needs VoxCeleb2 training data):
    python -m training_v2.trainECAPAModel_v2 \
      --train_list /path/to/VoxCeleb2/train_list.txt \
      --train_path /path/to/VoxCeleb2/train/wav \
      --musan_path /path/to/musan_split \
      --rir_path   /path/to/RIRS_NOISES/simulated_rirs
'''

import argparse, glob, os, sys, torch, warnings, time

# Allow importing tools/loss/dataLoader from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools import *
from dataLoader import train_loader
from training_v2.ECAPAModel_v2 import ECAPAModel_v2

parser = argparse.ArgumentParser(description="ECAPA_trainer_v2 — Method 1: Temporal Aggregation")

## Training settings
parser.add_argument('--num_frames', type=int,   default=200,    help='Duration of the input segments (200 = 2 sec)')
parser.add_argument('--max_epoch',  type=int,   default=15,     help='Maximum number of epochs (15 for fine-tuning)')
parser.add_argument('--batch_size', type=int,   default=400,    help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=4,      help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,      help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.0001, help='Learning rate (0.0001 = 10x lower for fine-tuning)')
parser.add_argument("--lr_decay",   type=float, default=0.90,   help='LR decay every [test_step] epochs (0.90 = faster for fine-tuning)')

## Training and evaluation path/lists, save path
parser.add_argument('--train_list', type=str,   default="/data08/VoxCeleb2/train_list.txt",
                    help='Path to VoxCeleb2 training list')
parser.add_argument('--train_path', type=str,   default="/data08/VoxCeleb2/train/wav",
                    help='Path to VoxCeleb2 training audio (wav)')
parser.add_argument('--eval_list',  type=str,   default="Voxceleb/veri_test2.txt",
                    help='Path to evaluation trial list')
parser.add_argument('--eval_path',  type=str,   default="Voxceleb/",
                    help='Path to evaluation audio directory')
parser.add_argument('--musan_path', type=str,   default="/data08/Others/musan_split",
                    help='Path to MUSAN noise set (for augmentation)')
parser.add_argument('--rir_path',   type=str,   default="/data08/Others/RIRS_NOISES/simulated_rirs",
                    help='Path to RIR set (for augmentation)')
parser.add_argument('--save_path',  type=str,   default="training_v2/exps_modelA",
                    help='Directory to save score.txt and model checkpoints')
parser.add_argument('--initial_model', type=str, default="exps/pretrain.model",
                    help='Path to initial model — default: pretrained baseline')

## Model and loss settings (must match pretrain.model exactly)
parser.add_argument('--C',       type=int,   default=1024, help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.2,  help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,   help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=5994, help='Number of speakers in VoxCeleb2')

## Command
parser.add_argument('--eval', dest='eval', action='store_true', help='Only do evaluation (Stage 1)')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

## Search for existing checkpoints
modelfiles = glob.glob('%s/model_0*.model' % args.model_save_path)
modelfiles.sort()

## ─── Stage 1: Direct Evaluation ───────────────────────────────────────────
if args.eval == True:
    s = ECAPAModel_v2(**vars(args))
    print("Model %s loaded from previous state!" % args.initial_model)
    s.load_parameters(args.initial_model)
    EER, minDCF = s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path)
    print("EER %2.2f%%, minDCF %.4f%%" % (EER, minDCF))
    quit()

## ─── Stage 2: Fine-Tuning ─────────────────────────────────────────────────
## Define the data loader
trainloader = train_loader(**vars(args))
trainLoader = torch.utils.data.DataLoader(
    trainloader, batch_size=args.batch_size, shuffle=True,
    num_workers=args.n_cpu, drop_last=True
)

## Load from initial_model (pretrain.model by default)
if args.initial_model != "":
    print("Model %s loaded from previous state!" % args.initial_model)
    s = ECAPAModel_v2(**vars(args))
    s.load_parameters(args.initial_model)
    epoch = 1

## Or resume from last checkpoint in save_path
elif len(modelfiles) >= 1:
    print("Model %s loaded from previous state!" % modelfiles[-1])
    epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
    s = ECAPAModel_v2(**vars(args))
    s.load_parameters(modelfiles[-1])

## Train from scratch (not recommended — use pretrained weights)
else:
    epoch = 1
    s = ECAPAModel_v2(**vars(args))

EERs = []
score_file = open(args.score_save_path, "a+")

while True:
    ## Training for one epoch
    loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader)

    ## Evaluation every [test_step] epochs
    if epoch % args.test_step == 0:
        s.save_parameters(args.model_save_path + "/model_%04d.model" % epoch)
        EERs.append(s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path)[0])
        print(time.strftime("%Y-%m-%d %H:%M:%S"),
              "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%" %
              (epoch, acc, EERs[-1], min(EERs)))
        score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n" %
                         (epoch, lr, loss, acc, EERs[-1], min(EERs)))
        score_file.flush()

    if epoch >= args.max_epoch:
        quit()

    epoch += 1
