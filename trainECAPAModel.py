'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, os, torch, warnings, time, sys
from tools import *
from dataLoader import train_loader
from ECAPAModel import ECAPAModel
import csv
import matplotlib.pyplot as plt

# Add a function to evaluate specific thresholds
def evaluate_fixed_thresholds(model, eval_list, eval_path, thresholds, output_dir):
	"""
	Evaluate the model at specific thresholds using cached scores from eval_network.
	This is fast because it uses pre-computed scores instead of recomputing embeddings.
	"""
	from tools import ComputeErrorRates
	
	# Use cached scores from the model (computed during eval_network)
	scores = model.cached_scores
	labels = model.cached_labels
	
	# Compute metrics for each threshold
	fnrs, fprs, computed_thresholds = ComputeErrorRates(scores, labels)
	
	results = {}
	for threshold in thresholds:
		# Find the closest threshold in the computed thresholds
		idx = 0
		for i, t in enumerate(computed_thresholds):
			if abs(t - threshold) < abs(computed_thresholds[idx] - threshold):
				idx = i
		
		# Compute FAR and FRR at this threshold
		# FAR = FPR (False Positive Rate)
		# FRR = FNR (False Negative Rate)
		far = fprs[idx] / (fprs[idx] + (len(labels) - sum(labels)))  # FP / (FP + TN)
		frr = fnrs[idx] / sum(labels)  # FN / (FN + TP)
		
		# EER approximation at this threshold
		eer = (far + frr) / 2
		
		results[threshold] = {'EER': eer, 'FAR': far, 'FRR': frr}
		print(f"Threshold: {threshold:.3f}, EER: {eer:.4f}, FAR: {far:.4f}, FRR: {frr:.4f}")
		sys.stdout.flush()
	
	return results

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=400,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=4,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--train_list', type=str,   default="/data08/VoxCeleb2/train_list.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--train_path', type=str,   default="/data08/VoxCeleb2/train/wav",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
parser.add_argument('--eval_list',  type=str,   default="/data08/VoxCeleb1/veri_test2.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--eval_path',  type=str,   default="/data08/VoxCeleb1/test/wav",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
parser.add_argument('--musan_path', type=str,   default="/data08/Others/musan_split",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--rir_path',   type=str,   default="/data08/Others/RIRS_NOISES/simulated_rirs",     help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case');
parser.add_argument('--save_path',  type=str,   default="exps/exp1",                                     help='Path to save the score.txt and models')
parser.add_argument('--initial_model',  type=str,   default="",                                          help='Path of the initial_model')

## Model and Loss settings
parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=5994,   help='Number of speakers')

## Command
parser.add_argument('--visualize_scores', dest='visualize_scores', action='store_true', help='Visualize score distributions')
## parser.add_argument('--visualize_scores', dest='visualize_scores', action='store_true', help='Visualize score distributions')
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

## Define the data loader
if not args.eval:
    trainloader = train_loader(**vars(args))
    trainLoader = torch.utils.data.DataLoader(
        trainloader, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True
    )

## Search for the exist models
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()

## Only do evaluation, the initial_model is necessary
if args.eval == True:
	s = ECAPAModel(**vars(args))
	print("Model %s loaded from previous state!"%args.initial_model)
	s.load_parameters(args.initial_model)
	EER, minDCF = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)
	print("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
	sys.stdout.flush()
	
	# Evaluate fixed thresholds
	print("\n========== FIXED THRESHOLDS EVALUATION ==========")
	sys.stdout.flush()
	print("Debugging: --eval flag detected. Starting evaluation for fixed thresholds.")
	sys.stdout.flush()
	fixed_thresholds = [0.30, 0.31, 0.32, 0.33, 0.34]
	print(f"Debugging: Fixed thresholds to evaluate - {fixed_thresholds}")
	sys.stdout.flush()
	
	# Log to file to ensure code execution
	log_file = os.path.join(args.save_path, "eval_debug.log")
	with open(log_file, "w") as f:
		f.write("Starting fixed threshold evaluation\n")
	
	results = evaluate_fixed_thresholds(
		model=s,
		eval_list=args.eval_list,
		eval_path=args.eval_path,
		thresholds=fixed_thresholds,
		output_dir=args.save_path
	)
	# Enhanced debugging output
	print("Debugging: Starting to save results.")
	sys.stdout.flush()
	print(f"Debugging: Results dictionary: {results}")
	sys.stdout.flush()
	
	# Ensure output directory exists
	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)
		print(f"Debugging: Created directory {args.save_path}")
		sys.stdout.flush()
	
	# Save results to a CSV file
	csv_file_path = os.path.join(args.save_path, "threshold_030_034_results.csv")
	try:
		with open(csv_file_path, "w", newline="") as csvfile:
			csvwriter = csv.writer(csvfile)
			csvwriter.writerow(["Threshold", "EER", "FAR", "FRR"])
			for threshold, metrics in results.items():
				csvwriter.writerow([threshold, metrics['EER'], metrics['FAR'], metrics['FRR']])
		print(f"Debugging: CSV file successfully saved to {csv_file_path}")
		sys.stdout.flush()
	except Exception as e:
		print(f"Debugging: Error saving CSV file: {e}")
		sys.stdout.flush()
	
	# Save FAR/FRR vs Threshold plot
	thresholds = list(results.keys())
	far = [metrics['FAR'] for metrics in results.values()]
	frr = [metrics['FRR'] for metrics in results.values()]

	plt.figure()
	plt.plot(thresholds, far, label='FAR', color='red')
	plt.plot(thresholds, frr, label='FRR', color='blue')
	plt.title('FAR/FRR vs Threshold')
	plt.xlabel('Threshold')
	plt.ylabel('Error Rate')
	plt.legend()
	plot_file_path = os.path.join(args.save_path, "far_frr_vs_threshold_030_034.png")
	try:
		plt.savefig(plot_file_path)
		plt.close()
		print(f"Debugging: Plot successfully saved to {plot_file_path}")
		sys.stdout.flush()
	except Exception as e:
		print(f"Debugging: Error saving plot: {e}")
		sys.stdout.flush()
	
	# Final confirmation
	if os.path.exists(csv_file_path):
		print(f"✓ CSV file confirmed saved: {csv_file_path}")
	else:
		print(f"✗ CSV file NOT found at: {csv_file_path}")
	sys.stdout.flush()
	
	quit()

## If initial_model is exist, system will train from the initial_model
if args.initial_model != "":
	print("Model %s loaded from previous state!"%args.initial_model)
	s = ECAPAModel(**vars(args))
	s.load_parameters(args.initial_model)
	epoch = 1

## Otherwise, system will try to start from the saved model&epoch
elif len(modelfiles) >= 1:
	print("Model %s loaded from previous state!"%modelfiles[-1])
	epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
	s = ECAPAModel(**vars(args))
	s.load_parameters(modelfiles[-1])
## Otherwise, system will train from scratch
else:
	epoch = 1
	s = ECAPAModel(**vars(args))

EERs = []
score_file = open(args.score_save_path, "a+")

while(1):
	## Training for one epoch
	loss, lr, acc = s.train_network(epoch = epoch, loader = trainLoader)

	## Evaluation every [test_step] epochs
	if epoch % args.test_step == 0:
		s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
		EERs.append(s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)[0])
		print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"%(epoch, acc, EERs[-1], min(EERs)))
		score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"%(epoch, lr, loss, acc, EERs[-1], min(EERs)))
		score_file.flush()

	if epoch >= args.max_epoch:
		quit()

	epoch += 1
