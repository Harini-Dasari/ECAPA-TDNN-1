'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
import torch.nn.functional as F
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN
import matplotlib.pyplot as plt
import numpy as np

class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, **kwargs):
		super(ECAPAModel, self).__init__()
		## create model on gpu ECAPA-TDNN
		self.speaker_encoder = ECAPA_TDNN(C = C).cuda()
		## speaker Classifier Training not used in inference, so we can set m = 0 to disable the margin penalty
		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).cuda()
		## optimizer for training
		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		##Automatically reduce the learning rate by lr_decay every test_step epochs
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		##read current learning rate from optimizer
		lr = self.optim.param_groups[0]['lr']
		##these load batches of data and labels, and train the model
		##Batch 1: data   -> audio tensors
		#  labels -> speaker ID
		for num, (data, labels) in enumerate(loader, start = 1):

			self.zero_grad()
			labels            = torch.LongTensor(labels).cuda()
			##raw data->speaker embedding->loss->backpropagation
			#for loss we use AAMsoftmax, which is a common loss function for speaker verification tasks,
			#  it adds an angular margin to enhance the discriminative power of the embeddings.
			speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug = True)
			nloss, prec       = self.speaker_loss.forward(speaker_embedding, labels)			
			nloss.backward()
			#updqate the weights
			self.optim.step()

			#update matrix for calculating the training accuracy and loss
			index += len(labels)
			top1 += prec
			loss += nloss.detach().cpu().numpy()
			#detach() means removes the tensor from the computation graph, 
			# so it won't be used for backpropagation. .cpu() moves the tensor to the CPU memory,
			#  and .numpy() converts it to a NumPy array for easier manipulation and logging.
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
			" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
			sys.stderr.flush()
		sys.stdout.write("\n")
		return loss/num, lr, top1/index*len(labels)

	def eval_network(self, eval_list, eval_path):
		self.eval()
		files = []
		embeddings = {}
		# Read the evaluation list and extract unique audio files
		lines = open(eval_list).read().splitlines()
		for line in lines:
			files.append(line.split()[1])
			files.append(line.split()[2])
		setfiles = list(set(files))
		setfiles.sort()

		# Extract speaker embeddings for each unique audio file
		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			# Load the audio file converted to a tensor, and move it to the GPU
			audio, _  = soundfile.read(os.path.join(eval_path, file))
			# Full utterance
			data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()

			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				# Pad the audio by repeating it until it reaches the required length
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []

			# Generate 5 segments of the audio by sliding a window across it, and store them in a list
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			for asf in startframe:
				#store the segments in a list, and convert it to a tensor for processing
				feats.append(audio[int(asf):int(asf)+max_audio])
			feats = numpy.stack(feats, axis = 0).astype(numpy.float32)
			# Convert the list of segments into a tensor and move it to the GPU
			data_2 = torch.FloatTensor(feats).cuda()
			# Speaker embeddings
			with torch.no_grad():
				# Pass the full utterance and the segmented utterances through 
				# the speaker encoder to obtain their embeddings, and normalize them to have unit length
				embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2 = self.speaker_encoder.forward(data_2, aug = False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			embeddings[file] = [embedding_1, embedding_2]
		scores, labels  = [], []

		# Compute the scores for each pair of audio files in the evaluation list 
		# by taking the mean of the dot product of their embeddings, 
		# and store the scores and labels for later use
		for line in lines:			
			embedding_11, embedding_12 = embeddings[line.split()[1]]
			embedding_21, embedding_22 = embeddings[line.split()[2]]
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			score = (score_1 + score_2) / 2
			score = score.detach().cpu().numpy()
			scores.append(score)
			labels.append(int(line.split()[0]))
			
		# Coumpute EER and minDCF
		# EER = tuneThresholdfromScore(scores, labels, [1,0.1])[1]
		# fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		# minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

		# return EER, minDCF
		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [1,0.1])[1]
		#false negative rates (FNRs), false positive rates (FPRs), and corresponding thresholds for the given scores and labels.
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		# The ComputeMinDcf function calculates the minimum Detection Cost Function (minDCF)
		#  based on the FNRs, FPRs, thresholds,and specified parameters (0.05, 1, 1 in this case).
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

		# Cache scores and labels for later use
		self.cached_scores = scores
		self.cached_labels = labels

# Generate plots
		plot_score_distributions(scores, labels)
		fine_grained_threshold_sweep(scores, labels)

		return EER, minDCF
	
	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	# Load model parameters from a specified path, ensuring compatibility with the current model architecture.	
	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path, map_location = "cpu")
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)

# Visualize score distributions
def plot_score_distributions(scores, labels):
    genuine_scores = [scores[i] for i in range(len(scores)) if labels[i] == 1]
    impostor_scores = [scores[i] for i in range(len(scores)) if labels[i] == 0]

    plt.hist(genuine_scores, bins=50, alpha=0.5, label='Genuine', color='blue')
    plt.hist(impostor_scores, bins=50, alpha=0.5, label='Impostor', color='red')
    plt.title('Score Distributions')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('exps/eval_gpu/score_distributions_updated.png')  # Save updated plot

# Fine-grained threshold sweep
def fine_grained_threshold_sweep(scores, labels):
    thresholds = np.arange(-1.0, 1.0, 0.001)  # Updated range and step size
    far = []
    frr = []

    for threshold in thresholds:
        false_accepts = sum(1 for i in range(len(scores)) if scores[i] >= threshold and labels[i] == 0)
        false_rejects = sum(1 for i in range(len(scores)) if scores[i] < threshold and labels[i] == 1)
        total_genuine = sum(1 for label in labels if label == 1)
        total_impostor = sum(1 for label in labels if label == 0)

        far.append(false_accepts / total_impostor)
        frr.append(false_rejects / total_genuine)

    # Save the results to a CSV file
    import csv
    with open('exps/threshold_results_updated.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['threshold', 'far', 'frr'])
        for i, threshold in enumerate(thresholds):
            csvwriter.writerow([threshold, far[i], frr[i]])

    plt.plot(thresholds, far, label='FAR', color='red')
    plt.plot(thresholds, frr, label='FRR', color='blue')
    plt.title('FAR/FRR vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.savefig('exps/eval_gpu/far_frr_vs_threshold_updated.png')  # Save updated plot

# Call the functions after evaluation
# plot_score_distributions(scores, labels)
# fine_grained_threshold_sweep(scores, labels)