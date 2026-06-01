import csv
import matplotlib.pyplot as plt
import numpy as np

# Load the data
thresholds = []
far = []
frr = []

with open('exps/eval_metrics.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        thresholds.append(float(row['Threshold']))
        far.append(float(row['FAR']))
        frr.append(float(row['FRR']))

thresholds = np.array(thresholds)
far = np.array(far)
frr = np.array(frr)

# Find the EER point where FAR and FRR intersect
idxE = np.nanargmin(np.abs(far - frr))
eer_threshold = thresholds[idxE]
eer_val = max(far[idxE], frr[idxE]) * 100

plt.figure(figsize=(10, 6))
plt.plot(thresholds, far * 100, label='False Acceptance Rate (FAR)', color='blue', linewidth=2)
plt.plot(thresholds, frr * 100, label='False Rejection Rate (FRR)', color='red', linewidth=2)

# Mark the EER point
plt.axvline(x=eer_threshold, color='green', linestyle='--', label=f'EER Threshold: {eer_threshold:.4f}')
plt.plot(eer_threshold, eer_val, 'go', markersize=8, label=f'EER: {eer_val:.2f}%')

plt.title('False Acceptance Rate (FAR) and False Rejection Rate (FRR) vs. Score Threshold', fontsize=14)
plt.xlabel('Cosine Similarity Score Threshold', fontsize=12)
plt.ylabel('Error Rate (%)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig('exps/eer_plot.png', dpi=300)
print("Plot saved to exps/eer_plot.png")
