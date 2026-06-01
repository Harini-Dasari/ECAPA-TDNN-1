import csv
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    metrics_path = 'exps_reddots/eval_metrics.csv'
    output_img = 'exps_reddots/eer_plot.png'
    
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found. Run evaluation first.")
        return

    thresholds = []
    far = []
    frr = []

    print(f"Loading metrics from {metrics_path}...")
    with open(metrics_path, 'r') as f:
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

    print("Generating plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, far * 100, label='False Acceptance Rate (FAR)', color='blue', linewidth=2)
    plt.plot(thresholds, frr * 100, label='False Rejection Rate (FRR)', color='red', linewidth=2)

    # Mark the EER point
    plt.axvline(x=eer_threshold, color='green', linestyle='--', label=f'EER Threshold: {eer_threshold:.4f}')
    plt.plot(eer_threshold, eer_val, 'go', markersize=8, label=f'EER: {eer_val:.2f}%')

    plt.title('RedDots Dataset: FAR and FRR vs. Score Threshold', fontsize=14)
    plt.xlabel('Cosine Similarity Score Threshold', fontsize=12)
    plt.ylabel('Error Rate (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_img, dpi=300)
    print(f"Plot successfully saved to {output_img}")

if __name__ == '__main__':
    main()
