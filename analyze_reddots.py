import csv
import sys
import os

def main():
    metrics_path = 'exps_reddots/trial_predictions.csv'
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found.")
        return

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = 0

    pred_col = None

    with open(metrics_path, 'r') as f:
        reader = csv.DictReader(f)
        for col in reader.fieldnames:
            if col.startswith('Prediction_'):
                pred_col = col
                break
        
        if not pred_col:
            print("Error: Could not find Prediction column.")
            return

        for row in reader:
            gt = int(row['GroundTruth'])
            pred = int(row[pred_col])
            
            if gt == 1 and pred == 1:
                tp += 1
            elif gt == 0 and pred == 0:
                tn += 1
            elif gt == 0 and pred == 1:
                fp += 1
            elif gt == 1 and pred == 0:
                fn += 1
            
            total += 1

    accuracy = (tp + tn) / total * 100
    far = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    threshold_val = pred_col.replace('Prediction_', '')

    print(f"=== Trial Evaluation Analysis (Threshold: {threshold_val}) ===")
    print(f"Total Trials: {total}")
    print(f"True Positives (Correct Matches): {tp}")
    print(f"True Negatives (Correct Rejections): {tn}")
    print(f"False Positives (False Matches / FAR): {fp}")
    print(f"False Negatives (Missed Matches / FRR): {fn}")
    print("---")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"False Acceptance Rate (FAR): {far:.2f}%")
    print(f"False Rejection Rate (FRR): {frr:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")

if __name__ == '__main__':
    main()
