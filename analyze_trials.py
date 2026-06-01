import csv

tp = 0
tn = 0
fp = 0
fn = 0
total = 0

with open('exps/trial_predictions.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        gt = int(row['GroundTruth'])
        pred = int(row['Prediction_0.31'])
        
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

print("=== Trial Evaluation Analysis (Threshold: 0.31) ===")
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
