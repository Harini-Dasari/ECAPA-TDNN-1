# Trial Evaluation Summary (Fixed Threshold: 0.31)

This report summarizes the modifications, newly created files, and final performance metrics for testing the ECAPA-TDNN model trials at a fixed cosine similarity threshold of `0.31`.

## 1. Modifications Made
We modified the `eval_network` function in `ECAPAModel.py` to test every single audio pair at a fixed threshold of `0.31` instead of only tuning the threshold mathematically. This allows us to export explicit "Match" or "Non-Match" predictions.

```python
# Export individual trial predictions at threshold 0.31
import csv
fixed_threshold = 0.31
with open('exps/trial_predictions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Audio_1', 'Audio_2', 'GroundTruth', 'Score', 'Prediction_0.31', 'IsCorrect'])
    for line, score, label in zip(lines, scores, labels):
        parts = line.split()
        pred = 1 if score >= fixed_threshold else 0
        is_correct = (pred == label)
        writer.writerow([parts[1], parts[2], label, score, pred, is_correct])
```

## 2. Files Created
- [exps/trial_predictions.csv](file:///C:/Users/Harini/Documents/ECAPA-TDNN-1/exps/trial_predictions.csv): This CSV file contains the raw predictions, actual ground truth labels, and computed similarity scores for all 37,611 evaluated audio pairs.
- [analyze_trials.py](file:///C:/Users/Harini/Documents/ECAPA-TDNN-1/analyze_trials.py): A custom Python script created to aggregate the `trial_predictions.csv` data and calculate total True Positives, False Positives, Accuracy, FAR, and FRR.

## 3. Command Executed
We ran the following command in WSL to crunch the numbers using the custom analysis script:
```bash
wsl python3 analyze_trials.py
```

## 4. Final Performance Rates

*   **Accuracy:** 99.03%
*   **False Acceptance Rate (FAR):** 0.98%
*   **False Rejection Rate (FRR):** 0.95%
*   **Precision:** 99.02%
*   **Recall:** 99.05%
*   **F1 Score:** 99.03%

**Counts Breakdown:**

| Metric | Count | Description |
| :--- | :--- | :--- |
| **Total Trials** | `37,611` | Total audio pairs tested |
| **True Positives** | `18,623` | Correctly identified matching speakers |
| **True Negatives** | `18,625` | Correctly rejected different speakers |
| **False Positives** | `184` | Different speakers incorrectly flagged as a match |
| **False Negatives** | `179` | Matching speakers incorrectly rejected |
