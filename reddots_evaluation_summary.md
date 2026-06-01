# RedDots Dataset Evaluation Summary

> [!NOTE]
> This document summarizes the evaluation of the pretrained **ECAPA-TDNN model** on the **RedDots** dataset. 
> To maintain a clean and structured repository, all outputs and evaluation scripts specific to RedDots are isolated into dedicated files and directories (e.g., `exps_reddots/`, `eval_reddots.py`).

## 1. Optimal Threshold Discovery

The evaluation computed cosine similarity scores for all required trials in `red_dot_trail.txt`. By finding the intersection point where False Acceptance Rate (FAR) matches the False Rejection Rate (FRR), we determined the optimal decision boundary for this specific dataset.

**Optimal EER Threshold found:** `0.4059`

This threshold was then dynamically used to classify the trials as matches or non-matches.

## 2. Overall Performance Metrics

The overarching system performance on RedDots using the pretrained weights:

- **Equal Error Rate (EER):** 2.44%
- **Minimum Detection Cost Function (minDCF):** 0.3619%
- **Threshold for EER:** 0.405856

> [!TIP]
> **Comparison with VoxCeleb:** The EER is higher here compared to the VoxCeleb test set (~0.97%), which is expected as RedDots presents different domain challenges and shorter utterance constraints. The necessary threshold is also stricter (0.4059 vs VoxCeleb's 0.31).

## 3. Threshold vs Error Rate Plot

The relationship between the decision threshold and the resulting error rates (FAR / FRR) is visualized below:

![RedDots EER Plot](C:\Users\Harini\.gemini\antigravity-ide\brain\46a8b861-0902-4b4e-9b88-e9c1c712a301\reddots_eer_plot.png)

## 4. Trial Classification Breakdown (Threshold: 0.4059)

Using the derived optimal threshold of `0.4059`, here is the breakdown of the **1,233,280 total trials** evaluated:

| Metric | Count | Description |
| :--- | :--- | :--- |
| **Total Trials** | 1,233,280 | Total audio pairs tested |
| **True Positives** | 3,163 | Correctly identified matching speakers |
| **True Negatives** | 1,200,064 | Correctly rejected different speakers |
| **False Positives** | 29,974 | Different speakers incorrectly flagged as a match |
| **False Negatives** | 79 | Matching speakers incorrectly rejected |

## 5. Final Performance Rates

Based on the trial classification at the EER threshold:

- **Accuracy:** 97.56%
- **False Acceptance Rate (FAR):** 2.44%
- **False Rejection Rate (FRR):** 2.44%
- **Precision:** 9.55%
- **Recall:** 97.56%
- **F1 Score:** 17.39%

> [!WARNING]
> While Accuracy is high (97.56%), Precision is low (9.55%). This happens because the RedDots dataset is heavily imbalanced with non-target trials (1.2 million non-matches vs 3.2k true matches). At an EER of 2.44%, the absolute number of False Positives (29,974) swamps the True Positives (3,163).

## 6. Project Structure & Code Modifications

To keep the RedDots evaluation organized and avoid interfering with the original VoxCeleb configurations, here is exactly what was changed and created in the codebase:

### What We Changed (Existing Codebase)
- We strictly **avoided modifying** the original core training scripts (`trainECAPAModel.py`) and model definitions (`ECAPAModel.py`) for the RedDots evaluation to preserve the VoxCeleb baseline functionality.
- We adapted the evaluation logic from `ECAPAModel.py` (specifically the `eval_network` method) and implemented it directly into a standalone RedDots script, enabling custom enrollment multi-utterance averaging without altering the original model class.

### What We Created (New Files & Folders)
- **[eval_reddots.py](file:///C:/Users/Harini/Documents/ECAPA-TDNN-1/eval_reddots.py)**: The primary execution script built from scratch. It reads the RedDots `.trn` and `.txt` files, caches all unique utterances, extracts model embeddings, averages the multi-utterance enrollments, computes similarity scores, and outputs the EER.
- **[plot_reddots.py](file:///C:/Users/Harini/Documents/ECAPA-TDNN-1/plot_reddots.py)**: A new script to visualize the `eval_metrics.csv` and generate the EER plot.
- **[analyze_reddots.py](file:///C:/Users/Harini/Documents/ECAPA-TDNN-1/analyze_reddots.py)**: A new script to dynamically extract the EER threshold and parse `trial_predictions.csv` to calculate the final performance rates (Accuracy, Precision, Recall, etc.).
- **`exps_reddots/`**: A new output directory created to securely hold RedDots-specific artifacts:
  - `eval_metrics.csv` (FAR/FRR thresholds curve)
  - `trial_predictions.csv` (Individual trial scores and predictions)
  - `eer_plot.png` (The visualized intersection point)

## 7. End-to-End Execution Commands

To reproduce these results from start to finish, the following commands were run in the WSL environment:

**Step 1: Run the RedDots Evaluation (Extract Embeddings & Compute EER)**
```bash
wsl -e python3 eval_reddots.py
```
*(This creates the `exps_reddots` folder, calculates the threshold, and saves `eval_metrics.csv` and `trial_predictions.csv`)*

**Step 2: Generate the EER Visualization Plot**
```bash
wsl -e python3 plot_reddots.py
```
*(This reads the metrics and saves `exps_reddots/eer_plot.png`)*

**Step 3: Analyze the Final Performance Rates**
```bash
wsl -e python3 analyze_reddots.py
```
*(This parses the predictions using the optimal threshold and outputs Accuracy, Precision, Recall, F1 Score, FAR, and FRR)*
