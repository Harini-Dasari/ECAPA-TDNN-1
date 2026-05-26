# ECAPA-TDNN File Summary

This file lists the main files in the repository, what each one does, and the threshold values used for evaluation.

| File | Purpose | Threshold value(s) |
|---|---|---|
| `dataLoader.py` | Builds the training and evaluation dataset loaders. | No fixed threshold defined. |
| `model.py` | Defines the ECAPA-TDNN speaker encoder architecture. | No fixed threshold defined. |
| `loss.py` | Defines the AAM-Softmax loss used during training. | No fixed threshold defined. |
| `tools.py` | Utility functions for EER, FAR/FRR, minDCF, and threshold tuning. | Uses computed thresholds from scores, not one fixed value. |
| `ECAPAModel.py` | Wraps the encoder for training/evaluation, computes scores, EER, minDCF, and threshold plots. | `tuneThresholdfromScore(scores, labels, [1, 0.1])`, `ComputeMinDcf(..., 0.05, 1, 1)`, and fine sweep `-1.0` to `1.0` with step `0.001`. |
| `trainECAPAModel.py` | Main training/evaluation script. Runs fixed-threshold evaluation and saves CSV/PNG reports. | Fixed thresholds: `0.30, 0.31, 0.32, 0.33, 0.34`. Also saves reports using `0.31`-based final eval flow. |
| `threshold_sweep.py` | Standalone, structured threshold sweep script for clean CSV/JSON reports. | Defaults: start `0.1`, end `1.0`, step `0.1`. Can also be run with custom ranges like `0.30` to `0.34` step `0.01`. |
| `final_evaluate.py` | Final evaluation runner that computes fixed-threshold metrics and stores outputs in `exps/final_eval`. | Default threshold: `0.31`. |
| `README.md` | General project overview and usage notes. | No fixed threshold defined. |
| `codebase.md` | Detailed documentation of the repository and evaluation workflow. | Describes `0.31`, `0.30-0.34`, and sweep ranges. |
| `System1.md` | Additional project notes and documentation. | No fixed threshold defined. |
| `ECAPAModel_explained.md` | Line-by-line explanation of `ECAPAModel.py`. | Documents the thresholds used in `ECAPAModel.py`. |
| `trainECAPAModel_explained.md` | Line-by-line explanation of `trainECAPAModel.py`. | Documents the fixed thresholds used in `trainECAPAModel.py`. |
| `requirements.txt` | Dependency list for the project. | No threshold defined. |

## Quick takeaway

- If you want the main final evaluation threshold, use `0.31` in `final_evaluate.py`.
- If you want the fixed-threshold report block, `trainECAPAModel.py` uses `0.30` to `0.34`.
- If you want clean sweep experiments, `threshold_sweep.py` is the better script.
