#!/usr/bin/env python3
"""Plot FAR/FRR for a single RedDots sweep CSV and mark EER.

Usage:
  python3 plot_reddots_single_thresholds.py --csv exps/red-dot/reddot_0.28_to_0.30_phrase.csv --out exps/red-dot/phrase_0.28_to_0.30.png
"""
import argparse
import csv
import math
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def load_sweep(path: str) -> Dict[str, List[float]]:
    out = {"threshold": [], "FAR": [], "FRR": [], "Accuracy": []}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                t = float(r.get("threshold", r.get("Threshold", r.get("thresh", "0"))))
            except Exception:
                continue
            def getf(k):
                for cand in (k, k.lower(), k.upper()):
                    v = r.get(cand)
                    if v is not None:
                        try:
                            return float(v)
                        except Exception:
                            return math.nan
                return math.nan
            out["threshold"].append(t)
            out["FAR"].append(getf("FAR"))
            out["FRR"].append(getf("FRR"))
            out["Accuracy"].append(getf("Accuracy"))
    zipped = list(zip(out["threshold"], out["FAR"], out["FRR"], out["Accuracy"]))
    zipped.sort(key=lambda x: x[0])
    th, far, frr, acc = zip(*zipped)
    return {"threshold": list(th), "FAR": list(far), "FRR": list(frr), "Accuracy": list(acc)}


def estimate_eer(sweep: Dict[str, List[float]]) -> Tuple[float, float]:
    best_idx = 0
    best_diff = float('inf')
    for i, (f, r) in enumerate(zip(sweep["FAR"], sweep["FRR"])):
        d = abs(f - r)
        if d < best_diff:
            best_diff = d
            best_idx = i
    thr = sweep["threshold"][best_idx]
    eer = 0.5 * (sweep["FAR"][best_idx] + sweep["FRR"][best_idx])
    return thr, eer


def make_plot(sweep: Dict[str, List[float]], out_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(sweep["threshold"], sweep["FAR"], label="FAR", color="#1f77b4", linestyle='-')
    plt.plot(sweep["threshold"], sweep["FRR"], label="FRR", color="#ff7f0e", linestyle='--')
    thr, eer = estimate_eer(sweep)
    plt.axvline(thr, color="gray", linestyle=":")
    ymax = max(max(sweep["FAR"]), max(sweep["FRR"]))
    plt.text(thr, ymax*0.9 if ymax>0 else 0.01, f"EER={eer:.4f}\nthr={thr:.3f}", ha='center')
    plt.xlabel("Decision threshold")
    plt.ylabel("Rate")
    plt.title("Phrase-aware RedDots — FAR/FRR (0.280–0.300)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    sweep = load_sweep(args.csv)
    make_plot(sweep, args.out)


if __name__ == '__main__':
    main()
