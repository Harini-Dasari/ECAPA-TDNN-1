#!/usr/bin/env python3
"""Plot FAR/FRR curves for speaker-only and phrase-aware RedDots sweeps.

Usage:
  python3 plot_reddots_compare_thresholds.py --speaker_csv exps/red-dot/m_part_01_threshold_sweep.csv --phrase_csv exps/red-dot/m_part_01_phrase_gate_threshold_sweep.csv --out exps/red-dot/compare_far_frr.png
"""
import argparse
import csv
import math
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def load_sweep(path: str) -> Dict[str, List[float]]:
    # expects CSV with columns: threshold,FAR,FRR,Accuracy (header may vary but these are used)
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
            far = getf("FAR")
            frr = getf("FRR")
            acc = getf("Accuracy")
            out["threshold"].append(t)
            out["FAR"].append(far)
            out["FRR"].append(frr)
            out["Accuracy"].append(acc)
    # sort by threshold
    zipped = list(zip(out["threshold"], out["FAR"], out["FRR"], out["Accuracy"]))
    zipped.sort(key=lambda x: x[0])
    th, far, frr, acc = zip(*zipped)
    return {"threshold": list(th), "FAR": list(far), "FRR": list(frr), "Accuracy": list(acc)}


def estimate_eer(sweep: Dict[str, List[float]]) -> Tuple[float, float]:
    # EER approx: threshold where |FAR-FRR| minimal; EER ~ (FAR+FRR)/2 at that point
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


def make_plot(speaker_sweep: Dict[str, List[float]], phrase_sweep: Dict[str, List[float]], out_path: str):
    plt.figure(figsize=(10, 6))

    # FAR/FRR speaker-only
    plt.plot(speaker_sweep["threshold"], speaker_sweep["FAR"], label="FAR (speaker-only)", color="#1f77b4", linestyle="-")
    plt.plot(speaker_sweep["threshold"], speaker_sweep["FRR"], label="FRR (speaker-only)", color="#1f77b4", linestyle="--")

    # FAR/FRR phrase-aware
    plt.plot(phrase_sweep["threshold"], phrase_sweep["FAR"], label="FAR (phrase-aware)", color="#ff7f0e", linestyle="-")
    plt.plot(phrase_sweep["threshold"], phrase_sweep["FRR"], label="FRR (phrase-aware)", color="#ff7f0e", linestyle="--")

    # EER markers
    thr_s, eer_s = estimate_eer(speaker_sweep)
    thr_p, eer_p = estimate_eer(phrase_sweep)
    plt.axvline(thr_s, color="#1f77b4", linestyle=":", linewidth=1)
    plt.axvline(thr_p, color="#ff7f0e", linestyle=":", linewidth=1)
    plt.text(thr_s, max(max(speaker_sweep["FAR"] + speaker_sweep["FRR"]), 0.01), f"EER_s={eer_s:.4f}\nthr={thr_s:.3f}", color="#1f77b4", va="bottom", ha="right")
    plt.text(thr_p, max(max(phrase_sweep["FAR"] + phrase_sweep["FRR"]), 0.01), f"EER_p={eer_p:.4f}\nthr={thr_p:.3f}", color="#ff7f0e", va="bottom", ha="left")

    plt.xlabel("Decision threshold")
    plt.ylabel("Rate")
    plt.title("RedDots m_part_01 — FAR/FRR: Speaker-only vs Phrase-aware")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--speaker_csv", required=True)
    p.add_argument("--phrase_csv", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    speaker = load_sweep(args.speaker_csv)
    phrase = load_sweep(args.phrase_csv)

    make_plot(speaker, phrase, args.out)


if __name__ == "__main__":
    main()
