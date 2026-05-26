#!/usr/bin/env python3
"""Compute category counts and probabilities from a RedDots trial-scores CSV.

Produces a CSV with per-category totals, category probability (of whole set),
accepts at a given threshold, and accept rate per category.

Usage:
  python tools/compute_category_probabilities.py --scores exps/red-dot/m_part_01_trial_scores.csv --threshold 0.37 --out exps/red-dot/m_0.37_probs.csv
"""
import csv
import argparse
from pathlib import Path

def is_genuine_label(label: str) -> bool:
    if label is None:
        return False
    v = str(label).strip().lower()
    return v in ('1','true','t','y','yes','target','genuine')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scores', required=True, help='Path to trial scores CSV')
    p.add_argument('--threshold', type=float, default=None, help='Decision threshold (optional)')
    p.add_argument('--out', help='Output CSV path; default derived from threshold')
    args = p.parse_args()

    scores_path = Path(args.scores)
    if not scores_path.exists():
        raise SystemExit(f"Scores file not found: {scores_path}")

    out_path = Path(args.out) if args.out else scores_path.parent / f"{scores_path.stem}_probs_{args.threshold if args.threshold is not None else 'nothresh'}.csv"

    total = 0
    category_counts = {}
    category_accepts = {}

    with scores_path.open('r', newline='') as fin:
        reader = csv.DictReader(fin)
        for r in reader:
            total += 1
            # Prefer explicit category column
            category = (r.get('category') or '').strip()
            label = (r.get('label') or '').strip()
            # Fallback mapping
            if not category:
                if is_genuine_label(label):
                    category = 'target-correct'
                else:
                    category = 'non-genuine'

            category_counts.setdefault(category, 0)
            category_accepts.setdefault(category, 0)
            category_counts[category] += 1

            if args.threshold is not None:
                score = float(r.get('score', r.get('Score', 0)))
                decision = score >= args.threshold
                if decision:
                    category_accepts[category] += 1

    # Write output CSV
    with out_path.open('w', newline='') as fout:
        fieldnames = ['category','count','probability_of_category','accepts_at_threshold','accept_rate']
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
            accepts = category_accepts.get(cat, 0)
            prob = cnt / total if total>0 else 0.0
            accept_rate = accepts / cnt if cnt>0 else 0.0
            writer.writerow({'category':cat,'count':cnt,'probability_of_category':f'{prob:.8f}','accepts_at_threshold':accepts,'accept_rate':f'{accept_rate:.6f}'})

    print(f'Wrote probabilities to: {out_path}')
    print(f'total_trials={total}')

if __name__ == '__main__':
    main()
