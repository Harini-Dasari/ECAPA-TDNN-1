#!/usr/bin/env python3
"""Generate per-trial decisions CSV from a trial-scores CSV.
Usage:
  python tools/generate_decisions.py --scores exps/red-dot/m_part_01_trial_scores.csv --threshold 0.37
"""
import csv
import argparse
from pathlib import Path

def is_genuine_label(label: str) -> bool:
    if label is None:
        return False
    v = str(label).strip().lower()
    return v in ('1','true','t','y','yes','target','genuine')


def is_true_like(value: str) -> bool:
    if value is None:
        return False
    v = str(value).strip().lower()
    return v in ('1', 'true', 't', 'y', 'yes')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scores', required=True, help='Path to trial scores CSV')
    p.add_argument('--threshold', type=float, required=True)
    p.add_argument('--out', help='Output CSV path; default derived from threshold')
    p.add_argument('--require_phrase_match', action='store_true', help='Accept only if score passes threshold and phrase_match is true')
    args = p.parse_args()

    scores_path = Path(args.scores)
    if not scores_path.exists():
        raise SystemExit(f"Scores file not found: {scores_path}")

    out_path = Path(args.out) if args.out else scores_path.parent / f"{scores_path.stem}_decisions_{args.threshold:.2f}.csv"

    total = 0
    genuine = 0
    nongenuine = 0
    fp = 0
    fn = 0
    ta = 0
    tr = 0

    with scores_path.open('r', newline='') as fin, out_path.open('w', newline='') as fout:
        reader = csv.DictReader(fin)
        base_fields = list(reader.fieldnames or [])
        if 'phrase_match' not in [f.lower() for f in base_fields]:
            base_fields.append('phrase_match')
        fieldnames = base_fields + ['decision','is_genuine','correct']
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for r in reader:
            total += 1
            score = float(r.get('score', r.get('Score', 0)))
            label = r.get('label', '')
            genuine_flag = is_genuine_label(label)
            if genuine_flag:
                genuine += 1
            else:
                nongenuine += 1

            phrase_match = is_true_like(r.get('phrase_match', r.get('PhraseMatch', '0')))
            decision = score >= args.threshold
            if args.require_phrase_match:
                decision = decision and phrase_match
            correct = decision == genuine_flag
            if decision and not genuine_flag:
                fp += 1
            if not decision and genuine_flag:
                fn += 1
            if decision and genuine_flag:
                ta += 1
            if not decision and not genuine_flag:
                tr += 1

            out_row = dict(r)
            out_row['decision'] = 'accept' if decision else 'reject'
            out_row['is_genuine'] = 'genuine' if genuine_flag else 'non-genuine'
            out_row['correct'] = '1' if correct else '0'
            out_row['phrase_match'] = '1' if phrase_match else '0'
            writer.writerow(out_row)

    # Print a compact summary
    print(f'Wrote decisions to: {out_path}')
    print(f'total={total}, genuine={genuine}, non_genuine={nongenuine}')
    print(f'true_accepts={ta}, false_rejects={fn}, false_accepts={fp}, true_rejects={tr}')
    print(f'accuracy={(ta+tr)/total:.6f}')

if __name__ == '__main__':
    main()
