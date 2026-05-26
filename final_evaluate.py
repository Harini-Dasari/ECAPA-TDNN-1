import os
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ECAPAModel import ECAPAModel
from tools import ComputeErrorRates, tuneThresholdfromScore


def compute_fixed_metrics(scores, labels, threshold):
    scores = np.array(scores)
    labels = np.array(labels)
    # genuine -> label 1, impostor -> label 0
    total_genuine = np.sum(labels == 1)
    total_impostor = np.sum(labels == 0)
    false_accepts = np.sum((scores >= threshold) & (labels == 0))
    false_rejects = np.sum((scores < threshold) & (labels == 1))
    far = false_accepts / total_impostor if total_impostor > 0 else 0.0
    frr = false_rejects / total_genuine if total_genuine > 0 else 0.0
    eer_approx = (far + frr) / 2.0
    # exact EER from tuneThresholdfromScore
    try:
        tuned = tuneThresholdfromScore(scores.tolist(), labels.tolist(), [1, 0.1])
        exact_eer = float(tuned[1])
    except Exception:
        exact_eer = np.nan
    return {
        'threshold': float(threshold),
        'FAR': float(far),
        'FRR': float(frr),
        'EER_approx': float(eer_approx),
        'EER': float(exact_eer)
    }


def save_score_distribution(scores, labels, outpath):
    genuine = [scores[i] for i in range(len(scores)) if labels[i] == 1]
    impostor = [scores[i] for i in range(len(scores)) if labels[i] == 0]
    plt.figure()
    plt.hist(genuine, bins=50, alpha=0.6, label='Genuine', color='blue')
    plt.hist(impostor, bins=50, alpha=0.6, label='Impostor', color='red')
    plt.title('Score Distributions')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(outpath)
    plt.close()


def save_far_frr_curve(scores, labels, outpath, thr_min=-1.0, thr_max=1.0, step=0.001):
    thresholds = np.arange(thr_min, thr_max, step)
    far = []
    frr = []
    s = np.array(scores)
    l = np.array(labels)
    total_genuine = np.sum(l == 1)
    total_impostor = np.sum(l == 0)
    for t in tqdm(thresholds, desc="Thresholds"):
        fa = np.sum((s >= t) & (l == 0))
        fr = np.sum((s < t) & (l == 1))
        far.append(float(fa / total_impostor) if total_impostor>0 else 0.0)
        frr.append(float(fr / total_genuine) if total_genuine>0 else 0.0)
    plt.figure()
    plt.plot(thresholds, far, label='FAR', color='red')
    plt.plot(thresholds, frr, label='FRR', color='blue')
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.title('FAR / FRR vs Threshold')
    plt.legend()
    plt.savefig(outpath)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_model', required=True, help='Path to pretrained model')
    parser.add_argument('--eval_lists', nargs='+', required=True, help='One or more trial list files')
    parser.add_argument('--eval_path', default='Datasets', help='Root path for audio files')
    parser.add_argument('--save_path', default='exps/final_eval', help='Directory to save final evaluation outputs')
    parser.add_argument('--threshold', type=float, default=0.31, help='Fixed threshold to evaluate')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Load model
    model = ECAPAModel(lr=0.001, lr_decay=0.97, C=1024, n_class=5994, m=0.2, s=30, test_step=1)
    model.load_parameters(args.initial_model)

    summary_rows = []

    for eval_list in tqdm(args.eval_lists, desc="Eval lists"):
        base = os.path.splitext(os.path.basename(eval_list))[0]
        print(f"Evaluating {eval_list} -> {base}")
        sys.stdout.flush()
        EER, minDCF = model.eval_network(eval_list=eval_list, eval_path=args.eval_path)
        # use cached scores
        scores = model.cached_scores
        labels = model.cached_labels
        metrics = compute_fixed_metrics(scores, labels, args.threshold)

        # Save CSV with single-row summary
        csv_out = os.path.join(args.save_path, f"{base}_threshold_{int(args.threshold*1000):03d}_summary.csv")
        try:
            import csv
            with open(csv_out, 'w', newline='') as cf:
                writer = csv.writer(cf)
                writer.writerow(['trial_list', 'threshold', 'EER_exact', 'EER_approx', 'FAR', 'FRR', 'minDCF'])
                writer.writerow([base, metrics['threshold'], metrics['EER'], metrics['EER_approx'], metrics['FAR'], metrics['FRR'], float(minDCF)])
            print(f"Saved CSV: {csv_out}")
        except Exception as e:
            print(f"Error saving CSV {csv_out}: {e}")

        # Save score distribution plot
        score_png = os.path.join(args.save_path, f"{base}_score_distributions.png")
        try:
            save_score_distribution(scores, labels, score_png)
            print(f"Saved score distribution: {score_png}")
        except Exception as e:
            print(f"Error saving score distribution {score_png}: {e}")

        # Save FAR/FRR curve (fine-grained)
        farfrr_png = os.path.join(args.save_path, f"{base}_far_frr_curve.png")
        try:
            save_far_frr_curve(scores, labels, farfrr_png)
            print(f"Saved FAR/FRR curve: {farfrr_png}")
        except Exception as e:
            print(f"Error saving FAR/FRR curve {farfrr_png}: {e}")

        summary_rows.append({
            'trial_list': base,
            'threshold': metrics['threshold'],
            'EER_exact': metrics['EER'],
            'EER_approx': metrics['EER_approx'],
            'FAR': metrics['FAR'],
            'FRR': metrics['FRR'],
            'minDCF': float(minDCF)
        })

    # Save global summary CSV
    global_csv = os.path.join(args.save_path, 'final_eval_summary.csv')
    try:
        import csv
        with open(global_csv, 'w', newline='') as gf:
            writer = csv.writer(gf)
            writer.writerow(['trial_list', 'threshold', 'EER_exact', 'EER_approx', 'FAR', 'FRR', 'minDCF'])
            for r in summary_rows:
                writer.writerow([r['trial_list'], r['threshold'], r['EER_exact'], r['EER_approx'], r['FAR'], r['FRR'], r['minDCF']])
        print(f"Saved global summary: {global_csv}")
    except Exception as e:
        print(f"Error saving global summary: {e}")


if __name__ == '__main__':
    main()
