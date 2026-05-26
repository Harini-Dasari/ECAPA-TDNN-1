from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_threshold_sweep(csv_path: Path) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {
        "threshold": [],
        "far": [],
        "frr": [],
    }

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            data["threshold"].append(float(row["threshold"]))
            data["far"].append(float(row["far"]))
            data["frr"].append(float(row["frr"]))

    return data


def make_plot(csv_path: Path, output_path: Path) -> None:
    data = load_threshold_sweep(csv_path)

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader)

    eer_threshold = float(first_row["eer_threshold"])
    min_dcf_threshold = float(first_row["min_dcf_threshold"])
    best_accuracy_threshold = float(first_row["best_accuracy_threshold"])
    eer_value = float(first_row["eer"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data["threshold"], data["far"], label="FAR", color="#1f77b4", linewidth=2)
    ax.plot(data["threshold"], data["frr"], label="FRR", color="#d62728", linewidth=2)

    ax.axvline(eer_threshold, color="#2ca02c", linestyle="--", linewidth=1.8, label=f"EER threshold = {eer_threshold:.3f}")
    ax.axvline(min_dcf_threshold, color="#9467bd", linestyle="--", linewidth=1.8, label=f"minDCF threshold = {min_dcf_threshold:.3f}")
    ax.axvline(best_accuracy_threshold, color="#ff7f0e", linestyle=":", linewidth=2, label=f"best accuracy threshold = {best_accuracy_threshold:.3f}")

    ax.scatter([eer_threshold], [eer_value], color="#2ca02c", zorder=5)
    ax.set_title("RedDots m_part_01: FAR and FRR vs Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(svg_path)
    plt.close(fig)

    print(f"Saved plot: {output_path}")
    print(f"Saved plot: {svg_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot FAR and FRR versus threshold for a RedDots sweep CSV.")
    parser.add_argument("--csv", type=Path, default=Path("exps/red-dot/m_part_01_threshold_sweep.csv"), help="Path to the threshold sweep CSV.")
    parser.add_argument("--output", type=Path, default=Path("exps/red-dot/m_part_01_far_frr_vs_threshold.png"), help="Output PNG path.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    make_plot(args.csv, args.output)


if __name__ == "__main__":
    main()