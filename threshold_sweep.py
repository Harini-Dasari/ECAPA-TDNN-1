from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from sklearn import metrics
from tqdm import tqdm

from ECAPAModel import ECAPAModel
from tools import ComputeErrorRates, ComputeMinDcf

MAX_AUDIO = 300 * 160 + 240


def resolve_audio_path(eval_path: str | os.PathLike[str], relative_path: str) -> Path:
    return Path(eval_path) / relative_path


def read_eval_list(eval_list: str | os.PathLike[str]) -> tuple[list[tuple[int, str, str]], list[str]]:
    pairs: list[tuple[int, str, str]] = []
    files: set[str] = set()
    with open(eval_list, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            label_str, file_one, file_two = line.split()
            pairs.append((int(label_str), file_one, file_two))
            files.add(file_one)
            files.add(file_two)
    return pairs, sorted(files)


def read_audio(audio_path: Path) -> np.ndarray:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        audio, _ = sf.read(audio_path)
    except Exception:
        audio_tensor, _ = torchaudio.load(str(audio_path))
        audio = audio_tensor.numpy().T
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return np.asarray(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float32)


def build_split_segments(audio: np.ndarray) -> np.ndarray:
    if audio.shape[0] <= MAX_AUDIO:
        shortage = MAX_AUDIO - audio.shape[0]
        audio = np.pad(audio, (0, shortage), mode="wrap")
    startframe = np.linspace(0, audio.shape[0] - MAX_AUDIO, num=5)
    segments = [audio[int(start): int(start) + MAX_AUDIO] for start in startframe]
    return np.stack(segments, axis=0).astype(np.float32)


def model_signature(model_path: str | os.PathLike[str]) -> str:
    path = Path(model_path)
    stat_result = path.stat()
    payload = f"{path.resolve()}|{stat_result.st_size}|{int(stat_result.st_mtime)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def cache_key(eval_list: str | os.PathLike[str], eval_path: str | os.PathLike[str], model_path: str | os.PathLike[str]) -> str:
    payload = f"{Path(eval_list).resolve()}|{Path(eval_path).resolve()}|{model_signature(model_path)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def extract_embeddings(
    encoder: torch.nn.Module,
    files: list[str],
    eval_path: str | os.PathLike[str],
    split_batch_size: int,
    device: torch.device,
    cache_file: Path | None,
    use_cache: bool,
) -> dict[str, dict[str, torch.Tensor]]:
    if use_cache and cache_file is not None and cache_file.exists():
        return torch.load(cache_file, map_location="cpu")

    encoder.eval()
    embeddings: dict[str, dict[str, torch.Tensor]] = {}
    pending_split_tensors: list[torch.Tensor] = []
    pending_split_files: list[str] = []

    def flush_split_batches() -> None:
        nonlocal pending_split_tensors, pending_split_files
        if not pending_split_tensors:
            return
        batch = torch.cat(pending_split_tensors, dim=0).to(device, non_blocking=device.type == "cuda")
        with torch.inference_mode():
            split_embeddings = encoder(batch, aug=False)
            split_embeddings = F.normalize(split_embeddings, p=2, dim=1)
        split_embeddings = split_embeddings.detach().cpu().view(len(pending_split_files), 5, -1)
        for file_name, file_embedding in zip(pending_split_files, split_embeddings):
            embeddings[file_name]["split"] = file_embedding
        pending_split_tensors = []
        pending_split_files = []

    for file_name in tqdm(files, desc="Extract embeddings"):
        audio = read_audio(resolve_audio_path(eval_path, file_name))
        embeddings[file_name] = {}

        full_tensor = torch.from_numpy(audio).unsqueeze(0)
        with torch.inference_mode():
            full_embedding = encoder(full_tensor.to(device, non_blocking=device.type == "cuda"), aug=False)
            full_embedding = F.normalize(full_embedding, p=2, dim=1)
        embeddings[file_name]["full"] = full_embedding.detach().cpu()

        pending_split_tensors.append(torch.from_numpy(build_split_segments(audio)))
        pending_split_files.append(file_name)
        if len(pending_split_files) >= max(1, split_batch_size):
            flush_split_batches()

    flush_split_batches()
    if use_cache and cache_file is not None:
        torch.save(embeddings, cache_file)
    return embeddings


def score_trials(
    embeddings: dict[str, dict[str, torch.Tensor]],
    pairs: list[tuple[int, str, str]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    if device.type == "cuda":
        scoring_embeddings = {
            file_name: {key: tensor.to(device) for key, tensor in file_parts.items()}
            for file_name, file_parts in embeddings.items()
        }
    else:
        scoring_embeddings = embeddings

    scores: list[float] = []
    labels: list[int] = []
    rows: list[dict[str, object]] = []

    for label, file_one, file_two in tqdm(pairs, desc="Score pairs"):
        embedding_11 = scoring_embeddings[file_one]["full"]
        embedding_12 = scoring_embeddings[file_one]["split"]
        embedding_21 = scoring_embeddings[file_two]["full"]
        embedding_22 = scoring_embeddings[file_two]["split"]
        with torch.inference_mode():
            score_one = torch.mean(torch.matmul(embedding_11, embedding_21.T))
            score_two = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = ((score_one + score_two) / 2).item()
        scores.append(score)
        labels.append(label)
        rows.append({"label": label, "score": score, "file_one": file_one, "file_two": file_two})

    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64), rows


def sweep_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold_start: float,
    threshold_end: float,
    threshold_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    thresholds = np.arange(threshold_start, threshold_end + threshold_step * 0.5, threshold_step, dtype=np.float64)
    positive_mask = labels == 1
    negative_mask = ~positive_mask
    genuine_count = int(np.sum(positive_mask))
    impostor_count = int(np.sum(negative_mask))
    if genuine_count == 0 or impostor_count == 0:
        raise ValueError("Evaluation list must contain both genuine and impostor trials")

    decisions = scores[:, None] >= thresholds[None, :]
    true_positive = np.sum(decisions & positive_mask[:, None], axis=0)
    false_positive = np.sum(decisions & negative_mask[:, None], axis=0)
    false_negative = genuine_count - true_positive
    true_negative = impostor_count - false_positive

    far = false_positive / float(impostor_count)
    frr = false_negative / float(genuine_count)
    accuracy = (true_positive + true_negative) / float(labels.shape[0])
    return thresholds, far, frr, accuracy


def save_threshold_csv(
    output_csv: Path,
    thresholds: np.ndarray,
    far: np.ndarray,
    frr: np.ndarray,
    accuracy: np.ndarray,
    summary: dict[str, float],
) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["threshold", "far", "frr", "accuracy", "eer", "eer_threshold", "min_dcf", "min_dcf_threshold", "best_accuracy", "best_accuracy_threshold"],
        )
        writer.writeheader()
        for threshold, far_value, frr_value, acc_value in zip(thresholds, far, frr, accuracy):
            writer.writerow(
                {
                    "threshold": float(threshold),
                    "far": float(far_value),
                    "frr": float(frr_value),
                    "accuracy": float(acc_value),
                    "eer": float(summary["eer"]),
                    "eer_threshold": float(summary["eer_threshold"]),
                    "min_dcf": float(summary["min_dcf"]),
                    "min_dcf_threshold": float(summary["min_dcf_threshold"]),
                    "best_accuracy": float(summary["best_accuracy"]),
                    "best_accuracy_threshold": float(summary["best_accuracy_threshold"]),
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ECAPA-TDNN pretrained threshold sweep")
    parser.add_argument("--initial_model", type=str, required=True, help="Path to pretrained checkpoint in exps/")
    parser.add_argument("--eval_list", type=str, required=True, help="VoxCeleb1 trial list")
    parser.add_argument("--eval_path", type=str, required=True, help="Root directory of VoxCeleb1 test WAVs")
    parser.add_argument("--output_csv", type=str, default="exps/threshold_results.csv", help="CSV file to store threshold sweep results")
    parser.add_argument("--summary_json", type=str, default="exps/threshold_summary.json", help="JSON file to store the final summary")
    parser.add_argument("--threshold_start", type=float, default=0.1, help="Threshold sweep start")
    parser.add_argument("--threshold_end", type=float, default=1.0, help="Threshold sweep end")
    parser.add_argument("--threshold_step", type=float, default=0.1, help="Threshold sweep step")
    parser.add_argument("--split_batch_size", type=int, default=32, help="Number of utterances batched for split-window extraction")
    parser.add_argument("--cache_embeddings", action="store_true", default=True, help="Cache embeddings to disk")
    parser.add_argument("--no_cache_embeddings", action="store_false", dest="cache_embeddings", help="Disable embedding cache")
    parser.add_argument("--lr", type=float, default=0.001, help="ECAPAModel constructor placeholder")
    parser.add_argument("--lr_decay", type=float, default=0.97, help="ECAPAModel constructor placeholder")
    parser.add_argument("--C", type=int, default=1024, help="Channel size for speaker encoder")
    parser.add_argument("--m", type=float, default=0.2, help="AAM softmax margin")
    parser.add_argument("--s", type=float, default=30, help="AAM softmax scale")
    parser.add_argument("--n_class", type=int, default=5994, help="Number of speakers")
    parser.add_argument("--test_step", type=int, default=1, help="ECAPAModel constructor placeholder")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECAPAModel(lr=args.lr, lr_decay=args.lr_decay, C=args.C, n_class=args.n_class, m=args.m, s=args.s, test_step=args.test_step)
    print(f"Model loaded from {args.initial_model}")
    model.load_parameters(args.initial_model)
    model.eval()

    pairs, files = read_eval_list(args.eval_list)
    cache_file = output_csv.parent / f"embeddings_{cache_key(args.eval_list, args.eval_path, args.initial_model)}.pt"
    embeddings = extract_embeddings(model.speaker_encoder, files, args.eval_path, args.split_batch_size, device, cache_file, args.cache_embeddings)
    scores, labels, rows = score_trials(embeddings, pairs, device)

    thresholds, far, frr, accuracy = sweep_thresholds(scores, labels, args.threshold_start, args.threshold_end, args.threshold_step)
    best_threshold_index = int(np.argmax(accuracy))
    eer_index = int(np.nanargmin(np.abs(far - frr)))
    fnrs, fprs, dcf_thresholds = ComputeErrorRates(scores.tolist(), labels.tolist())
    min_dcf, min_dcf_threshold = ComputeMinDcf(fnrs, fprs, dcf_thresholds, 0.05, 1, 1)
    eer = float(max(far[eer_index], frr[eer_index]))
    eer_threshold = float(thresholds[eer_index])
    summary = {
        "total_pairs": len(pairs),
        "total_utterances": len(files),
        "cache_hit": bool(args.cache_embeddings and cache_file.exists()),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "far_at_eer": float(far[eer_index]),
        "frr_at_eer": float(frr[eer_index]),
        "best_accuracy": float(accuracy[best_threshold_index]),
        "best_accuracy_threshold": float(thresholds[best_threshold_index]),
        "best_accuracy_far": float(far[best_threshold_index]),
        "best_accuracy_frr": float(frr[best_threshold_index]),
        "min_dcf": float(min_dcf),
        "min_dcf_threshold": float(min_dcf_threshold),
        "threshold_start": args.threshold_start,
        "threshold_end": args.threshold_end,
        "threshold_step": args.threshold_step,
    }

    save_threshold_csv(output_csv, thresholds, far, frr, accuracy, summary)
    summary["threshold_csv"] = str(output_csv)
    summary["embedding_cache"] = str(cache_file)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
