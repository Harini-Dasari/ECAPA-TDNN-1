from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ECAPAModel import ECAPAModel
from tools import ComputeErrorRates, ComputeMinDcf

MAX_AUDIO = 300 * 160 + 240

def read_pcm_audio(audio_path: Path) -> np.ndarray:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    audio = np.fromfile(audio_path, dtype="<i2")
    if audio.size == 0:
        raise ValueError(f"Empty PCM file: {audio_path}")
    audio = audio.astype(np.float32) / 32768.0
    return audio

#  but modified to be more modular and reusable for the threshold sweep process.
def build_split_segments(audio: np.ndarray) -> np.ndarray:
    if audio.shape[0] <= MAX_AUDIO:
        shortage = MAX_AUDIO - audio.shape[0]
        audio = np.pad(audio, (0, shortage), mode="wrap")
    startframe = np.linspace(0, audio.shape[0] - MAX_AUDIO, num=5)
    segments = [audio[int(start): int(start) + MAX_AUDIO] for start in startframe]
    return np.stack(segments, axis=0).astype(np.float32)

# The following functions are specific to the RedDots protocol and metadata handling,
#  and are designed to be reusable for any evaluation script that needs to process RedDots data.
def resolve_audio_path(pcm_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.suffix != ".pcm":
        relative = relative.with_suffix(".pcm")
    return pcm_root / relative

# The rest of the code is focused on the main evaluation logic, including loading the model,
#  processing the RedDots protocols, extracting embeddings, scoring trials, sweeping thresholds,
def model_signature(model_path: str | os.PathLike[str]) -> str:
    path = Path(model_path)
    stat_result = path.stat()
    payload = f"{path.resolve()}|{stat_result.st_size}|{int(stat_result.st_mtime)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

def read_enrollment_protocol(trn_path: Path) -> dict[str, list[str]]:
    enrollments: dict[str, list[str]] = {}
    with trn_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            enroll_id, utterance_list = line.split(maxsplit=1)
            enrollments[enroll_id] = utterance_list.split(",")
    return enrollments


def read_trial_protocol(ndx_path: Path) -> list[dict[str, str]]:
    mapping = {
        ("Y", "N", "N", "N"): "target-correct",
        ("N", "Y", "N", "N"): "target-wrong",
        ("N", "N", "Y", "N"): "imposter-correct",
        ("N", "N", "N", "Y"): "imposter-wrong",
    }
    trials: list[dict[str, str]] = []
    with ndx_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            enroll_id, test_utterance, flag_1, flag_2, flag_3, flag_4 = line.split(",")
            category = mapping.get((flag_1, flag_2, flag_3, flag_4))
            if category is None:
                raise ValueError(f"Unrecognized RedDots trial flags in line: {line}")
            trials.append(
                {
                    "enroll_id": enroll_id,
                    "test_utterance": test_utterance,
                    "category": category,
                    "label": "1" if category == "target-correct" else "0",
                }
            )
    return trials


def read_script_metadata(script_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    if not script_path.exists():
        raise FileNotFoundError(f"Script metadata file not found: {script_path}")
    metadata_by_utt: dict[str, str] = {}
    metadata_by_enroll: dict[str, str] = {}
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with script_path.open("r", encoding=encoding) as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or ";" not in line:
                        continue
                    utterance_id, phrase_text = line.split(";", maxsplit=1)
                    clean_text = phrase_text.strip()
                    metadata_by_utt[utterance_id] = clean_text
                    parts = utterance_id.split("_")
                    if len(parts) >= 2:
                        enroll_like = "_".join(parts[-2:])
                        metadata_by_enroll.setdefault(enroll_like, clean_text)
            break
        except UnicodeDecodeError:
            metadata_by_utt.clear()
            metadata_by_enroll.clear()
            continue
    return metadata_by_utt, metadata_by_enroll


def extract_phrase_id_from_enroll(enroll_id: str) -> str:
    parts = enroll_id.split("_")
    return parts[-1] if parts else ""


def extract_utterance_key(test_utterance: str) -> str:
    # test_utterance may be like m0001/20150130084216628_m0001_37
    return Path(test_utterance).name


def extract_phrase_id_from_utterance(test_utterance: str) -> str:
    key = extract_utterance_key(test_utterance)
    parts = key.split("_")
    return parts[-1] if parts else ""


def collect_unique_files(enrollments: dict[str, list[str]], trials: list[dict[str, str]]) -> list[str]:
    files: set[str] = set()
    for utterance_list in enrollments.values():
        files.update(utterance_list)
    for trial in trials:
        files.add(trial["test_utterance"])
    return sorted(files)


def extract_embeddings(
    encoder: torch.nn.Module,
    files: list[str],
    pcm_root: Path,
    device: torch.device,
    split_batch_size: int,
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
        audio_path = resolve_audio_path(pcm_root, file_name)
        audio = read_pcm_audio(audio_path)
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


def build_enrollment_templates(
    embeddings: dict[str, dict[str, torch.Tensor]],
    enrollments: dict[str, list[str]],
) -> dict[str, dict[str, torch.Tensor]]:
    templates: dict[str, dict[str, torch.Tensor]] = {}
    for enroll_id, utterances in enrollments.items():
        full_stack = torch.stack([embeddings[utterance]["full"].squeeze(0) for utterance in utterances], dim=0)
        split_stack = torch.stack([embeddings[utterance]["split"] for utterance in utterances], dim=0)
        templates[enroll_id] = {
            "full": full_stack.mean(dim=0, keepdim=True),
            "split": split_stack.mean(dim=0),
        }
    return templates


def score_trials(
    embeddings: dict[str, dict[str, torch.Tensor]],
    templates: dict[str, dict[str, torch.Tensor]],
    trials: list[dict[str, str]],
    device: torch.device,
    script_metadata_by_utt: dict[str, str] | None = None,
    script_metadata_by_enroll: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    if device.type == "cuda":
        scoring_embeddings = {
            file_name: {key: tensor.to(device) for key, tensor in file_parts.items()}
            for file_name, file_parts in embeddings.items()
        }
        scoring_templates = {
            enroll_id: {key: tensor.to(device) for key, tensor in template_parts.items()}
            for enroll_id, template_parts in templates.items()
        }
    else:
        scoring_embeddings = embeddings
        scoring_templates = templates

    scores: list[float] = []
    labels: list[int] = []
    phrase_matches: list[int] = []
    rows: list[dict[str, object]] = []

    for trial in tqdm(trials, desc="Score trials"):
        enroll_id = trial["enroll_id"]
        test_utterance = trial["test_utterance"]
        enroll_embedding_full = scoring_templates[enroll_id]["full"]
        enroll_embedding_split = scoring_templates[enroll_id]["split"]
        test_embedding_full = scoring_embeddings[test_utterance]["full"]
        test_embedding_split = scoring_embeddings[test_utterance]["split"]
        with torch.inference_mode():
            score_one = torch.mean(torch.matmul(enroll_embedding_full, test_embedding_full.T))
            score_two = torch.mean(torch.matmul(enroll_embedding_split, test_embedding_split.T))
            score = ((score_one + score_two) / 2).item()
        label = 1 if trial["label"] == "1" else 0
        enroll_phrase = script_metadata_by_enroll.get(enroll_id, "") if script_metadata_by_enroll is not None else ""
        test_key = extract_utterance_key(test_utterance)
        test_phrase = script_metadata_by_utt.get(test_key, "") if script_metadata_by_utt is not None else ""
        enroll_phrase_id = extract_phrase_id_from_enroll(enroll_id)
        test_phrase_id = extract_phrase_id_from_utterance(test_utterance)
        phrase_match = int(bool(enroll_phrase_id) and enroll_phrase_id == test_phrase_id)
        scores.append(score)
        labels.append(label)
        phrase_matches.append(phrase_match)
        rows.append(
            {
                "enroll_id": enroll_id,
                "test_utterance": test_utterance,
                "category": trial["category"],
                "label": label,
                "score": score,
                "enroll_phrase": enroll_phrase,
                "test_phrase": test_phrase,
                "enroll_phrase_id": enroll_phrase_id,
                "test_phrase_id": test_phrase_id,
                "phrase_match": phrase_match,
            }
        )

    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64), np.asarray(phrase_matches, dtype=np.int64), rows


def sweep_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    phrase_matches: np.ndarray | None,
    threshold_start: float,
    threshold_end: float,
    threshold_step: float,
    phrase_gate: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    thresholds = np.arange(threshold_start, threshold_end + threshold_step * 0.5, threshold_step, dtype=np.float64)
    positive_mask = labels == 1
    negative_mask = ~positive_mask
    genuine_count = int(np.sum(positive_mask))
    impostor_count = int(np.sum(negative_mask))
    if genuine_count == 0 or impostor_count == 0:
        raise ValueError("RedDots trials must contain both genuine and non-genuine trials")

    decisions = scores[:, None] >= thresholds[None, :]
    if phrase_gate:
        if phrase_matches is None:
            raise ValueError("Phrase-gated evaluation requires phrase match metadata")
        decisions = decisions & (phrase_matches[:, None] == 1)
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
    parser = argparse.ArgumentParser(description="RedDots threshold sweep for ECAPA-TDNN")
    parser.add_argument("--initial_model", type=str, required=True, help="Path to pretrained checkpoint")
    parser.add_argument("--ndx", type=str, default="reddots/ndx/m_part_01.ndx", help="RedDots trial list")
    parser.add_argument("--trn", type=str, default="reddots/ndx/m_part_01.trn", help="RedDots enrollment list")
    parser.add_argument("--script_txt", type=str, default="reddots/infos/script.txt", help="Optional RedDots utterance-to-phrase metadata")
    parser.add_argument("--phrase_gate", action="store_true", help="Require phrase_match to accept a trial in addition to the score threshold")
    parser.add_argument("--pcm_root", type=str, default="reddots/pcm", help="Root directory containing PCM audio")
    parser.add_argument("--output_dir", type=str, default="exps/red-dot", help="Directory to save RedDots sweep outputs")
    parser.add_argument("--output_csv", type=str, default="exps/red-dot/reddots_threshold_sweep.csv", help="CSV file to store threshold sweep results")
    parser.add_argument("--summary_json", type=str, default="exps/red-dot/reddots_threshold_summary.json", help="JSON file to store the final summary")
    parser.add_argument("--threshold_start", type=float, default=-1.0, help="Threshold sweep start")
    parser.add_argument("--threshold_end", type=float, default=1.0, help="Threshold sweep end")
    parser.add_argument("--threshold_step", type=float, default=0.001, help="Threshold sweep step")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    ndx_path = Path(args.ndx)
    trn_path = Path(args.trn)
    script_path = Path(args.script_txt)
    pcm_root = Path(args.pcm_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECAPAModel(lr=args.lr, lr_decay=args.lr_decay, C=args.C, n_class=args.n_class, m=args.m, s=args.s, test_step=args.test_step)
    print(f"Model loaded from {args.initial_model}")
    model.load_parameters(args.initial_model)
    model.eval()

    enrollments = read_enrollment_protocol(trn_path)
    trials = read_trial_protocol(ndx_path)
    script_metadata_by_utt, script_metadata_by_enroll = read_script_metadata(script_path)
    files = collect_unique_files(enrollments, trials)

    cache_name = f"reddots_embeddings_{model_signature(args.initial_model)}_{ndx_path.stem}_{trn_path.stem}.pt"
    cache_file = output_dir / cache_name
    embeddings = extract_embeddings(
        model.speaker_encoder,
        files,
        pcm_root,
        device,
        args.split_batch_size,
        cache_file,
        args.cache_embeddings,
    )
    templates = build_enrollment_templates(embeddings, enrollments)
    scores, labels, phrase_matches, rows = score_trials(
        embeddings,
        templates,
        trials,
        device,
        script_metadata_by_utt=script_metadata_by_utt,
        script_metadata_by_enroll=script_metadata_by_enroll,
    )

    thresholds, far, frr, accuracy = sweep_thresholds(
        scores,
        labels,
        phrase_matches,
        args.threshold_start,
        args.threshold_end,
        args.threshold_step,
        args.phrase_gate,
    )
    best_accuracy_index = int(np.argmax(accuracy))
    eer_index = int(np.nanargmin(np.abs(far - frr)))
    fnrs, fprs, dcf_thresholds = ComputeErrorRates(scores.tolist(), labels.tolist())
    min_dcf, min_dcf_threshold = ComputeMinDcf(fnrs, fprs, dcf_thresholds, 0.05, 1, 1)
    eer = float(max(far[eer_index], frr[eer_index]))
    eer_threshold = float(thresholds[eer_index])
    summary = {
        "total_pairs": len(trials),
        "total_utterances": len(files),
        "cache_hit": bool(args.cache_embeddings and cache_file.exists()),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "far_at_eer": float(far[eer_index]),
        "frr_at_eer": float(frr[eer_index]),
        "min_dcf": float(min_dcf),
        "min_dcf_threshold": float(min_dcf_threshold),
        "best_accuracy": float(accuracy[best_accuracy_index]),
        "best_accuracy_threshold": float(thresholds[best_accuracy_index]),
        "threshold_start": float(args.threshold_start),
        "threshold_end": float(args.threshold_end),
        "threshold_step": float(args.threshold_step),
        "phrase_gate": bool(args.phrase_gate),
        "genuine_trials": int(np.sum(labels == 1)),
        "nongenuine_trials": int(np.sum(labels == 0)),
        "phrase_match_trials": int(np.sum(phrase_matches == 1)),
        "phrase_mismatch_trials": int(np.sum(phrase_matches == 0)),
    }

    save_threshold_csv(output_csv, thresholds, far, frr, accuracy, summary)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    row_csv = output_dir / f"{ndx_path.stem}_trial_scores.csv"
    with row_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "enroll_id",
                "test_utterance",
                "category",
                "label",
                "score",
                "enroll_phrase",
                "test_phrase",
                "enroll_phrase_id",
                "test_phrase_id",
                "phrase_match",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved RedDots sweep CSV: {output_csv}")
    print(f"Saved RedDots summary JSON: {summary_json}")
    print(f"Saved RedDots trial scores: {row_csv}")
    print(f"EER {eer * 100:.2f}%, minDCF {min_dcf:.4f}, best accuracy threshold {summary['best_accuracy_threshold']:.3f}")


if __name__ == "__main__":
    main()