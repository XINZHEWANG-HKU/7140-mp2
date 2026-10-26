"""Train the compact MP2 candidate ranker using train references only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, check_data, load_inputs, parse_answer, read_jsonl
from model_utils import (
    MODEL_FILENAME,
    MODEL_VERSION,
    answer_counts,
    fit_feature_state,
    select_from_scores,
    transform_features,
)


# -1 is an explicit majority-only fallback, evaluated last so an equally accurate
# learned/fused method wins the tie while the selected method can never trail baseline.
ALPHA_GRID = np.asarray(
    [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, -1.0],
    dtype=np.float32,
)


def build_targets(bank: list[dict], references: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Build uniform listwise targets when several candidates are correct."""
    if len(bank) != len(references):
        raise ValueError("Bank/reference length mismatch")
    correct = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, (row, reference) in enumerate(zip(bank, references)):
        if row["id"] != reference["id"] or row["question"] != reference["question"]:
            raise ValueError("Bank/reference alignment mismatch")
        gold = parse_answer(reference["answer"])
        for candidate_index, candidate in enumerate(row["candidates"]):
            correct[row_index, candidate_index] = float(parse_answer(candidate["text"]) == gold)
        if not correct[row_index].any():
            raise ValueError(f"No correct candidate for training row {row['id']}")
    targets = correct / correct.sum(axis=1, keepdims=True)
    return targets, correct.astype(bool)


def accuracy(choices: list[int], correct: np.ndarray) -> float:
    chosen = correct[np.arange(len(correct)), np.asarray(choices)]
    return 100.0 * float(chosen.mean())


def choose_alpha(bank: list[dict], scores: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    best_alpha = 0.0
    best_accuracy = -1.0
    for alpha in ALPHA_GRID:
        choices = select_from_scores(bank, scores, float(alpha))
        candidate_accuracy = accuracy(choices, correct)
        if candidate_accuracy > best_accuracy:
            best_accuracy = candidate_accuracy
            best_alpha = float(alpha)
    return best_alpha, best_accuracy


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def train_ranker(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_bank: list[dict],
    validation_correct: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    l2: float,
    patience: int,
    seed: int,
) -> tuple[np.ndarray, float, float, dict]:
    """Optimize listwise cross-entropy with Adam and validation early stopping."""
    rng = np.random.default_rng(seed)
    dimension = train_features.shape[1]
    weights = np.zeros(dimension, dtype=np.float32)
    bias = np.float32(0.0)
    first_w = np.zeros_like(weights)
    second_w = np.zeros_like(weights)
    first_b = np.float32(0.0)
    second_b = np.float32(0.0)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    step = 0

    best = None
    stale_epochs = 0
    history: list[dict] = []
    question_indices = np.arange(len(train_targets))

    for epoch in range(1, epochs + 1):
        rng.shuffle(question_indices)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, len(question_indices), batch_size):
            batch_questions = question_indices[start : start + batch_size]
            candidate_indices = (batch_questions[:, None] * 4 + np.arange(4)).reshape(-1)
            x = train_features[candidate_indices]
            target = train_targets[batch_questions]
            logits = (x @ weights + bias).reshape(-1, 4)
            probabilities = softmax(logits)
            batch_n = len(batch_questions)
            epoch_loss += float(-np.sum(target * np.log(np.maximum(probabilities, 1e-8))))
            seen += batch_n

            gradient_logits = ((probabilities - target) / batch_n).reshape(-1)
            gradient_w = x.T @ gradient_logits + l2 * weights
            gradient_b = np.float32(gradient_logits.sum())

            step += 1
            first_w = beta1 * first_w + (1.0 - beta1) * gradient_w
            second_w = beta2 * second_w + (1.0 - beta2) * np.square(gradient_w)
            first_b = beta1 * first_b + (1.0 - beta1) * gradient_b
            second_b = beta2 * second_b + (1.0 - beta2) * gradient_b * gradient_b
            corrected_first_w = first_w / (1.0 - beta1**step)
            corrected_second_w = second_w / (1.0 - beta2**step)
            corrected_first_b = first_b / (1.0 - beta1**step)
            corrected_second_b = second_b / (1.0 - beta2**step)
            weights -= learning_rate * corrected_first_w / (np.sqrt(corrected_second_w) + epsilon)
            bias -= learning_rate * corrected_first_b / (np.sqrt(corrected_second_b) + epsilon)

        validation_scores = (validation_features @ weights + bias).reshape(-1, 4)
        model_choices = np.argmax(validation_scores, axis=1).astype(int).tolist()
        model_accuracy = accuracy(model_choices, validation_correct)
        alpha, fused_accuracy = choose_alpha(validation_bank, validation_scores, validation_correct)
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(seen, 1),
            "validation_model_accuracy": model_accuracy,
            "validation_fused_accuracy": fused_accuracy,
            "consensus_alpha": alpha,
        }
        history.append(record)
        print(json.dumps(record))

        # Accuracy is primary; loss resolves exact ties without influencing the protocol.
        rank = (fused_accuracy, model_accuracy, -record["train_loss"])
        if best is None or rank > best[0]:
            best = (rank, weights.copy(), float(bias), alpha, record.copy())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best is None:
        raise RuntimeError("Training produced no checkpoint")
    _, best_weights, best_bias, best_alpha, best_record = best
    return best_weights, best_bias, best_alpha, {"best": best_record, "history": history}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=Path("runs/model_bundle"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7506)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.projection_dim < 1:
        parser.error("epochs, patience, batch-size and projection-dim must be positive")

    started = time.perf_counter()
    check_data(args.data_dir, ("train", "validation"))
    train_bank, train_arrays = load_inputs("train", args.data_dir)
    validation_bank, validation_arrays = load_inputs("validation", args.data_dir)
    train_references = read_jsonl(args.data_dir / "gsm8k_train.jsonl")
    validation_references = read_jsonl(args.data_dir / "gsm8k_validation.jsonl")
    train_targets, _ = build_targets(train_bank, train_references)
    _, validation_correct = build_targets(validation_bank, validation_references)

    feature_state = fit_feature_state(
        train_bank, train_arrays["embeddings"], args.projection_dim, args.seed
    )
    train_features = transform_features(train_bank, train_arrays["embeddings"], feature_state)
    validation_features = transform_features(
        validation_bank, validation_arrays["embeddings"], feature_state
    )

    weights, bias, alpha, training_log = train_ranker(
        train_features,
        train_targets,
        validation_features,
        validation_bank,
        validation_correct,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        l2=args.l2,
        patience=args.patience,
        seed=args.seed,
    )

    baseline_choices = np.argmax(answer_counts(validation_bank), axis=1).astype(int).tolist()
    baseline_accuracy = accuracy(baseline_choices, validation_correct)
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / MODEL_FILENAME
    np.savez(
        model_path,
        model_version=np.asarray(MODEL_VERSION, dtype=np.int32),
        weights=weights.astype(np.float32),
        bias=np.asarray(bias, dtype=np.float32),
        consensus_alpha=np.asarray(alpha, dtype=np.float32),
        **feature_state,
    )

    asset_bytes = sum(array.nbytes for array in feature_state.values()) + weights.nbytes + 12
    report = {
        "method": "compact-listwise-ranker-v1",
        "seed": args.seed,
        "hyperparameters": {
            "epochs_requested": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "projection_dim": args.projection_dim,
        },
        "training_examples": len(train_bank),
        "selection_examples": len(validation_bank),
        "validation_majority_accuracy": baseline_accuracy,
        "validation_best": training_log["best"],
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "uncompressed_array_bytes": asset_bytes,
        "training_seconds": time.perf_counter() - started,
        "history": training_log["history"],
    }
    report_path = args.output / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
