"""Train small CPU neural rankers from scratch using supplied training labels.

Validation selects epochs and a small ensemble/fusion grid. Test inputs and
references are never loaded here. Inference exports contain only NumPy arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

# Set before importing libraries. One process, no data-loader workers or GPUs.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "4"

import numpy as np
import torch
from torch import nn

from common import DATA, PROTOCOL, check_data, load_inputs, parse_answer, read_jsonl
from features_v2 import FEATURE_VERSION, answer_metadata, fit_normalization, transform
from model_utils import load_checkpoint, raw_model_scores, select_from_scores
from submission_v2 import choose, legacy_probabilities, neural_probabilities


ASSET_LIMIT = 16 * 2**20


def labels(bank: list[dict], references: list[dict]) -> np.ndarray:
    if len(bank) != len(references):
        raise ValueError("Reference count mismatch")
    result = np.zeros((len(bank), 4), dtype=bool)
    for index, (row, reference) in enumerate(zip(bank, references)):
        if row["id"] != reference["id"] or row["question"] != reference["question"]:
            raise ValueError("Reference alignment mismatch")
        gold = parse_answer(reference["answer"])
        if gold is None:
            raise ValueError("Reference has no parsed answer")
        result[index] = [parse_answer(candidate["text"]) == gold for candidate in row["candidates"]]
    if not result.any(axis=1).all():
        raise ValueError("Every training/validation question needs a correct candidate")
    return result


def metrics(choices, correct: np.ndarray, hard: np.ndarray) -> dict:
    right = correct[np.arange(len(correct)), np.asarray(choices)]
    single = correct.sum(axis=1) == 1
    return {
        "correct": int(right.sum()), "n": len(correct), "accuracy": 100 * float(right.mean()),
        "hard_correct": int(right[hard].sum()), "hard_n": int(hard.sum()),
        "hard_accuracy": 100 * float(right[hard].mean()) if hard.any() else None,
        "single_correct_candidate_n": int(single.sum()),
        "single_correct_candidate_accuracy": 100 * float(right[single].mean()) if single.any() else None,
    }


class Ranker(nn.Module):
    def __init__(self, dimension: int, hidden: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimension, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, features):
        return self.layers(features).reshape(-1, 4)


def marginal_loss(scores, correct, valid, group_weights=None):
    # A group with several correct choices only needs probability mass on any
    # of them. Unlike uniform targets, this does not demand equal confidence.
    scores = scores.masked_fill(~valid, -1e9)
    positive_scores = scores.masked_fill(~correct, -1e9)
    per_question = torch.logsumexp(scores, dim=1) - torch.logsumexp(positive_scores, dim=1)
    if group_weights is None:
        return per_question.mean()
    return (per_question * group_weights).sum() / group_weights.sum()


def regularized_listwise_loss(scores, correct, valid, group_weights, smoothing):
    scores = scores.masked_fill(~valid, -1e9)
    target = correct.float() / correct.sum(dim=1, keepdim=True)
    target = (1 - smoothing) * target + smoothing * valid.float() / valid.sum(dim=1, keepdim=True)
    per_question = -(target * torch.log_softmax(scores, dim=1)).sum(dim=1)
    return (per_question * group_weights).sum() / group_weights.sum()


def export_model(model: Ranker, prefix: str) -> dict:
    state = {}
    for name, layer_index in (("1", 0), ("2", 3), ("3", 6)):
        layer = model.layers[layer_index]
        state[prefix + "w" + name] = layer.weight.detach().cpu().numpy().T.copy()
        state[prefix + "b" + name] = layer.bias.detach().cpu().numpy().copy()
    return state


def fit_member(train_x, train_correct, train_valid, val_x, val_correct, val_valid, hard, args, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Ranker(train_x.shape[1], args.hidden, args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    x = torch.from_numpy(train_x)
    positives = torch.from_numpy(train_correct)
    valid = torch.from_numpy(train_valid)
    vx = torch.from_numpy(val_x)
    vp = torch.from_numpy(val_correct)
    vv = torch.from_numpy(val_valid)
    # Prioritize the minority-correct cases while retaining every informative
    # training question. The weight is fixed before validation inspection.
    weights = torch.from_numpy(np.where(train_correct.sum(axis=1) == 1, 1.5, 1.0).astype(np.float32))
    best_rank = None
    best_arrays = None
    best_record = None
    history = []
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = rng.permutation(len(train_correct))
        total_loss = 0.0
        total_seen = 0
        for start in range(0, len(permutation), args.batch_size):
            group = permutation[start:start + args.batch_size]
            candidate_indices = (group[:, None] * 4 + np.arange(4)).reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            batch_scores = model(x[candidate_indices])
            if getattr(args, "loss", "marginal") == "uniform":
                loss = regularized_listwise_loss(batch_scores, positives[group], valid[group],
                                                weights[group], args.label_smoothing)
            else:
                loss = marginal_loss(batch_scores, positives[group], valid[group], weights[group])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(group)
            total_seen += len(group)
        model.eval()
        with torch.no_grad():
            scores = model(vx).masked_fill(~vv, -1e9)
            choices = scores.argmax(dim=1).tolist()
            validation_loss = float(marginal_loss(scores[hard], vp[hard], vv[hard]))
        record = {
            "seed": seed, "epoch": epoch, "training_loss": total_loss / total_seen,
            "validation_marginal_loss_on_hard": validation_loss,
            **metrics(choices, val_correct, hard),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        rank = (record["hard_correct"], record["correct"], -validation_loss)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_arrays = export_model(model, "")
            best_record = record.copy()
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_arrays is None:
        raise RuntimeError("No trained checkpoint produced")
    return best_arrays, {"seed": seed, "best": best_record, "history": history,
                         "training_seconds": time.perf_counter() - started}


def select_fusion(bank, arrays, state, neural_probability, correct, valid, hard, has_legacy):
    legacy_probability = legacy_probabilities(bank, arrays, state, valid) if has_legacy else None
    weight_grid = (1.0, 0.75, 0.5, 0.25, 0.0) if has_legacy else (1.0,)
    records = []
    best = None
    for weight in weight_grid:
        probability = weight * neural_probability
        if has_legacy:
            probability = probability + (1 - weight) * legacy_probability
        for alpha in (0.0, 0.1, 0.25, 0.5):
            choices = choose(bank, probability, valid, alpha)
            record = {"neural_weight": weight, "consensus_alpha": alpha, **metrics(choices, correct, hard)}
            records.append(record)
            rank = (record["hard_correct"], record["correct"], -alpha, weight)
            if best is None or rank > best[0]:
                best = (rank, record)
    return best[1], records


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(method_name="compact-neural-selector-v2", defaults=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=Path("runs/model_bundle_v2"))
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7506, 7507, 7508])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--no-arithmetic", action="store_true", help="Retrain a feature ablation under the same protocol")
    parser.add_argument("--loss", choices=("marginal", "uniform"), default="marginal")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.hidden) < 1:
        parser.error("Integer training settings must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("Seeds must be distinct nonnegative integers")
    if not 0 <= args.dropout < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Invalid dropout, learning rate, or weight decay")
    if not 0 <= args.label_smoothing < 1:
        parser.error("Label smoothing must be in [0, 1)")
    if (args.output / "model.npz").exists():
        parser.error("Output already has model.npz; choose a new --output to preserve previous work")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    check_data(args.data_dir, ("train", "validation"))
    train_bank, train_arrays = load_inputs("train", args.data_dir)
    val_bank, val_arrays = load_inputs("validation", args.data_dir)
    train_correct = labels(train_bank, read_jsonl(args.data_dir / "gsm8k_train.jsonl"))
    val_correct = labels(val_bank, read_jsonl(args.data_dir / "gsm8k_validation.jsonl"))
    train_valid, train_hard, _ = answer_metadata(train_bank)
    val_valid, val_hard, _ = answer_metadata(val_bank)
    if not train_hard.any() or not val_hard.any():
        raise ValueError("Training and validation must contain answer-disagreement groups")
    state = fit_normalization(train_bank, train_arrays["embeddings"], not args.no_arithmetic)
    train_x = transform(train_bank, train_arrays["embeddings"], state).reshape(len(train_bank), 4, -1)
    train_x = train_x[train_hard].reshape(-1, train_x.shape[-1]).copy()
    val_x = transform(val_bank, val_arrays["embeddings"], state)
    estimated_bytes = sum(value.nbytes for value in state.values())
    estimated_bytes += len(args.seeds) * 4 * (train_x.shape[1] * args.hidden + args.hidden * 32 + args.hidden + 65)
    legacy_hash = None
    if args.baseline_checkpoint:
        legacy = load_checkpoint(args.baseline_checkpoint)
        state.update({"legacy_" + key: value for key, value in legacy.items()})
        legacy_path = args.baseline_checkpoint
        if legacy_path.is_dir():
            legacy_path = legacy_path / "model.npz"
        legacy_hash = file_hash(legacy_path)
        estimated_bytes += sum(value.nbytes for value in legacy.values())
    if estimated_bytes + 65536 > ASSET_LIMIT:
        parser.error("Requested ensemble would exceed 16 MiB; reduce --hidden or the number of --seeds")

    state.update({"feature_version": np.asarray(FEATURE_VERSION, dtype=np.int32),
                  "members": np.asarray(len(args.seeds), dtype=np.int32),
                  "has_legacy": np.asarray(int(args.baseline_checkpoint is not None), dtype=np.int32)})
    training_records = []
    for member, seed in enumerate(args.seeds):
        arrays, record = fit_member(
            train_x, train_correct[train_hard], train_valid[train_hard],
            val_x, val_correct, val_valid, val_hard, args, seed,
        )
        state.update({f"member_{member}_" + name: value for name, value in arrays.items()})
        training_records.append(record)
    neural_probability = neural_probabilities(val_x, state, val_valid)
    neural_metrics = metrics(choose(val_bank, neural_probability, val_valid, 0.0), val_correct, val_hard)
    best, fusion_records = select_fusion(
        val_bank, val_arrays, state, neural_probability, val_correct, val_valid, val_hard,
        args.baseline_checkpoint is not None,
    )
    state["neural_weight"] = np.asarray(best["neural_weight"], dtype=np.float32)
    state["consensus_alpha"] = np.asarray(best["consensus_alpha"], dtype=np.float32)
    legacy_metrics = None
    if args.baseline_checkpoint:
        legacy = {key[len("legacy_"):]: value for key, value in state.items() if key.startswith("legacy_")}
        old_scores = raw_model_scores(val_bank, val_arrays["embeddings"], legacy)
        old_choices = select_from_scores(val_bank, old_scores, float(legacy["consensus_alpha"].item()))
        legacy_metrics = metrics(old_choices, val_correct, val_hard)

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "model.npz"
    np.savez(model_path, **state)
    if model_path.stat().st_size > ASSET_LIMIT:
        raise RuntimeError("Checkpoint exceeds the asset limit; do not submit it")
    source_dir = Path(__file__).resolve().parent
    report = {
        "protocol": PROTOCOL, "method": method_name,
        "training_device": "cpu", "threads": 4, "python_platform": platform.platform(),
        "numpy_version": np.__version__, "torch_version": torch.__version__,
        "seeds": args.seeds, "hyperparameters": {key: value for key, value in vars(args).items()
                                                       if key not in ("data_dir", "output", "baseline_checkpoint")},
        "train_questions": len(train_bank), "informative_training_questions": int(train_hard.sum()),
        "validation_questions": len(val_bank), "validation_hard_questions": int(val_hard.sum()),
        "selection_rule": "hard validation correct, then overall correct; never test performance",
        "baseline_checkpoint_sha256": legacy_hash, "validation_legacy": legacy_metrics,
        "validation_neural_ensemble": neural_metrics, "validation_selected": best,
        "fusion_grid": fusion_records, "members": training_records,
        "uncompressed_array_bytes": sum(value.nbytes for value in state.values()),
        "checkpoint_bytes": model_path.stat().st_size, "checkpoint_sha256": file_hash(model_path),
        "source_sha256": {name: file_hash(source_dir / name) for name in
                           ("train_v2.py", "features_v2.py", "submission_v2.py", "model_utils.py", "common.py")},
        "training_seconds": time.perf_counter() - started,
    }
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("members", "fusion_grid", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
