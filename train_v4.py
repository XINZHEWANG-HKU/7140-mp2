"""Train CPU boosted candidate-ranking trees and export NumPy-only inference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"

import numpy as np
import xgboost as xgb

from common import DATA, PROTOCOL, check_data, load_inputs, read_jsonl
from features_v2 import answer_metadata, fit_normalization, transform
from submission_v2 import load_bundle as load_neural, probabilities, choose
from submission_v4 import frozen_neural_probability, tree_scores
from train_v2 import file_hash, labels, metrics


def export_trees(booster, depth):
    features, thresholds, lefts, rights, leaves, roots = [], [], [], [], [], []
    for serialized in booster.get_dump(dump_format="json"):
        root = json.loads(serialized)
        stack = [root]
        nodes = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.get("children", []))
        offset = len(features)
        roots.append(offset)
        size = max(node["nodeid"] for node in nodes) + 1
        features.extend([-1] * size)
        thresholds.extend([0.0] * size)
        lefts.extend([0] * size)
        rights.extend([0] * size)
        leaves.extend([0.0] * size)
        for node in nodes:
            index = offset + node["nodeid"]
            if "leaf" in node:
                leaves[index] = node["leaf"]
            else:
                features[index] = int(node["split"].lstrip("f"))
                thresholds[index] = node["split_condition"]
                lefts[index] = offset + node["yes"]
                rights[index] = offset + node["no"]
    config = json.loads(booster.save_config())
    base = json.loads(config["learner"]["learner_model_param"]["base_score"])
    base = float(base[0]) if isinstance(base, list) else float(base)
    return {
        "tree_feature": np.asarray(features, dtype=np.int32),
        "tree_threshold": np.asarray(thresholds, dtype=np.float32),
        "tree_left": np.asarray(lefts, dtype=np.int32),
        "tree_right": np.asarray(rights, dtype=np.int32),
        "tree_leaf": np.asarray(leaves, dtype=np.float32),
        "tree_roots": np.asarray(roots, dtype=np.int32),
        "tree_base_score": np.asarray(base, dtype=np.float32),
        "tree_depth": np.asarray(depth, dtype=np.int32),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=Path("runs/model_bundle_v4"))
    parser.add_argument("--neural-checkpoint", type=Path)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=800)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7506)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--no-arithmetic", action="store_true")
    args = parser.parse_args()
    if (args.output / "model.npz").exists():
        parser.error("Output already contains a model; use a new output directory")
    if not 1 <= args.depth <= 6 or args.rounds < 1 or args.patience < 1:
        parser.error("Depth must be 1..6 and rounds/patience must be positive")
    started = time.perf_counter()
    check_data(args.data_dir, ("train", "validation"))
    train_bank, train_arrays = load_inputs("train", args.data_dir)
    val_bank, val_arrays = load_inputs("validation", args.data_dir)
    train_correct = labels(train_bank, read_jsonl(args.data_dir / "gsm8k_train.jsonl"))
    val_correct = labels(val_bank, read_jsonl(args.data_dir / "gsm8k_validation.jsonl"))
    train_valid, train_hard, _ = answer_metadata(train_bank)
    val_valid, val_hard, _ = answer_metadata(val_bank)
    state = fit_normalization(train_bank, train_arrays["embeddings"], not args.no_arithmetic)
    train_x = transform(train_bank, train_arrays["embeddings"], state).reshape(len(train_bank), 4, -1)
    train_x = train_x[train_hard].reshape(-1, train_x.shape[-1])
    val_x = transform(val_bank, val_arrays["embeddings"], state)
    dtrain = xgb.DMatrix(train_x, label=train_correct[train_hard].reshape(-1).astype(np.float32), nthread=4)
    dtrain.set_group(np.full(int(train_hard.sum()), 4, dtype=np.uint32))
    dtrain.set_weight(np.where(train_correct[train_hard].sum(axis=1) == 1, 1.5, 1.0).astype(np.float32))
    dval = xgb.DMatrix(val_x.reshape(len(val_bank), 4, -1)[val_hard].reshape(-1, val_x.shape[-1]),
                       label=val_correct[val_hard].reshape(-1).astype(np.float32), nthread=4)
    dval.set_group(np.full(int(val_hard.sum()), 4, dtype=np.uint32))
    history = {}

    def selection_accuracy(predictions, matrix):
        scores = predictions.reshape(-1, 4)
        scores = np.where(val_valid[val_hard], scores, -1e9)
        choices = scores.argmax(axis=1)
        accuracy = val_correct[val_hard][np.arange(len(choices)), choices].mean()
        return "selection_accuracy", float(accuracy)

    parameters = {
        "objective": "rank:pairwise", "tree_method": "hist", "device": "cpu",
        "nthread": 4, "max_depth": args.depth, "eta": args.learning_rate,
        "subsample": 0.85, "colsample_bytree": 0.65, "min_child_weight": 3,
        "lambda": 10, "alpha": 0.05, "max_bin": 128, "seed": args.seed,
        "lambdarank_pair_method": "mean", "lambdarank_num_pair_per_sample": 4,
        "disable_default_eval_metric": 1,
    }
    trained = xgb.train(parameters, dtrain, num_boost_round=args.rounds, evals=[(dval, "validation")],
                        custom_metric=selection_accuracy, maximize=True, early_stopping_rounds=args.patience,
                        evals_result=history, verbose_eval=20)
    best_iteration = int(trained.best_iteration)
    booster = trained[:best_iteration + 1]
    state.update(export_trees(booster, args.depth))
    exported_scores = tree_scores(val_x, state).reshape(-1, 4)
    full_validation_matrix = xgb.DMatrix(val_x, nthread=4)
    library_scores = booster.predict(full_validation_matrix, output_margin=True).reshape(-1, 4)
    export_error = float(np.max(np.abs(exported_scores - library_scores)))
    np.testing.assert_allclose(exported_scores, library_scores, atol=1e-5, rtol=1e-5)
    neural_hash = None
    neural_probability = None
    if args.neural_checkpoint:
        neural = load_neural(args.neural_checkpoint)
        state.update({"neural_" + key: value for key, value in neural.items()})
        neural_probability = frozen_neural_probability(val_bank, val_arrays, state, val_valid)
        neural_path = args.neural_checkpoint
        neural_path = neural_path / "model.npz" if neural_path.is_dir() else neural_path
        neural_hash = file_hash(neural_path)
    best = None
    fusion_records = []
    # A small predeclared grid, selected on validation only.
    for temperature in (0.5, 1.0, 2.0):
        tree_probability = probabilities(exported_scores / temperature, val_valid)
        for weight in ((1.0, 0.75, 0.5, 0.25, 0.0) if neural_probability is not None else (1.0,)):
            probability = weight * tree_probability
            if neural_probability is not None:
                probability += (1 - weight) * neural_probability
            for alpha in (0.0, 0.1, 0.25):
                choices = choose(val_bank, probability, val_valid, alpha)
                record = {"tree_weight": weight, "tree_temperature": temperature,
                          "consensus_alpha": alpha, **metrics(choices, val_correct, val_hard)}
                fusion_records.append(record)
                rank = (record["hard_correct"], record["correct"], -alpha, weight, -abs(temperature - 1))
                if best is None or rank > best[0]:
                    best = (rank, record)
    selected = best[1]
    state.update({"v4_version": np.asarray(4, dtype=np.int32),
                  "tree_weight": np.asarray(selected["tree_weight"], dtype=np.float32),
                  "tree_temperature": np.asarray(selected["tree_temperature"], dtype=np.float32),
                  "consensus_alpha": np.asarray(selected["consensus_alpha"], dtype=np.float32)})
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "model.npz"
    np.savez(model_path, **state)
    if model_path.stat().st_size > 16 * 2**20:
        raise RuntimeError("Checkpoint exceeds 16 MiB; do not submit it")
    tree_only = metrics(choose(val_bank, probabilities(exported_scores, val_valid), val_valid, 0), val_correct, val_hard)
    report = {
        "method": "boosted-tree-neural-selector-v4", "protocol": PROTOCOL,
        "training_device": "cpu", "threads": 4, "xgboost_version": xgb.__version__,
        "hyperparameters": parameters, "seed": args.seed, "no_arithmetic": args.no_arithmetic,
        "rounds_requested": args.rounds, "patience": args.patience,
        "best_iteration": best_iteration, "trees": len(state["tree_roots"]),
        "validation_tree_only": tree_only, "validation_selected": selected,
        "validation_history": history, "fusion_grid": fusion_records,
        "neural_checkpoint_sha256": neural_hash, "numpy_export_max_absolute_error": export_error,
        "uncompressed_array_bytes": sum(value.nbytes for value in state.values()),
        "checkpoint_bytes": model_path.stat().st_size, "checkpoint_sha256": file_hash(model_path),
        "training_seconds": time.perf_counter() - started,
        "source_sha256": {name: file_hash(Path(__file__).resolve().parent / name) for name in
                           ("train_v4.py", "submission_v4.py", "features_v2.py", "submission_v2.py", "model_utils.py", "train_v2.py")},
    }
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("fusion_grid", "validation_history", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
