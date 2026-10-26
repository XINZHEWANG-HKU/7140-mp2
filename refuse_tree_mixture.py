"""Select a tree/neural mixture on validation without retraining either model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"

import numpy as np

from common import DATA, load_inputs, read_jsonl
from features_v2 import answer_metadata, transform
from submission_v2 import load_bundle as load_neural, probabilities, choose
from submission_v4 import load_bundle, frozen_neural_probability, tree_scores
from train_v2 import labels, metrics, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-checkpoint", type=Path, required=True)
    parser.add_argument("--neural-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "model.npz").exists():
        parser.error("Use a fresh output directory")
    started = time.perf_counter()
    state = {key: value for key, value in load_bundle(args.tree_checkpoint).items()
             if not key.startswith("neural_")}
    state.update({"neural_" + key: value for key, value in load_neural(args.neural_checkpoint).items()})
    bank, arrays = load_inputs("validation")
    valid, hard, _ = answer_metadata(bank)
    correct = labels(bank, read_jsonl(DATA / "gsm8k_validation.jsonl"))
    scores = tree_scores(transform(bank, arrays["embeddings"], state), state).reshape(-1, 4)
    neural = frozen_neural_probability(bank, arrays, state, valid)
    best = None
    grid = []
    for temperature in (0.5, 1.0, 2.0):
        tree = probabilities(scores / temperature, valid)
        for weight in (1.0, 0.75, 0.5, 0.25, 0.0):
            probability = weight * tree + (1 - weight) * neural
            for alpha in (0.0, 0.1, 0.25):
                choices = choose(bank, probability, valid, alpha)
                record = {"tree_weight": weight, "tree_temperature": temperature,
                          "consensus_alpha": alpha, **metrics(choices, correct, hard)}
                grid.append(record)
                rank = (record["hard_correct"], record["correct"], -alpha, weight, -abs(temperature - 1))
                if best is None or rank > best[0]:
                    best = (rank, record)
    selected = best[1]
    for key in ("tree_weight", "tree_temperature", "consensus_alpha"):
        state[key] = np.asarray(selected[key], dtype=np.float32)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "model.npz"
    np.savez(path, **state)
    if path.stat().st_size > 16 * 2**20:
        raise RuntimeError("Checkpoint exceeds the asset limit")
    sources = []
    for directory in (args.tree_checkpoint, args.neural_checkpoint):
        source = directory / "model.npz" if directory.is_dir() else directory
        sources.append({"path": str(source), "sha256": file_hash(source)})
    report = {"method": "boosted-tree-neural-mixture-refusion", "sources": sources,
              "validation_selected": selected, "fusion_grid": grid,
              "uncompressed_array_bytes": sum(value.nbytes for value in state.values()),
              "checkpoint_bytes": path.stat().st_size, "checkpoint_sha256": file_hash(path),
              "refusion_seconds": time.perf_counter() - started, "source_sha256": file_hash(Path(__file__))}
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "fusion_grid"}, indent=2))


if __name__ == "__main__":
    main()
