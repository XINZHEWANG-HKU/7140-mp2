"""Reproduce the frozen v4-wide validation ablations; never evaluate test."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from submission_v4 import load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/best"))
    parser.add_argument("--include-no-arithmetic", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(root):
        parser.error("Use a fresh output directory inside code/")
    output.mkdir(parents=True)
    winner = load_bundle(args.checkpoint)

    def run(script, *arguments):
        subprocess.run([sys.executable, "-B", str(root / script), *map(str, arguments)], cwd=root, check=True)

    directories = []
    for name, tree_weight in (("no_tree", 0), ("no_neural", 1)):
        state = dict(winner)
        state["tree_weight"] = np.asarray(tree_weight, np.float32)
        destination = output / name
        destination.mkdir()
        np.savez(destination / "model.npz", **state)
        directories.append(destination)
    if args.include_no_arithmetic:
        neural, trees = output / "neural_training", output / "tree_training"
        run("train_v3.py", "--output", neural, "--seeds", 7506, 7507, 7508,
            "--hidden", 96, "--dropout", .5, "--weight-decay", .1, "--no-arithmetic")
        run("train_v4.py", "--output", trees, "--depth", 4, "--seed", 7506, "--no-arithmetic")
        state = load_bundle(trees)
        with np.load(neural / "model.npz", allow_pickle=False) as archive:
            neural_state = {key: archive[key].copy() for key in archive.files}
        neural_state["consensus_alpha"] = winner["neural_consensus_alpha"].copy()
        neural_state["neural_weight"] = winner["neural_neural_weight"].copy()
        state.update({"neural_" + key: value for key, value in neural_state.items()})
        for key in ("tree_weight", "tree_temperature", "consensus_alpha"):
            state[key] = winner[key].copy()
        destination = output / "no_arithmetic"
        destination.mkdir()
        np.savez(destination / "model.npz", **state)
        directories.append(destination)
    for directory in directories:
        # The generic measurement script compares against the full model, which
        # is inappropriate for an ablation. Use the official evaluator directly.
        predictions = directory / "validation-predictions.json"
        run("predict.py", "--method", "submission.py", "--checkpoint", directory,
            "--split", "validation", "--output", predictions)
        run("score_predictions.py", "--split", "validation", "--predictions", predictions,
            "--output", directory / "validation-score.json")
    summary = {directory.name: json.loads((directory / "validation-score.json").read_text(encoding="utf-8"))
               for directory in directories}
    (output / "ablation_scores.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
