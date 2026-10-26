"""Reproduce the frozen best pipeline from supplied train/validation data."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/reproduced_best"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    neural = args.output / "neural"
    trees = args.output / "trees"
    final = args.output / "final"
    if any((path / "model.npz").exists() for path in (neural, trees, final)):
        parser.error("Output contains existing models; select a fresh output directory")
    environment = os.environ.copy()
    temporary = root / "runs" / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    for name in ("TMP", "TEMP", "TMPDIR"):
        environment[name] = str(temporary)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "4"
    commands = [
        ["train_v3.py", "--output", str(neural), "--seeds", "7506", "7507", "7508",
         "--hidden", "96", "--dropout", "0.5", "--weight-decay", "0.1"],
        ["train_v4.py", "--output", str(trees), "--depth", "4", "--seed", "7506"],
        ["refuse_tree_mixture.py", "--tree-checkpoint", str(trees), "--neural-checkpoint", str(neural),
         "--output", str(final)],
    ]
    for command in commands:
        subprocess.run([sys.executable, "-B", str(root / command[0]), *command[1:]],
                       cwd=root, env=environment, check=True)
    print(f"Reproduced inference checkpoint: {final / 'model.npz'}")


if __name__ == "__main__":
    main()
