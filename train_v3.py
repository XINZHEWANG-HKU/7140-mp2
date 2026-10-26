"""V3: smaller network, stronger dropout, and smoothed multi-positive ranking."""
from pathlib import Path

from train_v2 import main


if __name__ == "__main__":
    main(method_name="regularized-neural-selector-v3", defaults={
        "output": Path("runs/model_bundle_v3"), "hidden": 32,
        "dropout": 0.35, "learning_rate": 0.0005, "weight_decay": 0.05,
        "loss": "uniform", "label_smoothing": 0.1,
    })
