"""Time fresh official prediction and scoring processes from CMD or any shell."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / f"{args.split}-predictions.json"
    score = args.output_dir / f"{args.split}-score.json"
    timing = args.output_dir / f"{args.split}-timing.json"
    code_dir = Path(__file__).resolve().parent
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "4"

    prediction_command = [
        sys.executable, str(code_dir / "predict.py"),
        "--method", str(args.method), "--split", args.split,
        "--data-dir", str(args.data_dir), "--output", str(predictions),
    ]
    if args.checkpoint is not None:
        prediction_command.extend(("--checkpoint", str(args.checkpoint)))
    scoring_command = [
        sys.executable, str(code_dir / "score_predictions.py"),
        "--split", args.split, "--data-dir", str(args.data_dir),
        "--predictions", str(predictions), "--output", str(score),
    ]

    started = time.perf_counter()
    subprocess.run(prediction_command, env=environment, check=True)
    prediction_wall_seconds = time.perf_counter() - started
    scoring_started = time.perf_counter()
    subprocess.run(scoring_command, env=environment, check=True)
    finished = time.perf_counter()

    processor = platform.processor()
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                processor = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass

    result = {
        "split": args.split,
        "prediction_wall_seconds": prediction_wall_seconds,
        "scoring_wall_seconds": finished - scoring_started,
        "total_wall_seconds": finished - started,
        "threads": 4,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": processor,
        "prediction_command": prediction_command,
        "scoring_command": scoring_command,
        "official_score_path": str(score),
    }
    timing.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
