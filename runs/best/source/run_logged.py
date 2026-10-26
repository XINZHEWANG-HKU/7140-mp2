"""Run a project Python command with persistent logs in the project directory."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Specify a project Python script after --")
    root = Path(__file__).resolve().parent
    log = args.log.resolve()
    script = (root / command[0]).resolve()
    if not log.is_relative_to(root) or not script.is_relative_to(root) or script.suffix != ".py":
        parser.error("Script and log must be inside the project code directory")
    log.parent.mkdir(parents=True, exist_ok=True)
    temporary = root / "runs" / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for variable in ("TMP", "TEMP", "TMPDIR"):
        environment[variable] = str(temporary)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = "4"
    full_command = [sys.executable, "-B", str(script), *command[1:]]
    wall_started = datetime.now(timezone(timedelta(hours=8))).isoformat()
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(full_command, cwd=root, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    record = {"command": full_command, "started_at_utc8": wall_started,
              "wall_seconds": time.perf_counter() - started, "return_code": return_code,
              "log": str(log), "threads": 4, "temporary_directory": str(temporary)}
    log.with_suffix(".run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
