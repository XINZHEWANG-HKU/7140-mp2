"""Reproduce frozen predictions with sequential processes and OS thread auditing."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


class ThreadEntry(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                ("th32ThreadID", wintypes.DWORD), ("th32OwnerProcessID", wintypes.DWORD),
                ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG), ("dwFlags", wintypes.DWORD)]


def thread_count(pid):
    if sys.platform == "win32":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
        kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel.Thread32First.argtypes = (wintypes.HANDLE, ctypes.POINTER(ThreadEntry))
        kernel.Thread32Next.argtypes = (wintypes.HANDLE, ctypes.POINTER(ThreadEntry))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        snapshot = kernel.CreateToolhelp32Snapshot(4, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = ThreadEntry()
            entry.dwSize = ctypes.sizeof(entry)
            present, count = kernel.Thread32First(snapshot, ctypes.byref(entry)), 0
            while present:
                count += int(entry.th32OwnerProcessID == pid)
                present = kernel.Thread32Next(snapshot, ctypes.byref(entry))
            return count
        finally:
            kernel.CloseHandle(snapshot)
    directory = Path("/proc") / str(pid) / "task"
    try:
        return len(list(directory.iterdir())) if directory.exists() else None
    except FileNotFoundError:
        return None


def timed_process(command, environment, cwd):
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=cwd, env=environment)
    peak = None
    while process.poll() is None:
        count = thread_count(process.pid)
        if count is not None:
            peak = max(peak or 0, count)
        time.sleep(.01)
    seconds = time.perf_counter() - started
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, command)
    return seconds, peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--method", type=Path, default=Path("submission.py"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/best"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists():
        parser.error("Use a new output directory; original evidence is never overwritten")
    if not output.is_relative_to(root):
        parser.error("Keep evaluation output inside code/")
    output.mkdir(parents=True)
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    predictions, score = (output / (args.split + suffix) for suffix in ("-predictions.json", "-score.json"))
    predictor = [sys.executable, "-B", str(root / "predict.py"), "--method", str(args.method),
                 "--checkpoint", str(args.checkpoint), "--split", args.split, "--data-dir", str(root / "data"),
                 "--output", str(predictions)]
    scorer = [sys.executable, "-B", str(root / "score_predictions.py"), "--split", args.split,
              "--data-dir", str(root / "data"), "--predictions", str(predictions), "--output", str(score)]
    started = time.perf_counter()
    prediction_seconds, prediction_threads = timed_process(predictor, environment, root)
    scoring_seconds, scoring_threads = timed_process(scorer, environment, root)
    elapsed = time.perf_counter() - started
    saved = json.loads((root / "runs/best" / (args.split + "-predictions.json")).read_text(encoding="utf-8"))
    fresh = json.loads(predictions.read_text(encoding="utf-8"))
    identical = fresh["ids"] == saved["ids"] and fresh["selected"] == saved["selected"]
    processor = platform.processor()
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            processor = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    record = {"split": args.split, "prediction_wall_seconds": prediction_seconds,
              "scoring_wall_seconds": scoring_seconds, "total_wall_seconds": elapsed,
              "library_thread_limit": 1, "prediction_peak_live_threads": prediction_threads,
              "scoring_peak_live_threads": scoring_threads, "processes_sequential": True,
              "monitoring_overhead_included": True, "processor": processor,
              "python_executable": sys.executable, "platform": platform.platform(),
              "prediction_command": predictor, "scoring_command": scorer,
              "all_choices_identical_to_frozen_result": identical,
              "weights_or_hyperparameters_changed": False, "reference_cpu_certified": False,
              "within_30_seconds_locally": elapsed <= 30,
              "sampled_thread_peaks_within_4": all(value <= 4 for value in
                  (prediction_threads, scoring_threads) if value is not None)}
    (output / (args.split + "-timing.json")).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    if not identical or not record["within_30_seconds_locally"] or not record["sampled_thread_peaks_within_4"]:
        raise RuntimeError("Frozen reproduction or local resource check failed")


if __name__ == "__main__":
    main()
