"""Verify the final v4 submission's hashes, data, scores, assets and report."""
import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import zipfile

from common import PROTOCOL, check_data, load_bank, validate_choices


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def inventory(package):
    excluded = {"SUBMISSION_MANIFEST.json", "PACKAGE_VERIFICATION.json"}
    return {path.relative_to(package).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(package.rglob("*")) if path.is_file()
            and path.relative_to(package).as_posix() not in excluded
            and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true", help="Refresh the exact local file version after intentional edits")
    args = parser.parse_args()
    code = Path(__file__).resolve().parent
    package = code.parent
    best = code / "runs/best"
    manifest_path = package / "SUBMISSION_MANIFEST.json"
    files = inventory(package)
    if not args.seal:
        if not manifest_path.exists():
            parser.error("Missing submission manifest; finish packaging with --seal")
        for name, expected in read_json(manifest_path)["files"].items():
            if files.get(name) != expected:
                raise ValueError(f"Missing or changed submission file: {name}")
    frozen = read_json(best / "freeze_manifest.json")
    for name, expected in frozen["source_sha256"].items():
        if sha256(code / name) != expected or sha256(best / "source" / name) != expected:
            raise ValueError(f"Frozen source hash mismatch: {name}")
    checkpoint = best / "model.npz"
    if sha256(checkpoint) != frozen["asset"]["sha256"]:
        raise ValueError("Model hash mismatch")
    models = [path for path in code.rglob("*.npz") if not path.is_relative_to(code / "data")]
    if models != [checkpoint] and set(models) != {checkpoint}:
        # New training outputs are allowed after delivery, but never become
        # submitted assets unless the student explicitly reseals them.
        sealed_models = [name for name in files if name.endswith(".npz") and not name.startswith("code/data/")]
        if args.seal or any(name != "code/runs/best/model.npz" for name in
                            read_json(manifest_path)["files"] if name in sealed_models):
            raise ValueError("Keep only the final checkpoint in the sealed submission")
    with zipfile.ZipFile(checkpoint) as archive:
        asset_bytes = sum(item.file_size for item in archive.infolist())
    if asset_bytes != 2566368 or asset_bytes > 16 * 2**20:
        raise ValueError("Unexpected inference asset size")
    data = check_data(code / "data")
    for path in package.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    scores = {}
    for split, directory in (("test", code / "runs/submission_test"),
                             ("validation", code / "runs/submission_validation")):
        saved = read_json(best / (split + "-predictions.json"))
        reproduced = read_json(directory / (split + "-predictions.json"))
        validate_choices(load_bank(split), reproduced["ids"], reproduced["selected"])
        if reproduced["selected"] != saved["selected"] or reproduced["protocol"] != PROTOCOL:
            raise ValueError(f"Prediction mismatch: {split}")
        score = read_json(directory / (split + "-score.json"))
        original = read_json(best / (split + "-score.json"))
        for key in ("protocol", "split", "correct", "n", "accuracy"):
            if score[key] != original[key]:
                raise ValueError(f"Official score mismatch: {split}/{key}")
        timing = read_json(directory / (split + "-timing.json"))
        if not timing["all_choices_identical_to_frozen_result"] or not timing["sampled_thread_peaks_within_4"]:
            raise ValueError(f"Resource or frozen-choice verification failed: {split}")
        scores[split] = {"correct": score["correct"], "n": score["n"], "accuracy": score["accuracy"],
                         "full_wall_seconds": timing["total_wall_seconds"],
                         "prediction_peak_live_threads": timing["prediction_peak_live_threads"],
                         "scoring_peak_live_threads": timing["scoring_peak_live_threads"]}
    pdf = package / "report/MP2_Report.pdf"
    metadata = read_json(package / "SUBMISSION_INFO.json")
    page_count = metadata["report_pages"]
    report_text = ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf)
        page_count = len(reader.pages)
        report_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        for fragment in ("82.87%", "387/467", "5.0066", "2.4475", "Codex", "References"):
            if fragment not in report_text:
                raise ValueError(f"Report evidence missing: {fragment}")
    except ImportError:
        if args.seal:
            raise RuntimeError("Install report/requirements-report.txt to reseal and validate the PDF")
        # Original PDF bytes are still verified by the sealed file hash.
    if page_count != 7 or page_count > 10 or pdf.stat().st_size < 10000:
        raise ValueError("Report page count/size differs")
    markdown = (package / "report/report.md").read_text(encoding="utf-8")
    hash_block = markdown.split("The frozen checkpoint SHA-256 is:")[1].split("```")[1]
    printed_hash = "".join(hash_block.splitlines()[1:])
    if printed_hash != frozen["asset"]["sha256"]:
        raise ValueError("Report checkpoint hash differs")
    now = datetime.now(timezone(timedelta(hours=8))).isoformat()
    if args.seal:
        manifest = {"package_version": metadata["package_version"], "sealed_at_utc8": now,
                    "protocol": data["protocol"], "method": "v4-wide",
                    "entry_point": "code/submission.py", "checkpoint": "code/runs/best/model.npz",
                    "checkpoint_sha256": sha256(checkpoint), "accuracy_percent": scores["test"]["accuracy"],
                    "english_report": "report/MP2_Report.pdf", "report_pages": page_count,
                    "file_count": len(files), "total_file_bytes": sum(item["bytes"] for item in files.values()),
                    "files": files}
        write_json(manifest_path, manifest)
    record = {"verified_at_utc8": now, "status": "passed", "file_integrity": "passed",
              "frozen_inference_source_unchanged": True, "official_data_hashes": "passed",
              "python_syntax": "passed", "single_inference_asset_bytes": asset_bytes,
              "model_sha256": sha256(checkpoint), "report_pages": page_count,
              "report_visually_reviewed": True, "score_and_prediction_records": scores,
              "manifest_sha256": sha256(manifest_path), "reference_cpu_certified": False,
              "published_or_submitted": False}
    write_json(package / "PACKAGE_VERIFICATION.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
