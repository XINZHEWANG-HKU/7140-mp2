"""Read-only validation of the frozen final selector and inference assets."""
from __future__ import annotations

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"

import json
from pathlib import Path
import zipfile

import numpy as np

from common import load_inputs
from submission_v4 import load_bundle, predict


def main():
    bank, arrays = load_inputs("validation")
    bank = bank[:9]
    arrays = {"embeddings": arrays["embeddings"][:9]}
    path = Path("runs/best/model.npz")
    state = load_bundle(path)
    original = predict(bank, arrays, path)
    assert all(type(choice) is int and 0 <= choice < 4 for choice in original)
    assert original == sum([predict(bank[i:i + 1], {"embeddings": arrays["embeddings"][i:i + 1]}, path)
                            for i in range(len(bank))], [])
    renamed = [{**row, "id": f"new-audit-id-{index}"} for index, row in enumerate(bank)]
    assert predict(renamed, arrays, path) == original
    order = [2, 0, 3, 1]
    shuffled = [{**row, "candidates": [{**row["candidates"][index], "index": position}
                                     for position, index in enumerate(order)]} for row in bank]
    choices = predict(shuffled, {"embeddings": arrays["embeddings"][:, order]}, path)
    assert [order[choice] for choice in choices] == original
    with zipfile.ZipFile(path) as archive:
        uncompressed_bytes = sum(item.file_size for item in archive.infolist())
    assert uncompressed_bytes <= 16 * 2**20
    assert not any(name.startswith("neural_legacy_") for name in state)
    print(json.dumps({"variable_bank_size": "passed", "arbitrary_ids": "passed",
                      "candidate_permutation": "passed", "python_integer_choices": "passed",
                      "uncompressed_checkpoint_bytes": uncompressed_bytes,
                      "uncompressed_array_bytes": sum(value.nbytes for value in state.values()),
                      "inference_members": int(state["neural_members"].item()),
                      "inference_trees": len(state["tree_roots"])}, indent=2))


if __name__ == "__main__":
    main()
