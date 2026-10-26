"""CPU-only NumPy inference for the compact neural MP2 ensemble."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from features_v2 import FEATURE_VERSION, answer_metadata, transform
from model_utils import answer_counts, raw_model_scores


def load_bundle(checkpoint) -> dict:
    if checkpoint is None:
        raise ValueError("Pass --checkpoint pointing to the v2 model directory or model.npz")
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "model.npz"
    with np.load(path, allow_pickle=False) as archive:
        state = {name: archive[name].copy() for name in archive.files}
    if int(state["feature_version"].item()) != FEATURE_VERSION:
        raise ValueError("Checkpoint feature version mismatch")
    return state


def probabilities(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    scores = np.where(valid, scores, np.float32(-1e9))
    values = np.exp(scores - scores.max(axis=1, keepdims=True)) * valid
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def neural_probabilities(features: np.ndarray, state: dict, valid: np.ndarray) -> np.ndarray:
    result = np.zeros(valid.shape, dtype=np.float32)
    members = int(state["members"].item())
    if members < 1:
        raise ValueError("An ensemble must have at least one member")
    for member in range(members):
        prefix = f"member_{member}_"
        hidden = np.maximum(features @ state[prefix + "w1"] + state[prefix + "b1"], 0)
        hidden = np.maximum(hidden @ state[prefix + "w2"] + state[prefix + "b2"], 0)
        scores = (hidden @ state[prefix + "w3"] + state[prefix + "b3"]).reshape(-1, 4)
        result += probabilities(scores, valid) / members
    return result


def legacy_probabilities(bank, arrays, state: dict, valid: np.ndarray) -> np.ndarray:
    legacy = {name[len("legacy_"):]: value for name, value in state.items() if name.startswith("legacy_")}
    scores = raw_model_scores(bank, arrays["embeddings"], legacy)
    alpha = float(legacy["consensus_alpha"].item())
    counts = answer_counts(bank)
    if alpha < 0:
        # Match the legacy deterministic majority fallback rather than inventing
        # a distribution over tied answers.
        result = np.zeros_like(scores)
        choices = np.argmax(np.where(valid, counts, -1), axis=1)
        result[np.arange(len(bank)), choices] = 1.0
        return result
    return probabilities(scores + np.float32(alpha) * np.log(counts), valid)


def choose(bank, probability: np.ndarray, valid: np.ndarray, consensus_alpha: float) -> list[int]:
    score = np.log(np.maximum(probability, 1e-12))
    score += np.float32(consensus_alpha) * np.log(answer_counts(bank))
    return np.argmax(np.where(valid, score, -1e9), axis=1).astype(int).tolist()


def predict(bank, arrays, checkpoint):
    if not bank:
        return []
    state = load_bundle(checkpoint)
    valid, _, _ = answer_metadata(bank)
    neural_weight = float(state["neural_weight"].item())
    if neural_weight > 0:
        features = transform(bank, arrays["embeddings"], state)
        probability = neural_weight * neural_probabilities(features, state, valid)
    else:
        probability = np.zeros(valid.shape, dtype=np.float32)
    if neural_weight < 1:
        if not int(state["has_legacy"].item()):
            raise ValueError("Mixture requires the legacy checkpoint")
        probability += (1 - neural_weight) * legacy_probabilities(bank, arrays, state, valid)
    return choose(bank, probability, valid, float(state["consensus_alpha"].item()))
