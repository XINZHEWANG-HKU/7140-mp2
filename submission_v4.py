"""NumPy inference for compact boosted trees and an optional neural mixture."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from features_v2 import answer_metadata, transform
from model_utils import answer_counts
from submission_v2 import legacy_probabilities, neural_probabilities, probabilities, choose


def tree_scores(features: np.ndarray, state: dict) -> np.ndarray:
    result = np.full(len(features), float(state["tree_base_score"].item()), dtype=np.float32)
    nodes_feature = state["tree_feature"]
    threshold = state["tree_threshold"]
    left, right = state["tree_left"], state["tree_right"]
    for root in state["tree_roots"]:
        node = np.full(len(features), int(root), dtype=np.int32)
        for _ in range(int(state["tree_depth"].item()) + 1):
            split = nodes_feature[node]
            active = np.flatnonzero(split >= 0)
            if not len(active):
                break
            current = node[active]
            go_left = features[active, split[active]] < threshold[current]
            node[active] = np.where(go_left, left[current], right[current])
        if (nodes_feature[node] >= 0).any():
            raise ValueError("Tree depth metadata is inconsistent")
        result += state["tree_leaf"][node]
    return result


def frozen_neural_probability(bank, arrays, state, valid):
    neural = {key[len("neural_"):]: value for key, value in state.items() if key.startswith("neural_")}
    weight = float(neural["neural_weight"].item())
    probability = np.zeros(valid.shape, dtype=np.float32)
    if weight > 0:
        features = transform(bank, arrays["embeddings"], neural)
        probability += weight * neural_probabilities(features, neural, valid)
    if weight < 1:
        probability += (1 - weight) * legacy_probabilities(bank, arrays, neural, valid)
    scores = np.log(np.maximum(probability, 1e-12))
    scores += float(neural["consensus_alpha"].item()) * np.log(answer_counts(bank))
    return probabilities(scores, valid)


def load_bundle(checkpoint):
    if checkpoint is None:
        raise ValueError("This method requires a frozen checkpoint")
    path = Path(checkpoint)
    path = path / "model.npz" if path.is_dir() else path
    with np.load(path, allow_pickle=False) as archive:
        state = {name: archive[name].copy() for name in archive.files}
    if int(state["v4_version"].item()) != 4:
        raise ValueError("Wrong v4 checkpoint version")
    return state


def predict(bank, arrays, checkpoint):
    if not bank:
        return []
    state = load_bundle(checkpoint)
    valid, _, _ = answer_metadata(bank)
    weight = float(state["tree_weight"].item())
    probability = np.zeros(valid.shape, dtype=np.float32)
    if weight > 0:
        features = transform(bank, arrays["embeddings"], state)
        scores = tree_scores(features, state).reshape(-1, 4)
        temperature = float(state["tree_temperature"].item())
        probability += weight * probabilities(scores / temperature, valid)
    if weight < 1:
        probability += (1 - weight) * frozen_neural_probability(bank, arrays, state, valid)
    return choose(bank, probability, valid, float(state["consensus_alpha"].item()))
