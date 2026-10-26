"""Shared feature extraction and checkpoint helpers for MP2.

This module never opens reference-answer files.  Both training and inference use
the same deterministic feature implementation.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path
import re

import numpy as np


MODEL_FILENAME = "model.npz"
MODEL_VERSION = 1

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
_WORD_RE = re.compile(r"[A-Za-z]+")
_OP_RE = re.compile(r"[+\-*/=]")
_UNCERTAINTY_RE = re.compile(
    r"\b(?:maybe|perhaps|probably|guess|uncertain|cannot|can't|not sure)\b",
    re.IGNORECASE,
)


def parse_candidate_answer(text: str) -> str | None:
    """Match the official scorer's final-answer parsing without importing gold IO."""
    text = str(text)
    tail = text.rsplit("####", 1)[1] if "####" in text else None
    if tail is None:
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
        tail = boxed[-1] if boxed else ""
    match = _NUMBER_RE.search(tail)
    if not match:
        return None
    try:
        return str(Fraction(match.group().replace(",", "")))
    except (ValueError, ZeroDivisionError):
        return None


def answer_counts(bank: list[dict]) -> np.ndarray:
    """Return the number of candidates sharing each candidate's parsed answer."""
    result = np.ones((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        answers = [parse_candidate_answer(c["text"]) for c in row["candidates"]]
        counts = Counter(answer for answer in answers if answer is not None)
        result[row_index] = [counts.get(answer, 1) if answer is not None else 1 for answer in answers]
    return result


def _manual_features(bank: list[dict], embeddings: np.ndarray) -> np.ndarray:
    """Build label-free candidate features, excluding answer-vote counts."""
    n = len(bank)
    flat = embeddings.reshape(-1, embeddings.shape[-1]).astype(np.float32, copy=False)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    normalized = (flat / np.maximum(norms, 1e-8)).reshape(n, 4, -1)
    similarities = np.einsum("nid,njd->nij", normalized, normalized, optimize=True)

    features: list[list[float]] = []
    for row_index, row in enumerate(bank):
        question_length = max(len(row["question"]), 1)
        for candidate_index, candidate in enumerate(row["candidates"]):
            text = candidate["text"]
            lower = text.lower()
            parsed = parse_candidate_answer(text)
            other_sims = np.delete(similarities[row_index, candidate_index], candidate_index)
            numbers = _NUMBER_RE.findall(text)
            boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
            answer_mentions = 0.0
            if parsed is not None:
                try:
                    canonical_numbers = []
                    for value in numbers:
                        canonical_numbers.append(str(Fraction(value.replace(",", ""))))
                    answer_mentions = float(sum(value == parsed for value in canonical_numbers))
                except (ValueError, ZeroDivisionError):
                    answer_mentions = 0.0

            features.append([
                np.log1p(len(text)),
                np.log1p(len(_WORD_RE.findall(text))),
                np.log1p(text.count("\n") + 1),
                np.log1p(len(numbers)),
                np.log1p(len(_OP_RE.findall(text))),
                np.log1p(len(text) / question_length),
                float(bool(boxed)),
                float("final answer" in lower or "answer:" in lower),
                float("step" in lower),
                float(bool(_UNCERTAINTY_RE.search(text))),
                float(parsed is not None),
                np.log1p(answer_mentions),
                float(other_sims.mean()),
                float(other_sims.max()),
                float(other_sims.min()),
                float(similarities[row_index, candidate_index].mean()),
            ])
    return np.asarray(features, dtype=np.float32)


def fit_feature_state(
    bank: list[dict], embeddings: np.ndarray, projection_dim: int, seed: int
) -> dict[str, np.ndarray]:
    """Fit preprocessing statistics on training inputs only."""
    flat = embeddings.reshape(-1, embeddings.shape[-1]).astype(np.float32, copy=False)
    embedding_mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    embedding_std = flat.std(axis=0, dtype=np.float64).astype(np.float32)
    embedding_std = np.maximum(embedding_std, np.float32(1e-5))

    manual = _manual_features(bank, embeddings)
    manual_mean = manual.mean(axis=0, dtype=np.float64).astype(np.float32)
    manual_std = manual.std(axis=0, dtype=np.float64).astype(np.float32)
    manual_std = np.maximum(manual_std, np.float32(1e-5))

    rng = np.random.default_rng(seed)
    projection = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(flat.shape[1]),
        size=(flat.shape[1], projection_dim),
    ).astype(np.float32)
    return {
        "embedding_mean": embedding_mean,
        "embedding_std": embedding_std,
        "manual_mean": manual_mean,
        "manual_std": manual_std,
        "projection": projection,
    }


def transform_features(
    bank: list[dict], embeddings: np.ndarray, state: dict[str, np.ndarray]
) -> np.ndarray:
    """Convert candidates to the matrix consumed by the linear ranker."""
    flat = embeddings.reshape(-1, embeddings.shape[-1]).astype(np.float32, copy=False)
    standardized = (flat - state["embedding_mean"]) / state["embedding_std"]
    standardized = np.clip(standardized, -8.0, 8.0).astype(np.float32, copy=False)
    nonlinear = np.tanh(standardized @ state["projection"]).astype(np.float32, copy=False)
    manual = _manual_features(bank, embeddings)
    manual = (manual - state["manual_mean"]) / state["manual_std"]
    manual = np.clip(manual, -8.0, 8.0).astype(np.float32, copy=False)
    return np.concatenate((standardized, nonlinear, manual), axis=1)


def resolve_checkpoint(checkpoint: Path | str | None) -> Path:
    if checkpoint is None:
        raise ValueError("This method requires --checkpoint pointing to model.npz or its directory")
    path = Path(checkpoint)
    path = path / MODEL_FILENAME if path.is_dir() else path
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_checkpoint(checkpoint: Path | str | None) -> dict[str, np.ndarray]:
    path = resolve_checkpoint(checkpoint)
    with np.load(path, allow_pickle=False) as archive:
        state = {name: archive[name].copy() for name in archive.files}
    version = int(np.asarray(state.get("model_version", -1)).item())
    if version != MODEL_VERSION:
        raise ValueError(f"Unsupported checkpoint version {version}; expected {MODEL_VERSION}")
    return state


def raw_model_scores(
    bank: list[dict], embeddings: np.ndarray, state: dict[str, np.ndarray]
) -> np.ndarray:
    features = transform_features(bank, embeddings, state)
    return (features @ state["weights"] + float(state["bias"].item())).reshape(len(bank), 4)


def select_from_scores(
    bank: list[dict], scores: np.ndarray, consensus_alpha: float
) -> list[int]:
    counts = answer_counts(bank)
    if consensus_alpha < 0.0:
        return np.argmax(counts, axis=1).astype(int).tolist()
    fused = scores + np.float32(consensus_alpha) * np.log(counts)
    return np.argmax(fused, axis=1).astype(int).tolist()

