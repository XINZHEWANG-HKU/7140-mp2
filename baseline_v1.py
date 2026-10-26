"""Final MP2 candidate selector.

Inference is offline and uses only the bank, supplied embeddings and the frozen
checkpoint.  Reference answers are deliberately inaccessible from this module.
"""
from __future__ import annotations

import numpy as np

from model_utils import answer_counts, load_checkpoint, raw_model_scores, select_from_scores


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order."""
    state = load_checkpoint(checkpoint)
    scores = raw_model_scores(bank, arrays["embeddings"], state)
    alpha = float(state["consensus_alpha"].item())
    return select_from_scores(bank, scores, alpha)


def predict_model_only(bank, arrays, checkpoint):
    """Ablation: learned ranker without parsed-answer consensus."""
    state = load_checkpoint(checkpoint)
    scores = raw_model_scores(bank, arrays["embeddings"], state)
    return np.argmax(scores, axis=1).astype(int).tolist()


def predict_majority(bank, arrays=None, checkpoint=None):
    """Training-free baseline: choose the most frequent parsed final answer."""
    counts = answer_counts(bank)
    return np.argmax(counts, axis=1).astype(int).tolist()
