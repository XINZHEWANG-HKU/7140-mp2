"""Label-free features for the second MP2 selector.

The arithmetic checker recognizes a limited set of explicit numeric equalities.
It evaluates a bounded arithmetic AST, never arbitrary candidate code. Its output
is evidence for a learned selector, not a replacement for the official parser.
"""
from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction
import math
import operator
import re

import numpy as np

from model_utils import _manual_features, parse_candidate_answer


FEATURE_VERSION = 2
_NUMBERS = re.compile(r"(?<![\w.])[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/\d+)?")
_WORDS = re.compile(r"[a-z]+")
_LATEX_FRAC = re.compile(r"\\(?:dfrac|tfrac|frac)\s*")
_LEFT_EXPRESSION = re.compile(r"([\d.() \t+*/^\-]+)$")
_RIGHT_EXPRESSION = re.compile(r"^([\d.() \t+*/^\-]+)")
_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow}
_CUES = (
    "total", "more", "less", "remaining", "left", "profit", "average", "each",
    "per", "times", "percent", "half", "difference", "together", "minutes",
    "hours", "days", "years", "cost", "spent", "rate", "how many",
)


def _braced(text: str, start: int) -> tuple[str, int] | None:
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        return None
    depth = 1
    for position in range(start + 1, min(len(text), start + 512)):
        depth += int(text[position] == "{") - int(text[position] == "}")
        if depth == 0:
            return text[start + 1:position], position + 1
    return None


def normalize_math(text: str) -> str:
    # Bound work on unusually long traces while retaining the complete input for
    # the other features and the official final-answer parser.
    text = text[:24000].replace("−", "-").replace("×", "*").replace("÷", "/")
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    for _ in range(8):
        replacements = []
        for match in _LATEX_FRAC.finditer(text):
            numerator = _braced(text, match.end())
            denominator = _braced(text, numerator[1]) if numerator else None
            if denominator:
                replacements.append((match.start(), denominator[1],
                                     f"(({numerator[0]})/({denominator[0]}))"))
        if not replacements:
            break
        # Only replace disjoint matches per pass; nested fractions are handled
        # in subsequent passes.
        last_end = -1
        disjoint = []
        for replacement in replacements:
            if replacement[0] >= last_end:
                disjoint.append(replacement)
                last_end = replacement[1]
        for start, end, replacement in reversed(disjoint):
            text = text[:start] + replacement + text[end:]
    for old, new in ((r"\times", "*"), (r"\cdot", "*"), (r"\div", "/"),
                     (r"\left", ""), (r"\right", ""), (r"\,", " "),
                     (r"\!", ""), (r"\(", " "), (r"\)", " ")):
        text = text.replace(old, new)
    return text


def bounded_arithmetic(expression: str) -> float:
    expression = expression.strip().rstrip(".").replace("^", "**")
    if not expression or len(expression) > 160:
        raise ValueError("Unsupported expression length")
    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST, depth: int = 0) -> float:
        if depth > 16:
            raise ValueError("Expression too deep")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand, depth + 1)
            if isinstance(node.op, ast.USub):
                value = -value
        elif isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and (abs(right) > 8 or abs(left) > 1e6):
                raise ValueError("Exponent out of range")
            value = _OPERATORS[type(node.op)](left, right)
        else:
            raise ValueError("Only numeric arithmetic is supported")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("Arithmetic out of range")
        return float(value)

    return visit(tree.body)


def arithmetic_features(text: str) -> list[float]:
    normalized = normalize_math(text)
    checked = good = bad = 0
    largest_error = 0.0
    for match in re.finditer("=", normalized):
        if checked >= 64:
            break
        left = _LEFT_EXPRESSION.search(normalized[max(0, match.start() - 160):match.start()])
        right = _RIGHT_EXPRESSION.match(normalized[match.end():match.end() + 160])
        if not left or not right:
            continue
        try:
            lhs = bounded_arithmetic(left.group(1))
            rhs = bounded_arithmetic(right.group(1))
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError):
            continue
        checked += 1
        relative_error = abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))
        largest_error = max(largest_error, relative_error)
        if math.isclose(lhs, rhs, rel_tol=1e-3, abs_tol=1e-5):
            good += 1
        else:
            bad += 1
    return [math.log1p(checked), math.log1p(good), math.log1p(bad),
            bad / max(checked, 1), float(bad > 0), largest_error]


def _numbers(text: str) -> list[float]:
    values = []
    for match in _NUMBERS.finditer(text[:24000]):
        try:
            number = float(Fraction(match.group().replace(",", "")))
            if math.isfinite(number) and abs(number) <= 1e12:
                values.append(number)
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
    return values


def extract_manual(bank: list[dict], embeddings: np.ndarray, use_arithmetic: bool) -> np.ndarray:
    base = _manual_features(bank, embeddings)
    extra = []
    for row in bank:
        texts = [candidate["text"] for candidate in row["candidates"]]
        answers = [parse_candidate_answer(text) for text in texts]
        counts = Counter(answer for answer in answers if answer is not None)
        lengths = np.asarray([len(text) for text in texts], dtype=np.float32)
        question = row["question"].lower()
        question_numbers = _numbers(question)
        question_words = set(_WORDS.findall(question))
        target_words = set(_WORDS.findall(question.rsplit("?", 1)[0].rsplit(".", 1)[-1]))
        cues = [float(cue in question) for cue in _CUES]
        for index, (text, answer) in enumerate(zip(texts, answers)):
            try:
                value = float(Fraction(answer)) if answer is not None else 0.0
            except OverflowError:
                value = -1e12 if answer.startswith("-") else 1e12
            if not math.isfinite(value):
                value = math.copysign(1e12, value)
            value = max(-1e12, min(value, 1e12))
            candidate_numbers = _numbers(text)
            candidate_words = set(_WORDS.findall(text.lower()))
            arithmetic = arithmetic_features(text) if use_arithmetic else [0.0] * 6
            votes = counts.get(answer, 0) if answer is not None else 0
            numerical_overlap = sum(any(math.isclose(number, q, rel_tol=1e-6, abs_tol=1e-6)
                                        for q in question_numbers) for number in candidate_numbers)
            extra.append([
                votes / 4.0, len(counts) / 4.0, max(counts.values(), default=0) / 4.0,
                sum(answer is not None for answer in answers) / 4.0,
                math.copysign(math.log1p(abs(value)), value), float(value < 0),
                float(value == 0), float(value.is_integer()),
                float(any(math.isclose(value, q, rel_tol=1e-6, abs_tol=1e-6) for q in question_numbers)),
                math.log1p(len(question_numbers)),
                numerical_overlap / max(len(candidate_numbers), 1),
                len(question_words & candidate_words) / max(len(question_words), 1),
                len(target_words & candidate_words) / max(len(target_words), 1),
                math.log1p(len(text) / max(float(lengths.mean()), 1.0)),
                float(len(text) == float(lengths.min())), float(len(text) == float(lengths.max())),
                *arithmetic, *cues,
            ])
    return np.concatenate((base, np.asarray(extra, dtype=np.float32)), axis=1)


def answer_metadata(bank: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.asarray([[parse_candidate_answer(c["text"]) is not None
                         for c in row["candidates"]] for row in bank], dtype=bool)
    distinct = np.asarray([len({parse_candidate_answer(c["text"]) for c in row["candidates"]
                              if parse_candidate_answer(c["text"]) is not None}) for row in bank])
    if len(bank) and not valid.any(axis=1).all():
        raise ValueError("Every question must contain a parseable candidate")
    return valid, distinct >= 2, distinct


def fit_normalization(bank: list[dict], embeddings: np.ndarray, use_arithmetic: bool) -> dict:
    flat = embeddings.reshape(-1, embeddings.shape[-1]).astype(np.float32, copy=False)
    manual = extract_manual(bank, embeddings, use_arithmetic)
    return {
        "embedding_mean": flat.mean(axis=0, dtype=np.float64).astype(np.float32),
        "embedding_std": np.maximum(flat.std(axis=0, dtype=np.float64), 1e-4).astype(np.float32),
        "manual_mean": manual.mean(axis=0, dtype=np.float64).astype(np.float32),
        "manual_std": np.maximum(manual.std(axis=0, dtype=np.float64), 1e-4).astype(np.float32),
        "use_arithmetic": np.asarray(int(use_arithmetic), dtype=np.int32),
    }


def transform(bank: list[dict], embeddings: np.ndarray, state: dict) -> np.ndarray:
    n = len(bank)
    if embeddings.shape != (n, 4, 1024):
        raise ValueError("Expected embeddings with shape [N, 4, 1024]")
    normalized = (embeddings - state["embedding_mean"]) / state["embedding_std"]
    normalized = np.clip(normalized, -8.0, 8.0).astype(np.float32)
    centered = normalized - normalized.mean(axis=1, keepdims=True)
    manual = extract_manual(bank, embeddings, bool(state["use_arithmetic"].item()))
    manual = np.clip((manual - state["manual_mean"]) / state["manual_std"], -8.0, 8.0)
    relative_manual = manual.reshape(n, 4, -1)
    relative_manual = relative_manual - relative_manual.mean(axis=1, keepdims=True)
    return np.concatenate((normalized.reshape(n * 4, -1), centered.reshape(n * 4, -1),
                           manual, relative_manual.reshape(n * 4, -1)), axis=1).astype(np.float32)
