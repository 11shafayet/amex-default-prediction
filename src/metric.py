from __future__ import annotations

from typing import Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


def _as_1d_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _validate_inputs(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    labels = _as_1d_array(y_true, "y_true").astype(int)
    predictions = _as_1d_array(y_pred, "y_pred").astype(float)

    if labels.shape[0] != predictions.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if labels.shape[0] == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels: 0 and 1.")
    if labels.sum() == 0:
        raise ValueError("y_true must contain at least one positive label.")

    return labels, predictions


def top_four_percent_captured(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    labels, predictions = _validate_inputs(y_true, y_pred)

    sorted_indices = np.argsort(predictions)[::-1]
    labels_sorted = labels[sorted_indices]

    weights = np.where(labels_sorted == 0, 20, 1)
    cutoff = 0.04 * weights.sum()
    selected_rows = np.cumsum(weights) <= cutoff

    captured_positives = np.sum(labels_sorted[selected_rows] == 1)
    total_positives = np.sum(labels == 1)
    return float(captured_positives / total_positives)


def weighted_gini(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    labels, predictions = _validate_inputs(y_true, y_pred)

    sorted_indices = np.argsort(predictions)[::-1]
    labels_sorted = labels[sorted_indices]

    weights = np.where(labels_sorted == 0, 20, 1)
    weight_random = np.cumsum(weights / weights.sum())

    weighted_positives = labels_sorted * weights
    total_positives = weighted_positives.sum()
    lorentz = np.cumsum(weighted_positives) / total_positives

    return float(np.sum((lorentz - weight_random) * weights))


def normalized_weighted_gini(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    labels, predictions = _validate_inputs(y_true, y_pred)
    return weighted_gini(labels, predictions) / weighted_gini(labels, labels)


def amex_metric(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the official AMEX metric: mean of normalized Gini and top-4 capture."""
    labels, predictions = _validate_inputs(y_true, y_pred)
    gini = normalized_weighted_gini(labels, predictions)
    top_four = top_four_percent_captured(labels, predictions)
    return float(0.5 * (gini + top_four))


# Alias matching the notebook's first Stage 2 function name.
top_four_percent = top_four_percent_captured
