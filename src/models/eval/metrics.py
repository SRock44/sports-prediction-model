"""Model evaluation metrics: log-loss, Brier, ECE, pinball, coverage."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def compute_all_winner_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    return {
        "logloss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(np.mean((y_prob >= 0.5) == y_true)),
        "ece": compute_ece(y_true, y_prob),
        "n_samples": len(y_true),
    }


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (reliability diagram metric)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_prob = float(np.mean(y_prob[mask]))
        bin_acc = float(np.mean(y_true[mask]))
        ece += (mask.sum() / n) * abs(bin_acc - bin_prob)
    return float(ece)


def compute_pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> float:
    """Pinball (quantile) loss for a single quantile level."""
    errors = y_true - y_pred
    return float(np.mean(np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)))


def compute_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of observations inside the prediction interval."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index between two distributions."""
    bins = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    psi = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        base_pct = float(np.mean((baseline >= lo) & (baseline < hi))) + 1e-9
        curr_pct = float(np.mean((current >= lo) & (current < hi))) + 1e-9
        psi += (curr_pct - base_pct) * np.log(curr_pct / base_pct)

    return float(psi)
