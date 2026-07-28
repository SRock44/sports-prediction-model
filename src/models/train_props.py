"""LightGBM quantile regression props training pipeline.

Trains one model per (sport, stat) combination.
Outputs quantiles [0.10, 0.25, 0.50, 0.75, 0.90] for the predictive distribution.
Implied P(over X.5) is interpolated from the quantile CDF.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.core.logging import get_logger
from src.core.time import utc_now
from src.features.common import exponential_decay_weight
from src.models.eval.metrics import compute_coverage, compute_pinball_loss
from src.models.registry import log_model_run

# A working XGBoost CUDA device does NOT imply LightGBM's OpenCL "gpu" device
# works too - they're different backends (see train_winner.py's
# _lgb_gpu_available, added 2026-07-09 for exactly this bug: this function used
# to infer LGB support from XGBoost's CUDA probe, which meant every prop model
# silently trained on CPU at full core count instead of GPU - the direct cause
# of the 2026-07-10 thermal incident, two of these pegging every core to 100%
# in a small ITX case pushed the CPU to 102C). Reuse train_winner's real probe
# instead of re-deriving a second, wrong one here.
from src.models.train_winner import _lgb_gpu_available

log = get_logger(__name__)

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

# Recency decay: props are more player-specific, so higher λ (forgets faster)
_LAMBDA = {
    "nba": 0.50,
    "mlb": 0.40,
    "nhl": 0.50,  # matches nba's daily-cadence props decay
    "nascar": 0.25,  # matches the winner model's "nascar": 0.20 in train_winner.py,
    # slightly faster since finishing position is more form-sensitive race-to-race
}

_LGB_DEVICE = "gpu" if _lgb_gpu_available() else "cpu"

# LightGBM defaults n_jobs=-1 (every logical core) REGARDLESS of device="gpu" -
# GPU only offloads the histogram-building kernel, everything else still spins
# up LightGBM's full CPU thread pool. With train_props running two sports at
# once, that's two processes each grabbing every core - the actual cause of
# the 2026-07-10 thermal incident (102C package temp), not the CPU-vs-GPU
# device choice itself. Cap it so two concurrent runs can't oversubscribe.
_LGB_N_JOBS = 4


def train_props_model(
    sport: str,
    stat: str,
    training_df: pd.DataFrame,
    feature_names: list[str],
    holdout_df: pd.DataFrame,
    n_optuna_trials: int = 30,
    run_name: str | None = None,
) -> tuple[str, dict[str, float]]:
    """Train a LightGBM multi-quantile model. Returns (mlflow_run_id, metrics)."""
    lam = _LAMBDA.get(sport, 0.45)

    training_df = training_df.sort_values("scheduled_utc").reset_index(drop=True)
    split_idx = int(len(training_df) * 0.9)
    train_part = training_df.iloc[:split_idx]

    X_train = train_part[feature_names].values.astype(np.float32)
    y_train = train_part["target"].values.astype(np.float32)
    X_hold = holdout_df[feature_names].values.astype(np.float32)
    y_hold = holdout_df["target"].values.astype(np.float32)

    now = utc_now()

    def weights(df: pd.DataFrame) -> np.ndarray:
        days = (now - pd.to_datetime(df["scheduled_utc"])).dt.total_seconds() / 86400
        return np.array([exponential_decay_weight(d, lam) for d in days], dtype=np.float32)

    w_train = weights(train_part)

    # ── Optuna search on median quantile (q=0.5 minimises MAE) ───────────────
    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        m = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.5,
            device=_LGB_DEVICE,
            verbose=-1,
            n_jobs=_LGB_N_JOBS,
            **params,
        )
        m.fit(X_train, y_train, sample_weight=w_train)
        pred = m.predict(X_hold)
        return float(mean_absolute_error(y_hold, pred))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_optuna_trials, timeout=120)
    best_params = study.best_params
    log.info("optuna.props.best", sport=sport, stat=stat, params=best_params)

    # ── Train one model per quantile ──────────────────────────────────────────
    X_all = training_df[feature_names].values.astype(np.float32)
    y_all = training_df["target"].values.astype(np.float32)
    w_all = weights(training_df)

    quantile_models: dict[float, lgb.LGBMRegressor] = {}
    predictions: dict[float, np.ndarray] = {}

    for q in QUANTILES:
        m = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            device=_LGB_DEVICE,
            verbose=-1,
            n_jobs=_LGB_N_JOBS,
            **best_params,
        )
        m.fit(X_all, y_all, sample_weight=w_all)
        quantile_models[q] = m
        predictions[q] = m.predict(X_hold)

    # ── Evaluation ────────────────────────────────────────────────────────────
    metrics: dict[str, float] = {}
    for q in QUANTILES:
        metrics[f"pinball_q{int(q * 100)}"] = compute_pinball_loss(y_hold, predictions[q], q)
    metrics["mae_median"] = float(mean_absolute_error(y_hold, predictions[0.50]))

    # Coverage: 80% interval should contain truth ~80% of the time
    lower = predictions[0.10]
    upper = predictions[0.90]
    metrics["coverage_80"] = compute_coverage(y_hold, lower, upper)
    log.info("props.holdout_metrics", sport=sport, stat=stat, **metrics)

    # ── Wrap as a bundle for MLflow serialisation ─────────────────────────────
    bundle = QuantileBundle(quantile_models=quantile_models, quantiles=QUANTILES)

    run_id = log_model_run(
        run_name=run_name or f"{sport}_props_{stat}_{utc_now().strftime('%Y%m%d_%H%M')}",
        sport=sport,
        kind="props",
        target=stat,
        model=bundle,
        metrics=metrics,
        params=best_params,
        feature_names=feature_names,
        training_range=(
            str(training_df["scheduled_utc"].min()),
            str(training_df["scheduled_utc"].max()),
        ),
        model_framework="sklearn",
    )
    return run_id, metrics


class QuantileBundle:
    """Wraps N LightGBM quantile models into a single scikit-learn-compatible interface."""

    def __init__(
        self, quantile_models: dict[float, lgb.LGBMRegressor], quantiles: list[float]
    ) -> None:
        self.quantile_models = quantile_models
        self.quantiles = quantiles

    def predict(self, X: np.ndarray) -> dict[float, np.ndarray]:
        return {q: m.predict(X) for q, m in self.quantile_models.items()}

    def predict_row(self, x: np.ndarray) -> dict[str, float]:
        """Return quantile predictions for a single sample."""
        row = x.reshape(1, -1)
        return {str(q): float(m.predict(row)[0]) for q, m in self.quantile_models.items()}

    def implied_over_probability(self, x: np.ndarray, line: float) -> float:
        """Interpolate P(stat > line) from the quantile CDF."""
        preds = self.predict_row(x)
        sorted_q = sorted((float(k), v) for k, v in preds.items())
        qs = [q for q, _ in sorted_q]
        vals = [v for _, v in sorted_q]

        # If line is outside the quantile range, clamp
        if line <= vals[0]:
            return 1.0 - qs[0]
        if line >= vals[-1]:
            return 1.0 - qs[-1]

        # Linear interpolation between bracketing quantiles
        for i in range(len(vals) - 1):
            if vals[i] <= line <= vals[i + 1]:
                t = (line - vals[i]) / (vals[i + 1] - vals[i] + 1e-9)
                interpolated_q = qs[i] + t * (qs[i + 1] - qs[i])
                return float(1.0 - interpolated_q)

        return 0.5  # fallback
