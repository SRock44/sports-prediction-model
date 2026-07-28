"""Binary classifier for anytime-TD-scorer props.

Doesn't fit train_props.py's QuantileBundle machinery at all - "will this
player score any TD this game" is a probability-of-event question, not a
"how many yards" quantile-regression question. Reuses
src.cli._load_props_training_data / _nfl_stat_sql("ANYTIME_TD") for data
loading (rolling mean of a 0/1 target is a legitimate "TD rate over last N
games" feature, so the same rolling-feature pipeline the other props use
applies unchanged) - only the model type and evaluation differ.

Logs via model_framework="lightgbm" (mlflow.lightgbm.log_model, a native
flavor) rather than wrapping in a new custom class - avoids the exact
skops_trusted_types gap that silently broke every prop-model save for months
before it was caught (see src/models/registry.py's log_model_run docstring).
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.core.logging import get_logger
from src.core.time import utc_now
from src.features.common import exponential_decay_weight
from src.models.registry import log_model_run
from src.models.train_winner import _lgb_gpu_available

log = get_logger(__name__)

# Same recency-decay rationale as train_props.py: player-specific props decay
# faster than team-level signal.
_LAMBDA = {"nfl": 0.35}

_LGB_DEVICE = "gpu" if _lgb_gpu_available() else "cpu"
# Same thermal-safety cap as train_winner.py/train_props.py - see those
# modules' comments for the 2026-07-10 incident this guards against.
_LGB_N_JOBS = 4


def train_anytime_td_model(
    sport: str,
    training_df: pd.DataFrame,
    feature_names: list[str],
    holdout_df: pd.DataFrame,
    n_optuna_trials: int = 30,
    run_name: str | None = None,
) -> tuple[str, dict[str, float]]:
    """Train a binary anytime-TD classifier. Returns (mlflow_run_id, metrics)."""
    lam = _LAMBDA.get(sport, 0.35)

    training_df = training_df.sort_values("scheduled_utc").reset_index(drop=True)

    X_all = training_df[feature_names].values.astype(np.float32)
    y_all = training_df["target"].values.astype(int)
    X_hold = holdout_df[feature_names].values.astype(np.float32)
    y_hold = holdout_df["target"].values.astype(int)

    now = utc_now()

    def weights(df: pd.DataFrame) -> np.ndarray:
        days = (now - pd.to_datetime(df["scheduled_utc"])).dt.total_seconds() / 86400
        return np.array([exponential_decay_weight(d, lam) for d in days], dtype=np.float32)

    w_all = weights(training_df)

    # Last 10% chronologically held out from training as an internal
    # validation split for Optuna (holdout_df itself - a genuinely separate
    # season - is reserved for the final reported metrics only).
    split_idx = int(len(training_df) * 0.9)
    X_tr, y_tr, w_tr = X_all[:split_idx], y_all[:split_idx], w_all[:split_idx]
    X_val, y_val = X_all[split_idx:], y_all[split_idx:]

    def objective(trial: optuna.Trial) -> float:
        # dict[str, Any] rather than the inferred dict[str, float]: these are a
        # mix of ints and floats, and LGBMClassifier's typed kwargs reject a
        # float-valued **splat for its int/str-typed parameters.
        params: dict[str, Any] = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }
        m = lgb.LGBMClassifier(
            objective="binary",
            device=_LGB_DEVICE,
            verbose=-1,
            n_jobs=_LGB_N_JOBS,
            random_state=42,
            **params,
        )
        m.fit(X_tr, y_tr, sample_weight=w_tr)
        proba = np.asarray(m.predict_proba(X_val))[:, 1]
        return float(log_loss(y_val, proba, labels=[0, 1]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_optuna_trials, timeout=120)
    best_params = study.best_params
    log.info("optuna.anytime_td.best", sport=sport, params=best_params)

    # Final fit on all training data, evaluated on the real (separate-season) holdout.
    clf = lgb.LGBMClassifier(
        objective="binary",
        device=_LGB_DEVICE,
        verbose=-1,
        n_jobs=_LGB_N_JOBS,
        random_state=42,
        **best_params,
    )
    clf.fit(X_all, y_all, sample_weight=w_all)
    proba_hold = np.asarray(clf.predict_proba(X_hold))[:, 1]

    metrics = {
        "logloss": float(log_loss(y_hold, proba_hold, labels=[0, 1])),
        "brier": float(brier_score_loss(y_hold, proba_hold)),
        "accuracy": float(((proba_hold > 0.5).astype(int) == y_hold).mean()),
        "base_rate": float(y_hold.mean()),
        "n_samples": float(len(y_hold)),
    }
    log.info("anytime_td.holdout_metrics", sport=sport, **metrics)

    run_id = log_model_run(
        run_name=run_name or f"{sport}_props_ANYTIME_TD_{utc_now().strftime('%Y%m%d_%H%M')}",
        sport=sport,
        kind="props",
        target="ANYTIME_TD",
        model=clf,
        metrics=metrics,
        params=best_params,
        feature_names=feature_names,
        training_range=(
            str(training_df["scheduled_utc"].min()),
            str(training_df["scheduled_utc"].max()),
        ),
        model_framework="lightgbm",
    )
    return run_id, metrics
