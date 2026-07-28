"""Soccer winner-model training: a season-aligned-holdout + Optuna XGBoost approach,
refactored into an importable, schedulable form for a weekly retrain task.

Deliberately NOT merged into train_winner.py's StackedEnsemble/train_winner_model -
that pipeline is hardcoded to a binary "home_won" StackedEnsemble shape. Soccer's is a
single 3-way multi:softprob XGBClassifier with its own feature set. class_probabilities
(not the target string) is what signals a 3-way target to readers - target stays
"home_won", the universal cross-sport winner-target convention used elsewhere in this
codebase.

Class order is fixed at [0=home, 1=draw, 2=away] end to end - the training data's `y`
assignment and this module's holdout metrics must agree on it (as must any downstream
consumer of the predicted class_probabilities).

Building the training DataFrame itself (one row per completed game, pulled from a live
database) isn't included here — see this repo's README for why. This module picks up
from an already-assembled `training_df`, the same contract train_winner_model and
train_props_model use for every other sport.
"""

from __future__ import annotations

import time

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

from src.core.logging import get_logger
from src.core.time import utc_now
from src.features.common import exponential_decay_weight
from src.models.eval.metrics import compute_all_winner_metrics_multiclass
from src.models.registry import log_model_run

log = get_logger(__name__)

TARGET = "home_won"
_N_JOBS = 4


def _build_cv_folds(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, n_folds: int = 5
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    fold_size = len(X) // (n_folds + 1)
    folds = []
    for i in range(1, n_folds + 1):
        tr_end, val_end = i * fold_size, (i + 1) * fold_size
        if val_end > len(X):
            break
        folds.append(
            (
                X[:tr_end],
                y[:tr_end],
                w[:tr_end],
                X[tr_end:val_end],
                y[tr_end:val_end],
                w[tr_end:val_end],
            )
        )
    return folds


def train_soccer_winner_model(
    training_df: pd.DataFrame,
    n_optuna_trials: int = 100,
    run_name: str | None = None,
) -> tuple[str, dict[str, float], list[str]]:
    """Train + log (does not promote) a soccer winner challenger. Season-aligned holdout
    (most recent season vs. everything before it), not an arbitrary row-count split.
    Returns (run_id, metrics, feature_names); promotion is the caller's call.

    Expects training_df with one row per completed game: build_matchup_features'
    output columns plus `game_id`, `game_date`, `season`, `competition_code`, and `y`
    (0=home win, 1=draw, 2=away win).
    """
    df = training_df.sort_values("game_date").reset_index(drop=True)
    df = pd.get_dummies(df, columns=["competition_code"], prefix="competition")
    non_feature_cols = {"game_id", "game_date", "y", "season"}
    all_features = [c for c in df.columns if c not in non_feature_cols]
    feat_stds = df[all_features].astype(float).std()
    feature_names = feat_stds[feat_stds > 1e-6].index.tolist()

    max_season = df["season"].max()
    train_df = df[df["season"] < max_season].copy()
    holdout_df = df[df["season"] == max_season].copy()
    if train_df.empty or holdout_df.empty:
        cutoff = int(len(df) * 0.85)
        train_df = df.iloc[:cutoff].copy()
        holdout_df = df.iloc[cutoff:].copy()

    X_train = train_df[feature_names].values.astype(np.float32)
    y_train = train_df["y"].values.astype(int)
    X_hold = holdout_df[feature_names].values.astype(np.float32)
    y_hold = holdout_df["y"].values.astype(int)

    anchor = pd.to_datetime(train_df["game_date"].max())
    days_ago_train = (anchor - pd.to_datetime(train_df["game_date"])).dt.total_seconds() / 86400

    def make_weights(lam: float) -> np.ndarray:
        return np.array(
            [exponential_decay_weight(d, lam) for d in days_ago_train], dtype=np.float32
        )

    def objective(trial: optuna.Trial) -> float:
        lam = trial.suggest_float("lambda_decay", 0.05, 1.0, log=True)
        w = make_weights(lam)
        cv_folds = _build_cv_folds(X_train, y_train, w)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 1.5),
        }
        losses = []
        for Xtr, ytr, wtr, Xval, yval, wval in cv_folds:
            clf = xgb.XGBClassifier(
                **params,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=_N_JOBS,
                random_state=42,
            )
            clf.fit(Xtr, ytr, sample_weight=wtr, verbose=False)
            proba = clf.predict_proba(Xval)
            losses.append(float(log_loss(yval, proba, labels=[0, 1, 2], sample_weight=wval)))
        return float(np.mean(losses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    t0 = time.monotonic()
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=False)
    log.info(
        "soccer_train.search_done",
        seconds=round(time.monotonic() - t0, 0),
        best_cv_logloss=round(study.best_value, 5),
    )

    best_params = {k: v for k, v in study.best_params.items() if k != "lambda_decay"}
    best_lambda = study.best_params["lambda_decay"]
    w_train_final = make_weights(best_lambda)

    best = xgb.XGBClassifier(
        **best_params,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=_N_JOBS,
        random_state=42,
    )
    best.fit(X_train, y_train, sample_weight=w_train_final, verbose=False)

    proba_hold = best.predict_proba(X_hold)
    metrics = compute_all_winner_metrics_multiclass(y_hold, proba_hold)
    mask = y_hold != 1
    metrics["moneyline_accuracy"] = float(
        np.mean(np.where(proba_hold[mask, 0] > proba_hold[mask, 2], 0, 2) == y_hold[mask])
    )
    metrics["cv_logloss"] = study.best_value
    metrics["lambda_decay"] = best_lambda

    flat_params = {f"xgb_{k}": v for k, v in best_params.items()}
    run_id = log_model_run(
        run_name=run_name or f"soccer_winner_challenger_{utc_now().strftime('%Y%m%d_%H%M')}",
        sport="soccer",
        kind="winner",
        target=TARGET,
        model=best,
        metrics=metrics,
        params=flat_params,
        feature_names=feature_names,
        training_range=(str(train_df["game_date"].min()), str(train_df["game_date"].max())),
        model_framework="xgboost",
    )
    return run_id, metrics, feature_names
