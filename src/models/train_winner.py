"""XGBoost + LightGBM ensemble winner pipeline with manual isotonic calibration.

Key properties:
- Walk-forward 5-fold chronological CV in Optuna objective (no shuffling ever)
- 200 Optuna trials, MedianPruner, no timeout — full search
- XGBoost with n_estimators searched by Optuna; final fit uses early stopping to
  refine optimal tree count against calibration set
- Optional LightGBM soft ensemble (averaged 60/40 with XGBoost) if installed
- Manual isotonic regression calibration on prefit ensemble (no cv= leakage)
- Exponential recency sample weights anchored to training set end date
- Champion/challenger promotion gate
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.core.logging import get_logger
from src.core.time import utc_now
from src.features.common import exponential_decay_weight
from src.models.eval.metrics import compute_all_winner_metrics
from src.models.registry import log_model_run

log = get_logger(__name__)


# MLflow's tracking API stores every logged param as a string; a champion retrain
# reloads them via fixed_params and passes them straight into the base-learner
# constructors. XGBoost/LightGBM tolerate numeric strings, but CatBoost's C++
# param parser doesn't and raises CatBoostError. Coerce back to real int/float
# using each key's known type on the way in.
def _coerce_fixed_params(params: dict, int_keys: frozenset[str]) -> dict:
    coerced = {}
    for k, v in params.items():
        if not isinstance(v, str):
            coerced[k] = v
        elif k in int_keys:
            coerced[k] = int(v)
        else:
            coerced[k] = float(v)
    return coerced


# Promotion margins. A season with only a few hundred holdout games has a much
# larger standard error on log-loss than a daily-cadence sport with thousands -
# using the same tight margin for both means a small-sample sport's champion
# flips on pure noise. Small-sample sports get a wider margin, still under one
# standard error but at least asking for a real signal.
_MIN_LOGLOSS_IMPROVEMENT = 0.005
_MIN_LOGLOSS_IMPROVEMENT_SMALL_SAMPLE = 0.02
# nascar: per-driver-per-race win rate is a rare event (~1/37) - a small logloss
# delta between challenger and champion is more likely to be noise here than for
# a typical team-sport binary target, same reasoning a short season already
# applies for a different reason (small sample size).
_SMALL_SAMPLE_SPORTS = frozenset({"nfl", "nascar"})

_LAMBDA = {
    "nba": 0.30,
    "mlb": 0.50,
    # NFL: deliberately the slowest decay of any sport (default fallthrough
    # below is 0.25) - a 17-game season can't afford to aggressively discount
    # early-season games the way MLB discounts April with 162 games to lean on.
    "nfl": 0.18,
    "nhl": 0.30,  # same daily-cadence season shape as nba
    "nascar": 0.20,  # ~36 races/season - denser than NFL's 17, sparser than NBA's 82
}

try:
    import lightgbm as lgb

    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import catboost as cb

    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False


# Checks for a real, functioning NVIDIA driver directly rather than trusting
# XGBoost's device="cuda" fit() to raise when unavailable - newer XGBoost
# versions silently fall back to CPU with just a warning ("Falling back to
# prediction using DMatrix due to mismatched devices") instead of raising, so
# a fit()-based probe can return a false positive on GPU-less hosts. That false
# positive is harmless for XGBoost/LightGBM (both have their own silent CPU
# fallback) but fatal for CatBoost, which has no such fallback and hard-crashes
# with "CUDA driver version is insufficient" when task_type is set to "GPU"
# without a real driver behind it.
def _cuda_available() -> bool:
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5, check=False)
        return result.returncode == 0
    except Exception:
        return False


# LightGBM's "gpu" device needs its own OpenCL runtime/ICD — a working XGBoost
# CUDA device does NOT imply LightGBM GPU works, so this must be probed
# separately rather than inferred from _cuda_available().
def _lgb_gpu_available() -> bool:
    if not _HAS_LGB:
        return False
    try:
        import numpy as _np

        _probe = lgb.LGBMClassifier(device="gpu", n_estimators=1, verbose=-1)
        _probe.fit(_np.zeros((4, 2)), [0, 1, 0, 1])
        return True
    except Exception:
        return False


_GPU = _cuda_available()
_XGB_DEVICE = "cuda" if _GPU else "cpu"
# LightGBM uses "gpu" (not "cuda") as the device string
_LGB_DEVICE = "gpu" if _lgb_gpu_available() else "cpu"

# LightGBM defaults n_jobs=-1 (every logical core) regardless of device="gpu" -
# GPU only offloads the histogram kernel, everything else still spins up
# LightGBM's full CPU thread pool. Running challenger/props training for
# multiple sports concurrently means uncapped processes can oversubscribe every
# core at once - this caused a real thermal incident on a thermally-marginal
# host. Cap it so concurrent runs can't do that.
#
# Env-configurable rather than hardcoded: this value needs to differ between
# deployment targets sharing one codebase (a big dedicated training box vs. a
# small/thermally-marginal dev machine) - a literal here is either right for
# one or right for the other, never both. Default (16) is a reasonable ceiling
# for a dedicated multi-core training host; override down for smaller hardware.
_LGB_N_JOBS = int(os.environ.get("LGB_N_JOBS", "16"))

# CatBoost's CPU mode (task_type="CPU", used whenever _GPU is False) defaults
# thread_count=-1 (every core) same as LightGBM above - same thermal risk, same
# env-configurable approach, same default as _LGB_N_JOBS above.
_CAT_N_JOBS = int(os.environ.get("CAT_N_JOBS", "16"))


def _optuna_callback(model_tag: str, n_trials: int, start_time: float):
    """Optuna callback: prints one line per trial."""

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        elapsed = time.time() - start_time
        is_best = trial.value == study.best_value
        marker = " ◀ best" if is_best else ""
        print(
            f"  [{model_tag}] trial {trial.number + 1:>4}/{n_trials}"
            f"  loss={trial.value:.5f}"
            f"  best={study.best_value:.5f}"
            f"  elapsed={elapsed:.0f}s"
            f"{marker}",
            flush=True,
        )

    return _cb


class _XGBProgressCallback(xgb.callback.TrainingCallback):
    """Prints XGBoost round progress every N rounds during early-stopping fit."""

    def __init__(self, every: int = 100, tag: str = "XGB") -> None:
        self.every = every
        self.tag = tag
        self._start = time.time()

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        if (epoch + 1) % self.every == 0:
            val_loss = None
            for ds_metrics in evals_log.values():
                if "logloss" in ds_metrics:
                    val_loss = ds_metrics["logloss"][-1]
            elapsed = time.time() - self._start
            print(
                f"  [{self.tag}] round {epoch + 1:>5}  val_loss={val_loss:.5f}"
                if val_loss
                else f"  [{self.tag}] round {epoch + 1:>5}",
                f"  elapsed={elapsed:.0f}s",
                flush=True,
            )
        return False


class StackedEnsemble:
    """XGBoost + LightGBM + CatBoost base learners with a logistic meta-learner.

    Training flow (all on chronologically non-overlapping splits):
    1. Base learners trained on train_part
    2. Meta-learner (LogisticRegression) trained on base-learner OOF probs from calib_part
    3. Isotonic calibration on meta-learner output from calib_part

    Falls back to equal-weight average when only XGBoost is available.
    Implements sklearn predict_proba interface for MLflow pickle serialization.
    """

    def __init__(
        self,
        xgb_clf: xgb.XGBClassifier,
        iso: IsotonicRegression,
        lgb_clf=None,
        cat_clf=None,
        meta=None,
        proba_clip: float = 1e-6,
        iso_blend_weight: float = 0.7,
    ) -> None:
        self.xgb_clf = xgb_clf
        self.lgb_clf = lgb_clf
        self.cat_clf = cat_clf
        self.meta = meta  # fitted LogisticRegression or None
        self.iso = iso
        # Weight on the isotonic-calibrated probability vs. the raw ensemble
        # score in the final blend (see predict_proba). 1.0 = pure isotonic.
        self.iso_blend_weight = iso_blend_weight
        # How far from 0/1 predict_proba is clamped. Default (1e-6) barely
        # avoids a log(0) crash but still allows near-certain predictions that
        # are catastrophic when wrong under log-loss. A small-sample sport's
        # isotonic regression is prone to outputting a literal 1.000 for a
        # holdout game that then loses - one such game can contribute a
        # meaningful fraction of the entire holdout's total log-loss. Isotonic
        # regression saturates like this with small calibration sets: it can
        # map some input range to a hard 0/1 if the calibration data happened
        # to be unanimous there, with no cushion for real-world uncertainty a
        # sports outcome always has. A wider clip (e.g. 0.03) is passed for
        # small-sample sports at construction time; the daily-cadence sports
        # keep the original 1e-6 default.
        self.proba_clip = proba_clip
        self.classes_ = np.array([0, 1])

    def _base_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        cols = [self.xgb_clf.predict_proba(X)[:, 1]]
        if self.lgb_clf is not None:
            cols.append(self.lgb_clf.predict_proba(X)[:, 1])
        if self.cat_clf is not None:
            cols.append(self.cat_clf.predict_proba(X)[:, 1])
        return np.column_stack(cols)

    def _raw_proba(self, X: np.ndarray) -> np.ndarray:
        mat = self._base_proba_matrix(X)
        if self.meta is not None:
            return self.meta.predict_proba(mat)[:, 1]
        return mat.mean(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw_proba(X)
        # getattr, not self.<attr>: models pickled before these attributes
        # existed must not crash scoring on every game just because an older
        # artifact predates a field added since.
        clip = getattr(self, "proba_clip", 1e-6)
        cal_iso = self.iso.predict(raw)
        # Isotonic regression (PAV) fits a piecewise-*constant* step function.
        # With a modest calibration set, some steps end up wide and flat -
        # every game whose raw score falls in that band (often most of a
        # day's slate, since raw scores cluster narrowly too) then displays
        # identical confidence, which looks like a bug even though the
        # calibration itself is technically correct. Blending back in a slice
        # of the raw, continuous ensemble score restores per-game variance
        # inside those flat segments while the isotonic term still anchors the
        # output near its true calibrated rate.
        iso_weight = getattr(self, "iso_blend_weight", 0.7)
        blended = iso_weight * cal_iso + (1.0 - iso_weight) * raw
        cal = np.clip(blended, clip, 1.0 - clip)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _build_cv_folds(X: np.ndarray, y: np.ndarray, w: np.ndarray, n_folds: int = 5) -> list[tuple]:
    """Build expanding-window time-series CV folds (data must be sorted by date)."""
    n = len(X)
    fold_size = n // (n_folds + 1)
    folds = []
    for i in range(1, n_folds + 1):
        tr_end = i * fold_size
        val_end = (i + 1) * fold_size
        if val_end > n:
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


def train_winner_model(
    sport: str,
    training_df: pd.DataFrame,
    feature_names: list[str],
    holdout_df: pd.DataFrame,
    n_optuna_trials: int = 500,
    run_name: str | None = None,
    wide_search: bool = False,
    fixed_params: dict | None = None,
) -> tuple[str, dict[str, float]]:
    """Train calibrated ensemble model. Returns (mlflow_run_id, metrics_dict)."""
    _run_t0 = time.time()
    lam = _LAMBDA.get(sport, 0.25)

    # Ensure all expected feature columns exist — new features added to the config
    # won't be present in older matchup_features rows. Fill with 0.0 (neutral).
    # The near-constant filter below will drop them if they're still nearly all 0.
    training_df = training_df.reindex(
        columns=list(training_df.columns)
        + [f for f in feature_names if f not in training_df.columns],
        fill_value=0.0,
    )
    holdout_df = holdout_df.reindex(
        columns=list(holdout_df.columns)
        + [f for f in feature_names if f not in holdout_df.columns],
        fill_value=0.0,
    )

    # Drop near-constant features — catches empty odds/lineup tables where every
    # row has the same hardcoded default, which adds noise rather than signal.
    feat_stds = training_df[feature_names].std()
    live_features = feat_stds[feat_stds > 1e-6].index.tolist()
    dropped = [f for f in feature_names if f not in live_features]
    if dropped:
        print(
            f"[Prep] Dropping {len(dropped)} near-constant features: "
            f"{', '.join(dropped[:8])}{'...' if len(dropped) > 8 else ''}"
        )
        feature_names = live_features

    training_df = training_df.sort_values("game_date").reset_index(drop=True)

    # Last 20% held for calibration; more data = better isotonic fit, less overfitting
    n = len(training_df)
    calib_start = int(n * 0.80)
    train_part = training_df.iloc[:calib_start]
    calib_part = training_df.iloc[calib_start:]

    X_train = train_part[feature_names].values.astype(np.float32)
    y_train = train_part["y"].values.astype(int)
    X_calib = calib_part[feature_names].values.astype(np.float32)
    y_calib = calib_part["y"].values.astype(int)
    X_hold = holdout_df[feature_names].values.astype(np.float32)
    y_hold = holdout_df["y"].values.astype(int)

    anchor = pd.to_datetime(training_df["game_date"].max())

    def make_weights(df: pd.DataFrame) -> np.ndarray:
        days_ago = (anchor - pd.to_datetime(df["game_date"])).dt.total_seconds() / 86400
        return np.array([exponential_decay_weight(d, lam) for d in days_ago], dtype=np.float32)

    w_train = make_weights(train_part)
    w_calib = make_weights(calib_part)

    cv_folds = _build_cv_folds(X_train, y_train, w_train, n_folds=5)
    log.info(
        "training.start",
        sport=sport,
        n_train=len(train_part),
        n_calib=len(calib_part),
        n_hold=len(holdout_df),
        n_cv_folds=len(cv_folds),
        n_trials=n_optuna_trials,
    )

    print(f"\n{'=' * 65}")
    print(f"  Training {sport.upper()} winner model")
    print(f"  Train: {len(train_part):,}  Calib: {len(calib_part):,}  Holdout: {len(holdout_df):,}")
    print(
        f"  CV folds: {len(cv_folds)}  Optuna trials: {n_optuna_trials}  Device: {_XGB_DEVICE.upper()}"
    )
    print(f"{'=' * 65}\n")

    # ── XGBoost Optuna search ─────────────────────────────────────────────────
    # Constrained bounds (default) prevent the Run-4 failure where Optuna found
    # max_depth=10 / n_estimators=4115 that overfit CV but collapsed on holdout.
    # wide_search=True is reserved for monthly deep-exploration runs only.
    #
    # is_small_sample tightens further, on top of the narrow defaults: a
    # small-sample sport's training set is a fraction of the daily-cadence
    # sports' size, and an untightened search on that little data can produce
    # a holdout log-loss worse than the naive always-0.5 baseline despite good
    # accuracy - a textbook overfit/miscalibration signature.
    is_small_sample = sport in _SMALL_SAMPLE_SPORTS
    _xgb_n_est_hi = 5000 if wide_search else (600 if is_small_sample else 2000)
    _xgb_depth_hi = 10 if wide_search else (5 if is_small_sample else 8)
    _xgb_lr_lo = 0.001 if wide_search else 0.005
    _xgb_min_child_lo = 5 if is_small_sample else 1

    def xgb_objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, _xgb_n_est_hi),
            "max_depth": trial.suggest_int("max_depth", 3, _xgb_depth_hi),
            "learning_rate": trial.suggest_float("learning_rate", _xgb_lr_lo, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", _xgb_min_child_lo, 30),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1.0 if is_small_sample else 0.1, 100.0, log=True
            ),
            "gamma": trial.suggest_float("gamma", 0.0, 1.5),
        }
        fold_losses = []
        for X_tr, y_tr, w_tr, X_val, y_val, w_val in cv_folds:
            clf = xgb.XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device=_XGB_DEVICE,
                random_state=42,
            )
            clf.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
            proba = clf.predict_proba(X_val)[:, 1]
            fold_losses.append(float(log_loss(y_val, proba, sample_weight=w_val)))
        return float(np.mean(fold_losses))

    # ── Load saved params or run Optuna search ───────────────────────────────
    import json

    _params_path = f"reports/{sport}_winner_xgb_params.json"
    os.makedirs("reports", exist_ok=True)

    if fixed_params is not None:
        # Use caller-supplied params directly (e.g. champion retrain) — no search, no file write
        best_xgb_params = _coerce_fixed_params(
            {
                k[4:]: v
                for k, v in fixed_params.items()
                if k.startswith("xgb_") and k != "xgb_best_n_trees"
            },
            int_keys=frozenset({"n_estimators", "max_depth", "min_child_weight"}),
        )
        print(f"[XGBoost] Using fixed params (champion retrain): {best_xgb_params}\n")
        log.info("optuna.xgb_fixed", sport=sport, params=best_xgb_params)
    elif n_optuna_trials == 0 and os.path.exists(_params_path):
        with open(_params_path) as _f:
            best_xgb_params = json.load(_f)
        print(f"[XGBoost] Skipping search — loaded saved params from {_params_path}")
        print(f"[XGBoost] Params: {best_xgb_params}\n")
        log.info("optuna.xgb_loaded", sport=sport, params=best_xgb_params)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
        xgb_study = optuna.create_study(direction="minimize", pruner=pruner)
        _actual_trials = n_optuna_trials if n_optuna_trials > 0 else 50
        print(
            f"[XGBoost] Starting Optuna search — {_actual_trials} trials, 5-fold walk-forward CV\n"
        )
        _xgb_t0 = time.time()
        xgb_study.optimize(
            xgb_objective,
            n_trials=_actual_trials,
            show_progress_bar=False,
            callbacks=[_optuna_callback("XGB", _actual_trials, _xgb_t0)],
        )
        best_xgb_params = xgb_study.best_params
        print(f"\n[XGBoost] Search done in {time.time() - _xgb_t0:.0f}s")
        print(f"[XGBoost] Best trial: loss={xgb_study.best_value:.5f}  params={best_xgb_params}\n")
        log.info(
            "optuna.xgb_best",
            sport=sport,
            params=best_xgb_params,
            cv_loss=xgb_study.best_value,
            device=_XGB_DEVICE,
        )
        # Save for future fast-resume runs
        with open(_params_path, "w") as _f:
            json.dump(best_xgb_params, _f, indent=2)
        print(f"[XGBoost] Saved best params to {_params_path}\n")

    # Refine n_estimators with early stopping against calib set
    # Note: callbacks omitted — XGBoost 3.x routes constructor callbacks through
    # inner_f which re-passes them to fit() where they are no longer accepted.
    _finder = xgb.XGBClassifier(
        **{k: v for k, v in best_xgb_params.items() if k != "n_estimators"},
        n_estimators=5000,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=_XGB_DEVICE,
        early_stopping_rounds=50,
        random_state=42,
    )
    print("[XGBoost] Early-stopping fit (max 5000 trees, stop after 50 no-improve) ...")
    _es_t0 = time.time()
    _finder.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_calib, y_calib)],
        sample_weight_eval_set=[w_calib],
        verbose=False,
    )
    best_n_trees = _finder.best_iteration or best_xgb_params["n_estimators"]
    print(f"[XGBoost] Early stop: best_n_trees={best_n_trees}  ({time.time() - _es_t0:.0f}s)\n")
    log.info("xgb.early_stopping", sport=sport, best_n_trees=best_n_trees)

    # Final XGBoost on train_part with optimal tree count
    best_xgb = xgb.XGBClassifier(
        **{k: v for k, v in best_xgb_params.items() if k != "n_estimators"},
        n_estimators=best_n_trees,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=_XGB_DEVICE,
        random_state=42,
    )
    print(f"[XGBoost] Final fit with {best_n_trees} trees ...")
    _f_t0 = time.time()
    best_xgb.fit(X_train, y_train, sample_weight=w_train, verbose=False)
    print(f"[XGBoost] Final fit done ({time.time() - _f_t0:.0f}s)\n")

    # ── Optional LightGBM search ──────────────────────────────────────────────
    lgb_clf = None
    best_lgb_params: dict = {}

    if _HAS_LGB:
        _lgb_leaves_hi = 200 if wide_search else (31 if is_small_sample else 80)
        _lgb_n_est_hi = 3000 if wide_search else (600 if is_small_sample else 1500)
        _lgb_min_child_lo = 20 if is_small_sample else 5

        def lgb_objective(trial: optuna.Trial) -> float:
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 20, _lgb_leaves_hi),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, _lgb_n_est_hi),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", _lgb_min_child_lo, 60),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-2 if is_small_sample else 1e-8, 10.0, log=True
                ),
            }
            fold_losses = []
            for X_tr, y_tr, w_tr, X_val, y_val, w_val in cv_folds:
                clf = lgb.LGBMClassifier(
                    **params,
                    objective="binary",
                    device=_LGB_DEVICE,
                    n_jobs=_LGB_N_JOBS,
                    random_state=42,
                    verbose=-1,
                )
                clf.fit(
                    X_tr,
                    y_tr,
                    sample_weight=w_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)],
                )
                proba = clf.predict_proba(X_val)[:, 1]
                fold_losses.append(float(log_loss(y_val, proba, sample_weight=w_val)))
            return float(np.mean(fold_losses))

        _lgb_params_path = f"reports/{sport}_winner_lgb_params.json"
        _lgb_actual_trials = n_optuna_trials if n_optuna_trials > 0 else 0

        if fixed_params is not None:
            best_lgb_params = _coerce_fixed_params(
                {
                    k[4:]: v
                    for k, v in fixed_params.items()
                    if k.startswith("lgb_") and k != "lgb_best_n_trees"
                },
                int_keys=frozenset({"num_leaves", "n_estimators", "min_child_samples"}),
            )
            print(f"[LightGBM] Using fixed params (champion retrain): {best_lgb_params}\n")
            log.info("optuna.lgb_fixed", sport=sport, params=best_lgb_params)
        elif _lgb_actual_trials == 0 and os.path.exists(_lgb_params_path):
            with open(_lgb_params_path) as _f:
                best_lgb_params = json.load(_f)
            print(f"[LightGBM] Skipping search — loaded saved params from {_lgb_params_path}")
            log.info("optuna.lgb_loaded", sport=sport, params=best_lgb_params)
        else:
            _lgb_actual_trials = _lgb_actual_trials or 200
            _lgb_pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
            lgb_study = optuna.create_study(direction="minimize", pruner=_lgb_pruner)
            print(
                f"[LightGBM] Starting Optuna search — {_lgb_actual_trials} trials, 5-fold walk-forward CV\n"
            )
            _lgb_t0 = time.time()
            lgb_study.optimize(
                lgb_objective,
                n_trials=_lgb_actual_trials,
                show_progress_bar=False,
                callbacks=[_optuna_callback("LGB", _lgb_actual_trials, _lgb_t0)],
            )
            best_lgb_params = lgb_study.best_params
            print(f"\n[LightGBM] Search done in {time.time() - _lgb_t0:.0f}s")
            print(
                f"[LightGBM] Best trial: loss={lgb_study.best_value:.5f}  params={best_lgb_params}\n"
            )
            log.info(
                "optuna.lgb_best", sport=sport, params=best_lgb_params, cv_loss=lgb_study.best_value
            )
            with open(_lgb_params_path, "w") as _f:
                json.dump(best_lgb_params, _f, indent=2)
            print(f"[LightGBM] Saved best params to {_lgb_params_path}\n")

        lgb_clf = lgb.LGBMClassifier(
            **{k: v for k, v in best_lgb_params.items() if k != "n_estimators"},
            n_estimators=5000,
            objective="binary",
            device=_LGB_DEVICE,
            n_jobs=_LGB_N_JOBS,
            random_state=42,
            verbose=-1,
        )
        print("[LightGBM] Final fit with early stopping (max 5000 trees) ...")
        _lgbf_t0 = time.time()
        lgb_clf.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_calib, y_calib)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(200),
            ],
        )
        lgb_best_iter = getattr(lgb_clf, "best_iteration_", "n/a")
        print(
            f"[LightGBM] Final fit done — best_n_trees={lgb_best_iter}  ({time.time() - _lgbf_t0:.0f}s)\n"
        )
        log.info("lgb.best_iteration", sport=sport, n_trees=lgb_best_iter)

    # ── CatBoost base learner (optional) ─────────────────────────────────────
    cat_clf = None
    best_cat_params: dict = {}

    if _HAS_CAT:
        _cat_params_path = f"reports/{sport}_winner_cat_params.json"
        _cat_iter_hi = 500 if is_small_sample else 1500
        _cat_depth_hi = 5 if is_small_sample else 8

        def cat_objective(trial: optuna.Trial) -> float:
            params = {
                "iterations": trial.suggest_int("iterations", 200, _cat_iter_hi),
                "depth": trial.suggest_int("depth", 4, _cat_depth_hi),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", 3.0 if is_small_sample else 1.0, 20.0, log=True
                ),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            }
            fold_losses = []
            for X_tr, y_tr, w_tr, X_val, y_val, w_val in cv_folds:
                clf = cb.CatBoostClassifier(
                    **params,
                    loss_function="Logloss",
                    task_type="GPU" if _GPU else "CPU",
                    thread_count=-1 if _GPU else _CAT_N_JOBS,
                    random_seed=42,
                    verbose=0,
                )
                clf.fit(X_tr, y_tr, sample_weight=w_tr)
                proba = clf.predict_proba(X_val)[:, 1]
                fold_losses.append(float(log_loss(y_val, proba, sample_weight=w_val)))
            return float(np.mean(fold_losses))

        if fixed_params is not None:
            best_cat_params = _coerce_fixed_params(
                {k[4:]: v for k, v in fixed_params.items() if k.startswith("cat_")},
                int_keys=frozenset({"iterations", "depth"}),
            )
            print(f"[CatBoost] Using fixed params (champion retrain): {best_cat_params}\n")
        elif n_optuna_trials == 0 and os.path.exists(_cat_params_path):
            with open(_cat_params_path) as _f:
                best_cat_params = json.load(_f)
            print(f"[CatBoost] Skipping search — loaded saved params from {_cat_params_path}")
        else:
            _cat_actual_trials = n_optuna_trials if n_optuna_trials > 0 else 50
            _cat_pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
            cat_study = optuna.create_study(direction="minimize", pruner=_cat_pruner)
            print(
                f"[CatBoost] Starting Optuna search — {_cat_actual_trials} trials, 5-fold walk-forward CV\n"
            )
            _cat_t0 = time.time()
            cat_study.optimize(
                cat_objective,
                n_trials=_cat_actual_trials,
                show_progress_bar=False,
                callbacks=[_optuna_callback("CAT", _cat_actual_trials, _cat_t0)],
            )
            best_cat_params = cat_study.best_params
            print(f"\n[CatBoost] Search done in {time.time() - _cat_t0:.0f}s")
            print(
                f"[CatBoost] Best trial: loss={cat_study.best_value:.5f}  params={best_cat_params}\n"
            )
            log.info("optuna.cat_best", sport=sport, params=best_cat_params)
            with open(_cat_params_path, "w") as _f:
                json.dump(best_cat_params, _f, indent=2)
            print(f"[CatBoost] Saved best params to {_cat_params_path}\n")

        cat_clf = cb.CatBoostClassifier(
            **best_cat_params,
            loss_function="Logloss",
            task_type="GPU" if _GPU else "CPU",
            thread_count=-1 if _GPU else _CAT_N_JOBS,
            random_seed=42,
            verbose=0,
        )
        print("[CatBoost] Final fit ...")
        _cat_f0 = time.time()
        cat_clf.fit(X_train, y_train, sample_weight=w_train)
        print(f"[CatBoost] Final fit done ({time.time() - _cat_f0:.0f}s)\n")

    # ── Logistic meta-learner on calib-set OOF probs ─────────────────────────
    # Each base model predicts P(home_win) on calib_part; meta-learner learns
    # optimal weighting — replaces the fixed 60/40 average from prior versions.
    calib_cols = [best_xgb.predict_proba(X_calib)[:, 1]]
    if lgb_clf is not None:
        calib_cols.append(lgb_clf.predict_proba(X_calib)[:, 1])
    if cat_clf is not None:
        calib_cols.append(cat_clf.predict_proba(X_calib)[:, 1])
    X_meta = np.column_stack(calib_cols)

    meta_learner: LogisticRegression | None = None
    if X_meta.shape[1] > 1:
        meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        meta_learner.fit(X_meta, y_calib, sample_weight=w_calib)
        meta_raw = meta_learner.predict_proba(X_meta)[:, 1]
        n_base = X_meta.shape[1]
        weights_str = ", ".join(f"{c:.3f}" for c in meta_learner.coef_[0])
        print(f"[Meta] Logistic meta-learner trained on {n_base} base models")
        print(
            f"[Meta] Learned coefficients: [{weights_str}]  intercept={meta_learner.intercept_[0]:.3f}\n"
        )
        log.info(
            "meta.logistic_coefficients",
            sport=sport,
            n_base=n_base,
            coef=meta_learner.coef_[0].tolist(),
        )
        ensemble_raw = meta_raw
    else:
        ensemble_raw = X_meta[:, 0]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(ensemble_raw, y_calib, sample_weight=w_calib)

    calibrated = StackedEnsemble(
        xgb_clf=best_xgb,
        iso=iso,
        lgb_clf=lgb_clf,
        cat_clf=cat_clf,
        meta=meta_learner,
        # Small calibration sets make isotonic regression prone to saturating
        # at hard 0/1 for some input range - see StackedEnsemble.__init__'s
        # docstring for why small-sample sports need a wider clip.
        proba_clip=0.03 if is_small_sample else 1e-6,
    )

    # ── Evaluate on holdout ───────────────────────────────────────────────────
    _base_names = (
        ["xgb"]
        + (["lgb"] if lgb_clf is not None else [])
        + (["cat"] if cat_clf is not None else [])
    )
    _ensemble_tag = "+".join(_base_names)
    _meta_tag = "logistic" if meta_learner is not None else "mean"

    print(f"[Eval] Scoring holdout ({len(y_hold):,} games) ...")
    proba_hold = calibrated.predict_proba(X_hold)[:, 1]
    metrics = compute_all_winner_metrics(y_hold, proba_hold)
    print(f"\n{'=' * 65}")
    print(f"  {sport.upper()} Holdout Results")
    print(f"  Accuracy : {metrics.get('accuracy', 0):.4f}")
    print(f"  Log-loss : {metrics.get('logloss', 0):.5f}")
    print(f"  Brier    : {metrics.get('brier', 0):.5f}")
    print(f"  ECE      : {metrics.get('ece', 0):.5f}")
    print(f"  N samples: {metrics.get('n_samples', len(y_hold)):,}")
    print(f"  Ensemble : {_ensemble_tag}  Meta: {_meta_tag}")
    print(f"{'=' * 65}\n")
    log.info("winner.holdout_metrics", sport=sport, **metrics)

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    flat_params: dict = {}
    for k, v in best_xgb_params.items():
        flat_params[f"xgb_{k}"] = v
    flat_params["xgb_best_n_trees"] = best_n_trees
    if lgb_clf is not None:
        for k, v in best_lgb_params.items():
            flat_params[f"lgb_{k}"] = v
    if cat_clf is not None:
        for k, v in best_cat_params.items():
            flat_params[f"cat_{k}"] = v
    flat_params["ensemble"] = _ensemble_tag
    flat_params["meta_learner"] = _meta_tag

    run_id = log_model_run(
        run_name=run_name or f"{sport}_winner_{utc_now().strftime('%Y%m%d_%H%M')}",
        sport=sport,
        kind="winner",
        target="home_won",
        model=calibrated,
        metrics=metrics,
        params=flat_params,
        feature_names=feature_names,
        training_range=(
            str(training_df["game_date"].min()),
            str(training_df["game_date"].max()),
        ),
        model_framework="sklearn",
    )

    total_elapsed = time.time() - _run_t0
    finished_et = datetime.now(ZoneInfo("America/New_York"))
    print(
        f"[Run] Total elapsed: {total_elapsed / 60:.1f} min "
        f"({total_elapsed:.0f}s) — finished {finished_et.strftime('%Y-%m-%d %I:%M:%S %p %Z')}\n"
    )
    log.info(
        "winner.run_complete",
        sport=sport,
        total_elapsed_seconds=round(total_elapsed, 1),
        finished_et=finished_et.isoformat(),
    )

    return run_id, metrics


def should_promote(
    challenger_metrics: dict[str, float],
    champion_metrics: dict[str, float],
    min_logloss_improvement: float | None = None,
    max_ece_increase: float = 0.02,
    max_brier_increase: float = 0.005,
    sport: str | None = None,
) -> tuple[bool, str]:
    """Promotion gate. Log-loss is primary; Brier/ECE are sanity checks only.

    min_logloss_improvement resolves per-sport when not passed explicitly:
    small-sample sports demand a wider margin than the daily-cadence sports do.

    Returns (should_promote, reason).
    """
    if min_logloss_improvement is None:
        min_logloss_improvement = (
            _MIN_LOGLOSS_IMPROVEMENT_SMALL_SAMPLE
            if sport in _SMALL_SAMPLE_SPORTS
            else _MIN_LOGLOSS_IMPROVEMENT
        )
    chall_ll = challenger_metrics.get("logloss", 999)
    champ_ll = champion_metrics.get("logloss", 999)
    chall_ece = challenger_metrics.get("ece", 999)
    champ_ece = champion_metrics.get("ece", 999)
    chall_brier = challenger_metrics.get("brier", 999)
    champ_brier = champion_metrics.get("brier", 999)

    if chall_ll >= champ_ll - min_logloss_improvement:
        return (
            False,
            f"log-loss {chall_ll:.4f} not better than champion {champ_ll:.4f} by {min_logloss_improvement}",
        )
    if chall_ece > champ_ece + max_ece_increase:
        return False, f"ECE {chall_ece:.4f} exceeds champion {champ_ece:.4f} by margin"
    if chall_brier > champ_brier + max_brier_increase:
        return False, f"Brier {chall_brier:.4f} exceeds champion {champ_brier:.4f} by margin"
    return True, "all gates passed"
