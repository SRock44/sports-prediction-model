"""MLflow model registry wrapper.

Only the MLflow-facing pieces are included here. The production system
also mirrors promotion/rollback state into a Postgres `models` table for
SQL joins on predictions — that part lives in the private application
repo alongside the rest of the DB schema and serving layer.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost

from src.core.config import settings
from src.core.logging import get_logger

log = get_logger(__name__)


def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def log_model_run(
    run_name: str,
    sport: str,
    kind: str,
    target: str,
    model: Any,
    metrics: dict[str, float],
    params: dict[str, Any],
    feature_names: list[str],
    training_range: tuple[str, str],
    model_framework: str = "sklearn",
) -> str:
    """Log a training run to MLflow. Returns the run_id."""
    _setup_mlflow()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "sport": sport,
                "kind": kind,
                "target": target,
                "training_start": training_range[0],
                "training_end": training_range[1],
                "n_features": len(feature_names),
            }
        )
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"feature_names": feature_names}, "feature_names.json")

        if model_framework == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        elif model_framework == "lightgbm":
            mlflow.lightgbm.log_model(model, artifact_path="model")
        else:
            # StackedEnsemble (winner model) and QuantileBundle (props model) both
            # wrap CatBoost/LightGBM/XGBoost sub-models, none of which are in
            # skops' default trusted-type allowlist. These are our own
            # application types (or well-known ML libraries), not arbitrary user
            # input, so explicitly trusting them is safe.
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                skops_trusted_types=[
                    "catboost.core.CatBoostClassifier",
                    "collections.OrderedDict",
                    "lightgbm.basic.Booster",
                    "lightgbm.sklearn.LGBMClassifier",
                    "lightgbm.sklearn.LGBMRegressor",
                    "src.models.train_props.QuantileBundle",
                    "src.models.train_winner.StackedEnsemble",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier",
                ],
            )

        log.info(
            "mlflow.run_logged",
            run_id=run.info.run_id,
            sport=sport,
            kind=kind,
            target=target,
            metrics=metrics,
        )
        return run.info.run_id  # type: ignore[no-any-return]


def _cap_inference_threads(model: Any) -> Any:
    """Force every LightGBM/CatBoost/XGBoost sub-estimator in a loaded model to
    single-threaded inference.

    train_winner.py/train_props.py deliberately set n_jobs/thread_count high on
    LightGBM/CatBoost so training runs fast on a dedicated training host - but those
    values are baked into the pickled model artifact and travel with it into scoring.
    If the serving process scores one game at a time from multiple concurrent worker
    processes, each predict_proba() call spins up a fresh multi-threaded OpenMP team
    *inside an already-forked child process*. OpenMP's runtime isn't fork-safe once a
    multi-threaded team exists post-fork - this reliably SIGSEGVs inside libgomp.so
    (a WorkerLostError with no exception the caller ever sees, no prediction written).
    Capping to 1 thread here means no OpenMP team is ever created, sidestepping the
    bug entirely - scoring one row at a time never needed many threads anyway.
    """
    seen: set[int] = set()

    def _cap(obj: Any) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        get_params = getattr(obj, "get_params", None)
        if callable(get_params):
            try:
                params = get_params()
            except Exception:
                params = {}
            overrides = {p: 1 for p in ("n_jobs", "thread_count") if p in params}
            if overrides:
                with contextlib.suppress(Exception):
                    obj.set_params(**overrides)
        # XGBClassifier.set_params(n_jobs=1) only affects predict_proba() - a raw
        # native Booster (e.g. via get_booster().predict(...) for SHAP contributions)
        # keeps its own nthread baked in from training regardless of the sklearn
        # wrapper's n_jobs, so that path needs capping separately.
        get_booster = getattr(obj, "get_booster", None)
        if callable(get_booster):
            with contextlib.suppress(Exception):
                get_booster().set_param({"nthread": 1})
        for attr in ("xgb_clf", "lgb_clf", "cat_clf"):
            _cap(getattr(obj, attr, None))
        quantile_models = getattr(obj, "quantile_models", None)
        if isinstance(quantile_models, dict):
            for sub in quantile_models.values():
                _cap(sub)

    _cap(model)
    return model


def load_model(run_id: str, framework: str = "sklearn", *, retries: int = 3) -> Any:
    """Load a model artifact from MLflow by run_id.

    Retries on failure: skops/xgboost's GPU device probe on load has been observed to
    intermittently throw a spurious ModuleNotFoundError when it races GPU visibility
    checks against another process on the host. Since callers are expected to smoke-test
    that an artifact loads before promoting it, a load failure here is treated as a
    transient runtime hiccup rather than a corrupt artifact.
    """
    _setup_mlflow()
    uri = f"runs:/{run_id}/model"

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            if framework == "xgboost":
                return mlflow.xgboost.load_model(uri)
            elif framework == "lightgbm":
                return mlflow.lightgbm.load_model(uri)
            return _cap_inference_threads(mlflow.sklearn.load_model(uri))
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                log.warning(
                    "model.load_retry",
                    run_id=run_id,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                time.sleep(1.5 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def get_run_metrics(run_id: str) -> dict[str, float]:
    _setup_mlflow()
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    return dict(run.data.metrics)
