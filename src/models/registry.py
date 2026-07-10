"""MLflow model registry wrapper.

Only the MLflow-facing pieces are included here. The production system
also mirrors promotion/rollback state into a Postgres `models` table for
SQL joins on predictions — that part lives in the private application
repo alongside the rest of the DB schema and serving layer.
"""

from __future__ import annotations

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


def load_model(run_id: str, framework: str = "sklearn") -> Any:
    """Load a model artifact from MLflow by run_id."""
    _setup_mlflow()
    uri = f"runs:/{run_id}/model"
    if framework == "xgboost":
        return mlflow.xgboost.load_model(uri)
    elif framework == "lightgbm":
        return mlflow.lightgbm.load_model(uri)
    return mlflow.sklearn.load_model(uri)


def get_run_metrics(run_id: str) -> dict[str, float]:
    _setup_mlflow()
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    return dict(run.data.metrics)
