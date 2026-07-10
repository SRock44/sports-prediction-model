"""Walk-forward backtest report generator.

Writes a Markdown report to reports/{sport}_{kind}_{date}.md with per-season
metrics, calibration data, and sanity checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.core.time import utc_now
from src.models.eval.metrics import compute_all_winner_metrics
from src.models.eval.walk_forward import walk_forward_splits
from src.models.train_winner import train_winner_model

log = get_logger(__name__)
REPORTS_DIR = Path("reports")


def generate_winner_backtest_report(
    sport: str,
    df: pd.DataFrame,
    feature_names: list[str],
    min_train_seasons: int = 2,
) -> Path:
    """Run walk-forward CV, compute metrics per fold, write Markdown report."""
    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / f"{sport}_winner_{utc_now().strftime('%Y%m%d')}.md"

    fold_results: list[dict[str, Any]] = []

    for fold_idx, (train_df, val_df) in enumerate(
        walk_forward_splits(df, min_train_seasons=min_train_seasons)
    ):
        val_season = val_df["season"].iloc[0]
        log.info("backtest.fold", sport=sport, val_season=val_season, train_n=len(train_df))

        try:
            # Use a separate MLflow experiment so backtest runs don't pollute
            # the production experiment that promote_model queries.
            import mlflow

            from src.core.config import settings

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(f"{settings.mlflow_experiment_name}_backtest")

            run_id, _ = train_winner_model(
                sport=sport,
                training_df=train_df,
                feature_names=feature_names,
                holdout_df=val_df,
                n_optuna_trials=10,
                run_name=f"{sport}_backtest_fold{fold_idx}",
            )
            from src.models.registry import load_model

            model = load_model(run_id, framework="sklearn")
            X_val = val_df[feature_names].values.astype(np.float32)
            y_val = val_df["target"].values.astype(int)
            proba = model.predict_proba(X_val)[:, 1]
            metrics = compute_all_winner_metrics(y_val, proba)
            fold_results.append({"season": val_season, **metrics})
        except Exception as exc:
            log.error("backtest.fold_failed", fold=fold_idx, error=str(exc))
            fold_results.append({"season": val_season, "error": str(exc)})

    _write_report(report_path, sport, "winner", fold_results, feature_names)
    log.info("backtest.report_written", path=str(report_path))
    return report_path


def _write_report(
    path: Path,
    sport: str,
    kind: str,
    fold_results: list[dict[str, Any]],
    feature_names: list[str],
) -> None:
    lines = [
        f"# Backtest Report: {sport.upper()} {kind}",
        f"Generated: {utc_now().isoformat()}",
        "",
        "## Per-Season Metrics (Walk-Forward CV)",
        "",
        "| Season | Log-Loss | Brier | Accuracy | ECE | N |",
        "|--------|----------|-------|----------|-----|---|",
    ]

    for r in fold_results:
        if "error" in r:
            lines.append(f"| {r['season']} | ERROR: {r['error']} | | | | |")
        else:
            lines.append(
                f"| {r['season']} | {r.get('logloss', '-'):.4f} | "
                f"{r.get('brier', '-'):.4f} | {r.get('accuracy', '-'):.3f} | "
                f"{r.get('ece', '-'):.4f} | {r.get('n_samples', '-')} |"
            )

    # Summary stats
    valid_folds = [r for r in fold_results if "error" not in r]
    if valid_folds:
        avg_ll = np.mean([r["logloss"] for r in valid_folds])
        avg_brier = np.mean([r["brier"] for r in valid_folds])
        avg_acc = np.mean([r["accuracy"] for r in valid_folds])
        lines += [
            "",
            "## Summary",
            "",
            f"- **Mean log-loss**: {avg_ll:.4f}",
            f"- **Mean Brier**: {avg_brier:.4f}",
            f"- **Mean accuracy**: {avg_acc:.3f}",
            f"- **Realistic baseline** ({sport} moneyline): NBA ~0.680-0.685 log-loss, MLB ~0.690",
            "",
            "## Sanity Checks",
            "",
            "- If accuracy > 0.72 for NBA or > 0.63 for MLB on holdout, **suspect leakage**.",
            "- All splits are strictly chronological (older train → newer val, no exceptions).",
            "",
            "## Features Used",
            "",
            f"{len(feature_names)} features: "
            + ", ".join(feature_names[:20])
            + (" ..." if len(feature_names) > 20 else ""),
        ]

    path.write_text("\n".join(lines), encoding="utf-8")
