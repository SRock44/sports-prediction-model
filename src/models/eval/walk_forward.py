"""Walk-forward cross-validation split generator.

Rule: train on seasons [N…N+k], validate on season N+k+1.
Never shuffle. Never allow any test sample's date to precede any training sample's date.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def walk_forward_splits(
    df: pd.DataFrame,
    date_col: str = "scheduled_utc",
    season_col: str = "season",
    min_train_seasons: int = 2,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield (train_df, val_df) for each walk-forward fold.

    Each fold adds one season to the training window and tests on the next.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    seasons = sorted(df[season_col].unique())

    if len(seasons) < min_train_seasons + 1:
        raise ValueError(f"Need at least {min_train_seasons + 1} seasons, got {len(seasons)}")

    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        val_season = seasons[i]

        train_df = df[df[season_col].isin(train_seasons)].copy()
        val_df = df[df[season_col] == val_season].copy()

        # Strict chronological guard — no leakage even across season boundaries
        assert train_df[date_col].max() <= val_df[date_col].min(), (
            f"Leakage detected in walk-forward split at season {val_season}"
        )

        yield train_df, val_df


def rolling_window_split(
    df: pd.DataFrame,
    date_col: str = "scheduled_utc",
    season_col: str = "season",
    window_seasons: int = 5,
    holdout_weeks: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (training_df, holdout_df) for a single production training run.

    - training_df: last `window_seasons` seasons, excluding the holdout period
    - holdout_df: last `holdout_weeks` weeks of completed games (never in training)
    """
    from datetime import timedelta

    from src.core.time import utc_now

    df = df.sort_values(date_col).reset_index(drop=True)
    seasons = sorted(df[season_col].unique())

    # Take only the most recent window_seasons
    recent_seasons = seasons[-window_seasons:] if len(seasons) >= window_seasons else seasons
    df = df[df[season_col].isin(recent_seasons)].copy()

    cutoff = utc_now() - timedelta(weeks=holdout_weeks)
    training_df = df[pd.to_datetime(df[date_col]) < cutoff].copy()
    holdout_df = df[
        (pd.to_datetime(df[date_col]) >= cutoff)
        & (df.get("status", pd.Series(["final"] * len(df))) == "final")
    ].copy()

    return training_df, holdout_df
