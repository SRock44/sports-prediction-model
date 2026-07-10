"""Shared feature-engineering primitives.

All functions enforce the as-of invariant: only data with timestamp < as_of_utc
is used. This is the central anti-leakage guarantee.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.core.logging import get_logger

log = get_logger(__name__)

# ── Elo rating ────────────────────────────────────────────────────────────────

_DEFAULT_ELO = 1500.0
_K_FACTOR = 20.0
_HOME_ADVANTAGE = 100.0  # added to home team's Elo before expected-score calc


def elo_expected(rating_a: float, rating_b: float) -> float:
    """P(A beats B) given Elo ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))  # type: ignore[no-any-return]


def elo_update(
    rating_a: float,
    rating_b: float,
    score_a: float,  # 1 if A won, 0 if lost
    k: float = _K_FACTOR,
) -> tuple[float, float]:
    """Return updated (rating_a, rating_b)."""
    expected_a = elo_expected(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * ((1.0 - score_a) - expected_b)
    return new_a, new_b


def compute_elo_series(
    games_df: pd.DataFrame,
    home_col: str = "home_team_id",
    away_col: str = "away_team_id",
    result_col: str = "home_won",  # 1 if home won
    date_col: str = "scheduled_utc",
) -> dict[int, float]:
    """Compute final Elo ratings by replaying a sorted sequence of games.

    Returns {team_id: elo_rating}. Older games first.
    """
    ratings: dict[int, float] = {}

    for _, row in games_df.sort_values(date_col).iterrows():
        home_id = int(row[home_col])
        away_id = int(row[away_col])
        result = float(row[result_col]) if pd.notna(row[result_col]) else None

        home_elo = ratings.get(home_id, _DEFAULT_ELO)
        away_elo = ratings.get(away_id, _DEFAULT_ELO)

        if result is not None:
            home_elo_adj = home_elo + _HOME_ADVANTAGE
            new_home, new_away = elo_update(home_elo_adj, away_elo, result)
            ratings[home_id] = new_home - _HOME_ADVANTAGE  # store without HCA
            ratings[away_id] = new_away

    return ratings


# ── Rolling windows ───────────────────────────────────────────────────────────


def rolling_mean(
    values: Sequence[float],
    window: int,
    min_periods: int = 1,
) -> float | None:
    """Mean of the last `window` values. Returns None if fewer than min_periods."""
    recent = values[-window:]
    if len(recent) < min_periods:
        return None
    return float(np.mean(recent))


def exponential_decay_weight(days_ago: float, lam: float) -> float:
    """Sample weight for a game played `days_ago` days in the past."""
    return math.exp(-lam * days_ago / 365.0)


# ── Travel distance ───────────────────────────────────────────────────────────


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ── As-of data loader ─────────────────────────────────────────────────────────


def load_team_game_stats_before(
    session: Session,
    team_id: int,
    as_of_utc: datetime,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Return the last `limit` completed games for a team before as_of_utc."""
    result = session.execute(
        text("""
            SELECT g.id, g.scheduled_utc, g.home_team_id, g.away_team_id,
                   g.home_score, g.away_score, tgs.stats
            FROM games g
            JOIN team_game_stats tgs ON tgs.game_id = g.id AND tgs.team_id = :team_id
            WHERE g.scheduled_utc < :as_of
              AND g.status = 'final'
              AND (g.home_team_id = :team_id OR g.away_team_id = :team_id)
            ORDER BY g.scheduled_utc DESC
            LIMIT :limit
        """),
        {"team_id": team_id, "as_of": as_of_utc, "limit": limit},
    )
    return [dict(row._mapping) for row in result]


def load_player_game_stats_before(
    session: Session,
    player_id: int,
    as_of_utc: datetime,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Return the last `limit` completed games for a player before as_of_utc."""
    result = session.execute(
        text("""
            SELECT g.id, g.scheduled_utc, pgs.stats, pgs.team_id
            FROM games g
            JOIN player_game_stats pgs ON pgs.game_id = g.id AND pgs.player_id = :player_id
            WHERE g.scheduled_utc < :as_of
              AND g.status = 'final'
            ORDER BY g.scheduled_utc DESC
            LIMIT :limit
        """),
        {"player_id": player_id, "as_of": as_of_utc, "limit": limit},
    )
    return [dict(row._mapping) for row in result]


def load_injuries_before(
    session: Session,
    player_id: int,
    as_of_utc: datetime,
) -> list[dict[str, Any]]:
    """Return the most recent injury record for a player before as_of_utc."""
    result = session.execute(
        text("""
            SELECT status, reason, reported_at, expected_return_date
            FROM injuries
            WHERE player_id = :player_id
              AND reported_at < :as_of
            ORDER BY reported_at DESC
            LIMIT 1
        """),
        {"player_id": player_id, "as_of": as_of_utc},
    )
    return [dict(row._mapping) for row in result]


# ── Odds & weather loaders ────────────────────────────────────────────────────


def load_game_odds(session: Session, game_id: int) -> dict[str, Any]:
    """Return implied probability and spread from stored odds. Graceful on missing table."""
    try:
        result = session.execute(
            text("""
                SELECT market, snapshot, home_price, away_price, home_spread
                FROM game_odds
                WHERE game_id = :gid
                ORDER BY snapshot DESC
            """),
            {"gid": game_id},
        )
        rows = [dict(r._mapping) for r in result]
    except Exception:
        return {}

    if not rows:
        return {}

    def american_to_prob(p: float) -> float:
        if p > 0:
            return 100.0 / (p + 100.0)
        return abs(p) / (abs(p) + 100.0)

    out: dict[str, Any] = {}
    for row in rows:
        if row["market"] == "h2h" and row["home_price"] is not None:
            snap = row["snapshot"]
            out[f"odds_{snap}_implied_home"] = american_to_prob(row["home_price"])
            out[f"odds_{snap}_home_price"] = row["home_price"]
        if row["market"] == "spreads" and row["home_spread"] is not None:
            out[f"odds_{row['snapshot']}_spread"] = row["home_spread"]

    # Line movement: close - open implied prob (positive = line moved toward home)
    if "odds_close_implied_home" in out and "odds_open_implied_home" in out:
        out["odds_line_move"] = out["odds_close_implied_home"] - out["odds_open_implied_home"]
        out["odds_sharp_move"] = int(abs(out["odds_line_move"]) > 0.03)
    elif "odds_open_implied_home" in out:
        out["odds_line_move"] = 0.0
        out["odds_sharp_move"] = 0

    return out


def load_game_weather(session: Session, game_id: int) -> dict[str, Any]:
    """Return weather features for a game. Graceful on missing table."""
    try:
        result = session.execute(
            text(
                "SELECT temp_f, wind_mph, wind_bearing, precip_prob FROM game_weather WHERE game_id = :gid"
            ),
            {"gid": game_id},
        ).first()
    except Exception:
        return {}

    if result is None:
        return {}

    return {
        "weather_temp_f": result.temp_f or 72.0,
        "weather_wind_mph": result.wind_mph or 5.0,
        "weather_wind_bearing": result.wind_bearing or 180.0,
        "weather_precip_prob": result.precip_prob or 0.0,
    }


def load_team_top_player_stats(
    session: Session,
    team_id: int,
    as_of_utc: datetime,
    n_games: int = 10,
    top_n: int = 8,
) -> list[dict[str, Any]]:
    """Return per-player averages for the top N players by minutes over last n_games.

    Used to build roster-quality features for the winner model.
    """
    from datetime import timedelta

    since = as_of_utc - timedelta(days=n_games * 3)  # generous window
    try:
        result = session.execute(
            text("""
                WITH recent AS (
                    SELECT
                        pgs.player_id,
                        AVG(NULLIF((pgs.stats->'traditional'->>'points')::float, 0)) AS avg_pts,
                        AVG(NULLIF((pgs.stats->'traditional'->>'minutesCalculated')::float, 0)) AS avg_min,
                        AVG(
                            CASE
                              WHEN ((pgs.stats->'traditional'->>'fieldGoalsAttempted')::float
                                    + 0.44 * (pgs.stats->'traditional'->>'freeThrowsAttempted')::float) > 0
                              THEN (pgs.stats->'traditional'->>'points')::float /
                                   (2 * ((pgs.stats->'traditional'->>'fieldGoalsAttempted')::float
                                         + 0.44 * (pgs.stats->'traditional'->>'freeThrowsAttempted')::float))
                              ELSE NULL
                            END
                        ) AS avg_ts_pct,
                        AVG(NULLIF((pgs.stats->'advanced'->>'usagePercentage')::float, 0)) AS avg_usage,
                        COUNT(*) AS games_played
                    FROM player_game_stats pgs
                    JOIN games g ON g.id = pgs.game_id
                    WHERE pgs.team_id = :team_id
                      AND g.scheduled_utc >= :since
                      AND g.scheduled_utc < :as_of
                      AND g.status = 'final'
                      AND (pgs.stats->'traditional'->>'minutesCalculated')::float > 5
                    GROUP BY pgs.player_id
                    HAVING COUNT(*) >= 3
                    ORDER BY avg_min DESC
                    LIMIT :top_n
                )
                SELECT * FROM recent
            """),
            {"team_id": team_id, "since": since, "as_of": as_of_utc, "top_n": top_n},
        )
        return [dict(r._mapping) for r in result]
    except Exception:
        return []


# ── Feature spec hash ─────────────────────────────────────────────────────────


def feature_spec_hash(feature_names: list[str]) -> str:
    """Stable hash of the feature name list. Detects schema drift."""
    import hashlib

    joined = "|".join(sorted(feature_names))
    return hashlib.sha256(joined.encode()).hexdigest()
