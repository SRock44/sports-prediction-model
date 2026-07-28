"""NHL player-level feature engineering for props models.

Skater props: GOALS, ASSISTS, POINTS, SOG. Goalie props: SAVES. Field names
match src/ingest/nhl/games.py's flat PlayerGameStats.stats shape (role +
raw NHL API fields), not NBA's nested statistics/traditional structure.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from src.features.common import (
    load_injuries_before,
    load_player_game_stats_before,
    rolling_mean,
)

_STAT_KEYS: dict[str, str] = {
    "GOALS": "goals",
    "ASSISTS": "assists",
    "POINTS": "points",
    "SOG": "sog",
    "SAVES": "saves",
}

_GOALIE_STATS = frozenset({"SAVES"})


def _toi_to_minutes(toi: str | None) -> float:
    """NHL API reports TOI as "MM:SS" — convert to decimal minutes."""
    if not toi:
        return 0.0
    try:
        minutes, seconds = toi.split(":")
        return int(minutes) + int(seconds) / 60.0
    except (ValueError, AttributeError):
        return 0.0


def build_player_features(
    session: Session,
    player_id: int,
    team_id: int,
    opponent_team_id: int,
    as_of_utc: datetime,
    stat: str = "POINTS",
    games_played_this_season: int = 0,
) -> dict[str, Any]:
    """Build NHL player features for a props model."""
    games = load_player_game_stats_before(session, player_id, as_of_utc, limit=25)
    injury_records = load_injuries_before(session, player_id, as_of_utc)

    feats: dict[str, Any] = {}

    status = injury_records[0].get("status", "active") if injury_records else "active"
    feats["injury_status"] = _encode_injury_status(status)
    feats["season_game_weight"] = min(1.0, games_played_this_season / 20.0)

    is_goalie_stat = stat in _GOALIE_STATS
    stat_key = _STAT_KEYS.get(stat, stat.lower())

    if not games:
        _fill_empty_player_feats(feats, stat, is_goalie_stat)
        return feats

    stat_values: list[float] = []
    per_min_values: list[float] = []
    minutes: list[float] = []

    for g in games:
        raw_stats = g["stats"] or {}
        val = _safe_float(raw_stats.get(stat_key))
        toi_min = _toi_to_minutes(raw_stats.get("toi"))

        if val is not None:
            stat_values.append(val)
        if toi_min > 0:
            minutes.append(toi_min)
            if val is not None:
                per_min_values.append(val / toi_min)

    for w in [5, 10, 20]:
        feats[f"{stat}_last{w}"] = rolling_mean(stat_values, w) or 0.0
        feats[f"{stat}_per_min_last{w}"] = rolling_mean(per_min_values, w) or 0.0

    # Named minutes_last{5,10} (not toi_last{5,10}) to match the training-side
    # feature names src/cli.py's _load_props_training_data generates — a
    # mismatch here would silently zero these out at scoring time (same class
    # of training/serving skew as the PRA bug noted in nba/player.py).
    feats[f"{stat}_std_last10"] = float(np.std(stat_values[-10:])) if len(stat_values) >= 3 else 0.0
    feats["minutes_last5"] = rolling_mean(minutes, 5) or (16.0 if not is_goalie_stat else 58.0)
    feats["minutes_last10"] = rolling_mean(minutes, 10) or (16.0 if not is_goalie_stat else 58.0)

    feats["home_away_split"] = 0.0  # enriched by matchup builder
    feats["opp_def_rtg_at_pos"] = 0.0  # filled by matchup builder from team features

    if games:
        last_game_date = games[0]["scheduled_utc"]
        rest_days = (as_of_utc - last_game_date).total_seconds() / 86400
        feats["rest_days"] = min(rest_days, 10.0)
    else:
        feats["rest_days"] = 2.0

    return feats


def _fill_empty_player_feats(feats: dict[str, Any], stat: str, is_goalie_stat: bool) -> None:
    for w in [5, 10, 20]:
        feats[f"{stat}_last{w}"] = 0.0
        feats[f"{stat}_per_min_last{w}"] = 0.0
    feats[f"{stat}_std_last10"] = 0.0
    feats["minutes_last5"] = 58.0 if is_goalie_stat else 16.0
    feats["minutes_last10"] = 58.0 if is_goalie_stat else 16.0
    feats["home_away_split"] = 0.0
    feats["opp_def_rtg_at_pos"] = 0.0
    feats["rest_days"] = 2.0


def _encode_injury_status(status: str) -> float:
    return {"active": 1.0, "probable": 0.9, "questionable": 0.6, "out": 0.0}.get(status, 0.8)


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
