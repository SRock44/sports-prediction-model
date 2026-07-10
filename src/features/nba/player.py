"""NBA player-level feature engineering for props models."""

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

_STAT_KEYS = {
    "PTS": ("statistics", "points"),
    "REB": ("statistics", "reboundsTotal"),
    "AST": ("statistics", "assists"),
    "3PM": ("statistics", "threePointersMade"),
    "MIN": ("statistics", "minutesCalculated"),
    "FGA": ("statistics", "fieldGoalsAttempted"),
    "TOV": ("statistics", "turnovers"),
    "STL": ("statistics", "steals"),
    "BLK": ("statistics", "blocks"),
}


def build_player_features(
    session: Session,
    player_id: int,
    team_id: int,
    opponent_team_id: int,
    as_of_utc: datetime,
    stat: str = "PTS",
    games_played_this_season: int = 0,
) -> dict[str, Any]:
    """Build ~40 player features for an NBA props model."""
    games = load_player_game_stats_before(session, player_id, as_of_utc, limit=25)
    injury_records = load_injuries_before(session, player_id, as_of_utc)

    feats: dict[str, Any] = {}

    # ── Injury / availability ─────────────────────────────────────────────────
    if injury_records:
        latest_injury = injury_records[0]
        status = latest_injury.get("status", "active")
    else:
        status = "active"
    feats["injury_status"] = _encode_injury_status(status)

    # ── Cold-start blend weight ───────────────────────────────────────────────
    feats["season_game_weight"] = min(1.0, games_played_this_season / 20.0)

    if not games:
        _fill_empty_player_feats(feats, stat)
        return feats

    # ── Extract stat time-series ──────────────────────────────────────────────
    stat_values: list[float] = []
    per_min_values: list[float] = []
    minutes: list[float] = []

    path = _STAT_KEYS.get(stat, ("statistics", stat.lower()))

    for g in games:
        raw_stats = g["stats"] or {}
        trad = raw_stats.get("traditional", raw_stats)
        stats_section = trad.get(path[0], trad)

        val = _safe_float(stats_section.get(path[1]))
        min_val = _safe_float(
            (trad.get("statistics") or trad).get("minutesCalculated")
            or (trad.get("statistics") or trad).get("minutes")
        )

        if val is not None:
            stat_values.append(val)
        if min_val is not None and min_val > 0:
            minutes.append(min_val)
            if val is not None:
                per_min_values.append(val / min_val)

    # ── Rolling averages ──────────────────────────────────────────────────────
    for w in [5, 10, 20]:
        feats[f"{stat}_last{w}"] = rolling_mean(stat_values, w) or 0.0
        feats[f"{stat}_per_min_last{w}"] = rolling_mean(per_min_values, w) or 0.0

    feats[f"{stat}_std_last10"] = float(np.std(stat_values[-10:])) if len(stat_values) >= 3 else 0.0
    feats["minutes_last5"] = rolling_mean(minutes, 5) or 20.0
    feats["minutes_last10"] = rolling_mean(minutes, 10) or 20.0

    # ── Home/away split ───────────────────────────────────────────────────────
    # We'd need game location; simplified here — same team_id check
    feats["home_away_split"] = 0.0  # enriched by matchup builder

    # ── Opponent defensive rating at position ─────────────────────────────────
    feats["opp_def_rtg_at_pos"] = 0.0  # filled by matchup builder from team_features

    # ── Rest days ─────────────────────────────────────────────────────────────
    if games:
        last_game_date = games[0]["scheduled_utc"]
        rest_days = (as_of_utc - last_game_date).total_seconds() / 86400
        feats["rest_days"] = min(rest_days, 10.0)
    else:
        feats["rest_days"] = 2.0

    return feats


def _fill_empty_player_feats(feats: dict[str, Any], stat: str) -> None:
    for w in [5, 10, 20]:
        feats[f"{stat}_last{w}"] = 0.0
        feats[f"{stat}_per_min_last{w}"] = 0.0
    feats[f"{stat}_std_last10"] = 0.0
    feats["minutes_last5"] = 20.0
    feats["minutes_last10"] = 20.0
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
