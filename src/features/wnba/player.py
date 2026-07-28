"""WNBA player-level feature engineering for props models.

Direct port of src/features/nba/player.py - same stat categories, same target
vocabulary (PTS/REB/AST/3PM/PRA per plan doc §10). Only change: rolling windows
scaled to the shorter season (5/10/20 -> 3/6/12), matching team.py's scaling.
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

# PRA isn't a real box-score field - it's points + rebounds + assists. Without
# this, the unmatched-stat fallback in build_player_features looked up a
# literal "pra" key that never exists, silently zeroing every PRA rolling
# feature at live scoring time (2026-07-12: every WNBA PRA prediction
# collapsed to ~6.5 regardless of player, while training - which computes PRA
# correctly via SQL - saw real values, a training/serving skew).
_COMPOSITE_STATS: dict[str, tuple[str, ...]] = {
    "PRA": ("PTS", "REB", "AST"),
}

_WINDOWS = [3, 6, 12]


def build_player_features(
    session: Session,
    player_id: int,
    team_id: int,
    opponent_team_id: int,
    as_of_utc: datetime,
    stat: str = "PTS",
    games_played_this_season: int = 0,
) -> dict[str, Any]:
    """Build player features for a WNBA props model."""
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
    # /12 (vs NBA's /20) to match the shorter season - a WNBA player reaches
    # "established this season" faster in game-count terms.
    feats["season_game_weight"] = min(1.0, games_played_this_season / 12.0)

    if not games:
        _fill_empty_player_feats(feats, stat)
        return feats

    # ── Extract stat time-series ──────────────────────────────────────────────
    stat_values: list[float] = []
    per_min_values: list[float] = []
    minutes: list[float] = []

    component_paths = (
        [_STAT_KEYS[c] for c in _COMPOSITE_STATS[stat]] if stat in _COMPOSITE_STATS else None
    )
    path = _STAT_KEYS.get(stat, ("statistics", stat.lower()))

    for g in games:
        raw_stats = g["stats"] or {}
        trad = raw_stats.get("traditional", raw_stats)

        if component_paths is not None:
            val = None
            total = 0.0
            for csec, ckey in component_paths:
                csection = trad.get(csec, trad)
                cval = _safe_float(csection.get(ckey))
                if cval is not None:
                    total += cval
                    val = total
        else:
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
    for w in _WINDOWS:
        feats[f"{stat}_last{w}"] = rolling_mean(stat_values, w) or 0.0
        feats[f"{stat}_per_min_last{w}"] = rolling_mean(per_min_values, w) or 0.0

    feats[f"{stat}_std_last6"] = float(np.std(stat_values[-6:])) if len(stat_values) >= 3 else 0.0
    feats["minutes_last3"] = rolling_mean(minutes, 3) or 20.0
    feats["minutes_last6"] = rolling_mean(minutes, 6) or 20.0

    # ── Home/away split ───────────────────────────────────────────────────────
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
    for w in _WINDOWS:
        feats[f"{stat}_last{w}"] = 0.0
        feats[f"{stat}_per_min_last{w}"] = 0.0
    feats[f"{stat}_std_last6"] = 0.0
    feats["minutes_last3"] = 20.0
    feats["minutes_last6"] = 20.0
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
