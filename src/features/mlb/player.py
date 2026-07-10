"""MLB player-level feature engineering for batter and pitcher props.

Feature names MUST match those produced by _load_props_training_data in cli.py.
Training uses rolling SQL features with the same column names — any divergence
produces all-zero inference inputs and collapsed predictions.
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

# Maps canonical stat code → (stats_section, json_key) for extraction
_BATTER_STATS = {
    "H": ("batting", "hits"),
    "HR": ("batting", "homeRuns"),
    "RBI": ("batting", "rbi"),
    "TB": ("batting", "totalBases"),
    "K": ("batting", "strikeOuts"),
    "BB": ("batting", "baseOnBalls"),
}

# Full stat name → (stats_section, json_key, ip_key)
# ip_key is the key for innings-pitched equivalent (used for per_inning features)
_PITCHER_STATS = {
    "PITCHER_K": ("pitching", "strikeOuts", "inningsPitched"),
    "PITCHER_ER": ("pitching", "earnedRuns", "inningsPitched"),
}


def build_batter_features(
    session: Session,
    player_id: int,
    opponent_pitcher_throws: str,
    stadium_factor: float,
    as_of_utc: datetime,
    stat: str = "H",
    games_played_this_season: int = 0,
) -> dict[str, Any]:
    """Features for an MLB batter props model.

    Feature names match _load_props_training_data(sport='mlb', stat=stat).
    Training uses a hardcoded denominator of 9 for per_min features (not real
    minutes), so minutes_last5/10 are always 9.0 and per_min = stat_val / 9.
    """
    games = load_player_game_stats_before(session, player_id, as_of_utc, limit=25)
    injury_records = load_injuries_before(session, player_id, as_of_utc)

    feats: dict[str, Any] = {}
    feats["injury_status"] = _encode_injury(injury_records)
    feats["season_game_weight"] = min(1.0, games_played_this_season / 20.0)
    feats["home_away_split"] = 0.0
    feats["opp_def_rtg_at_pos"] = 0.0

    if not games:
        _fill_batter_defaults(feats, stat)
        return feats

    _, col = _BATTER_STATS.get(stat, ("batting", stat.lower()))
    stat_vals: list[float] = []

    for g in games:
        raw = g["stats"] or {}
        batting = raw.get("batting", raw)
        val = _safe_float(batting.get(col))
        if val is not None:
            stat_vals.append(val)

    for w in [5, 10, 20]:
        mean_w = rolling_mean(stat_vals, w) or 0.0
        feats[f"{stat}_last{w}"] = mean_w
        # Training denominator is 9 (hardcoded innings equivalent for batters)
        feats[f"{stat}_per_min_last{w}"] = mean_w / 9.0

    feats[f"{stat}_std_last10"] = float(np.std(stat_vals[-10:])) if len(stat_vals) >= 3 else 0.0
    # Constant 9.0 matches the training SQL literal "'9'"
    feats["minutes_last5"] = 9.0
    feats["minutes_last10"] = 9.0

    if games:
        last_game_date = games[0]["scheduled_utc"]
        rest = (as_of_utc - last_game_date).total_seconds() / 86400
        feats["rest_days"] = min(rest, 10.0)
    else:
        feats["rest_days"] = 2.0

    return feats


def build_pitcher_features(
    session: Session,
    player_id: int,
    as_of_utc: datetime,
    stat: str = "PITCHER_K",
    games_played_this_season: int = 0,
) -> dict[str, Any]:
    """Features for an MLB pitcher props model.

    stat must be the FULL stat name (PITCHER_K, PITCHER_ER) so feature names
    match training, e.g. PITCHER_K_last5 not K_last5.
    """
    games = load_player_game_stats_before(session, player_id, as_of_utc, limit=15)
    injury_records = load_injuries_before(session, player_id, as_of_utc)

    feats: dict[str, Any] = {}
    feats["injury_status"] = _encode_injury(injury_records)
    feats["season_game_weight"] = min(1.0, games_played_this_season / 10.0)
    feats["home_away_split"] = 0.0
    feats["opp_def_rtg_at_pos"] = 0.0

    if not games:
        _fill_pitcher_defaults(feats, stat)
        return feats

    _section, col, ip_key = _PITCHER_STATS.get(stat, ("pitching", stat.lower(), "inningsPitched"))
    stat_vals: list[float] = []
    ip_vals: list[float] = []

    for g in games:
        raw = g["stats"] or {}
        pitching = raw.get("pitching", raw)
        val = _safe_float(pitching.get(col))
        ip = _parse_ip(pitching.get(ip_key))
        if val is not None:
            stat_vals.append(val)
        if ip is not None and ip > 0:
            ip_vals.append(ip)

    for w in [5, 10, 20]:
        mean_w = rolling_mean(stat_vals, w) or 0.0
        feats[f"{stat}_last{w}"] = mean_w
        mean_ip = rolling_mean(ip_vals, w) or 5.0
        feats[f"{stat}_per_min_last{w}"] = mean_w / mean_ip if mean_ip > 0 else 0.0

    feats[f"{stat}_std_last10"] = float(np.std(stat_vals[-10:])) if len(stat_vals) >= 3 else 0.0
    feats["minutes_last5"] = rolling_mean(ip_vals, 5) or 5.0
    feats["minutes_last10"] = rolling_mean(ip_vals, 10) or 5.0

    if games:
        last_game_date = games[0]["scheduled_utc"]
        rest = (as_of_utc - last_game_date).total_seconds() / 86400
        feats["rest_days"] = min(rest, 10.0)
    else:
        feats["rest_days"] = 5.0

    return feats


def _parse_ip(v: Any) -> float | None:
    """Parse innings pitched: '6.2' means 6 full innings + 2 outs = 6.667."""
    try:
        f = float(v)
        whole = int(f)
        frac = round(f - whole, 1)
        return whole + frac / 3.0
    except (TypeError, ValueError):
        return None


def _encode_injury(records: list[dict[str, Any]]) -> float:
    if not records:
        return 1.0
    status = records[0].get("status", "active")
    return {"active": 1.0, "il_10": 0.0, "il_60": 0.0, "questionable": 0.7}.get(status, 0.8)


def _fill_batter_defaults(feats: dict[str, Any], stat: str) -> None:
    for w in [5, 10, 20]:
        feats[f"{stat}_last{w}"] = 0.0
        feats[f"{stat}_per_min_last{w}"] = 0.0
    feats[f"{stat}_std_last10"] = 0.0
    feats["minutes_last5"] = 9.0
    feats["minutes_last10"] = 9.0
    feats["home_away_split"] = 0.0
    feats["opp_def_rtg_at_pos"] = 0.0
    feats["rest_days"] = 2.0


def _fill_pitcher_defaults(feats: dict[str, Any], stat: str) -> None:
    for w in [5, 10, 20]:
        feats[f"{stat}_last{w}"] = 0.0
        feats[f"{stat}_per_min_last{w}"] = 0.0
    feats[f"{stat}_std_last10"] = 0.0
    feats["minutes_last5"] = 5.0
    feats["minutes_last10"] = 5.0
    feats["home_away_split"] = 0.0
    feats["opp_def_rtg_at_pos"] = 0.0
    feats["rest_days"] = 5.0


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
