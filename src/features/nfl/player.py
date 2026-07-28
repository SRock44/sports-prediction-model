"""NFL player-level feature engineering (for props and QB-signal features).

Known gap, documented not silently dropped: snap-count percentage (nfl_data_py
has import_snap_counts) isn't ingested by Phase B, so there's no availability/
role-share feature beyond games_played_this_season and the raw usage stats
(attempts/carries/targets) already captured. Worth adding as its own ingest
task later if role-share proves predictive; not built here to keep Phase B/C
scoped to what's actually verified working end-to-end.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from src.features.common import load_player_game_stats_before, rolling_mean

_WINDOWS = (4, 8)

# stat -> the PlayerGameStats.stats key(s) this prop target rolls up from.
_STAT_FIELDS: dict[str, tuple[str, ...]] = {
    "PASS_YDS": ("passing_yards",),
    "PASS_TD": ("passing_tds",),
    "RUSH_YDS": ("rushing_yards",),
    "REC": ("receptions",),
    "REC_YDS": ("receiving_yards",),
    "REC_TD": ("receiving_tds",),
    "ANYTIME_TD": ("rushing_tds", "receiving_tds"),
}


def build_player_features(
    session: Session,
    player_id: int,
    team_id: int,
    opponent_team_id: int,
    as_of_utc: datetime,
    stat: str = "PASS_YDS",
    games_played_this_season: int = 0,
) -> dict[str, Any]:
    """Build rolling-usage features for one player, for one prop stat target."""
    games = load_player_game_stats_before(session, player_id, as_of_utc, limit=17)

    feats: dict[str, Any] = {"games_played_this_season": games_played_this_season}
    fields = _STAT_FIELDS.get(stat, (stat.lower(),))

    if not games:
        _fill_defaults(feats, stat)
        return feats

    target_values: list[float] = []
    attempts_or_targets: list[float] = []
    tds_any: list[float] = []

    # load_player_game_stats_before returns most-recent-first; rolling_mean's
    # values[-window:] assumes oldest-first. Fetch limit (17) exceeds the
    # windows used here (4, 8), so without reversing this would silently
    # average stale games instead of the true most recent N (verified as a
    # real bug in src/features/nfl/team.py against production data).
    for g in reversed(games):
        stats = g["stats"] or {}
        val = sum(float(stats.get(f, 0) or 0) for f in fields)
        target_values.append(val)

        usage = stats.get("attempts") or stats.get("carries") or stats.get("targets") or 0
        attempts_or_targets.append(float(usage))

        tds_any.append(
            float(stats.get("rushing_tds", 0) or 0) + float(stats.get("receiving_tds", 0) or 0)
        )

    for w in _WINDOWS:
        feats[f"{stat.lower()}_last{w}"] = rolling_mean(target_values, w) or 0.0
        feats[f"usage_last{w}"] = rolling_mean(attempts_or_targets, w) or 0.0

    # tds_any is now oldest-first (matching the reversal above), so "last 8"
    # is the tail of the list, not the head.
    recent_tds = tds_any[-8:]
    feats["any_td_rate_last8"] = (
        sum(1 for v in recent_tds if v > 0) / len(recent_tds) if recent_tds else 0.0
    )
    feats["games_available_last8"] = float(min(len(games), 8))

    return feats


def _fill_defaults(feats: dict[str, Any], stat: str) -> None:
    for w in _WINDOWS:
        feats[f"{stat.lower()}_last{w}"] = 0.0
        feats[f"usage_last{w}"] = 0.0
    feats["any_td_rate_last8"] = 0.0
    feats["games_available_last8"] = 0.0
