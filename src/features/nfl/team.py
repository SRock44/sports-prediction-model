"""NFL team-level feature engineering.

Windows are trailing-4/8/season-to-date, not NBA/MLB's 5/10/20/etc - a 17-game
season makes "last 20 games" meaningless (per NFL.md's own framing).

Known gap, documented not silently dropped: success rate and 3rd-down/red-zone
conversion rate genuinely need down-and-distance play-level data, which Phase B's
ingestion (weekly aggregate stats, not raw play-by-play) doesn't provide. EPA per
play, computed at ingest time in src/ingest/nfl/games.py, is used as the primary
efficiency signal instead - it's the metric NFL.md itself calls "the single best
team quality number," so this isn't a downgrade, just a narrower feature set than
the full plan envisioned.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from src.features.common import haversine_km, load_team_game_stats_before, rolling_mean

_WINDOWS = (4, 8, 17)

# Home-stadium lat/lon per team abbrev - nflverse's schedule data gives us the
# stadium name/id but not coordinates, so (mirroring MLB's static
# _PARK_FACTORS-style dict pattern) this is hand-maintained rather than a new
# geocoding ingest step. Covers each team's *current* home venue; historical
# relocated-team abbreviations (OAK, SD, STL, LA before 2016) intentionally
# fall back to the travel_km default rather than guessing an old stadium.
_STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.5276, -112.2626),
    "ATL": (33.7554, -84.4008),
    "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528),
    "CHI": (41.8623, -87.6167),
    "CIN": (39.0955, -84.5160),
    "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456),
    "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107),
    "IND": (39.7601, -86.1639),
    "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839),
    "LA": (33.9535, -118.3392),
    "LAC": (33.9535, -118.3392),
    "LV": (36.0909, -115.1833),
    "MIA": (25.9580, -80.2389),
    "MIN": (44.9735, -93.2575),
    "NE": (42.0909, -71.2643),
    "NO": (29.9511, -90.0812),
    "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745),
    "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316),
    "SF": (37.4032, -121.9698),
    "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713),
    "WAS": (38.9078, -76.8645),
}

_DOME_ROOFS = ("dome", "closed")


def build_team_features(
    session: Session,
    team_id: int,
    as_of_utc: datetime,
    elo_rating: float,
    team_abbrev: str | None = None,
    opponent_abbrev: str | None = None,
    is_home: bool = True,
) -> dict[str, Any]:
    """Build NFL team features. opponent_abbrev drives the away-team travel_km calc."""
    games = load_team_game_stats_before(session, team_id, as_of_utc, limit=17)

    feats: dict[str, Any] = {}
    feats["elo"] = elo_rating
    feats["is_home"] = int(is_home)

    if not is_home and team_abbrev and opponent_abbrev:
        away_coords = _STADIUM_COORDS.get(team_abbrev)
        home_coords = _STADIUM_COORDS.get(opponent_abbrev)
        feats["travel_km"] = (
            haversine_km(*away_coords, *home_coords) if away_coords and home_coords else 0.0
        )
    else:
        feats["travel_km"] = 0.0

    if not games:
        _fill_defaults(feats)
        return feats

    points_scored: list[float] = []
    points_allowed: list[float] = []
    epa_per_play: list[float] = []
    offensive_plays: list[float] = []
    turnovers_committed: list[float] = []
    won: list[int] = []
    game_dates: list[datetime] = []

    # load_team_game_stats_before returns most-recent-first (DESC). rolling_mean
    # and _compute_streak both assume oldest-first (they take from the *end* of
    # the list as "most recent") - reverse here so every list built below is in
    # the order those helpers actually expect. Verified against real data
    # (2024 Week 10 NO game): without this reversal, "last4" silently averaged
    # the 4 OLDEST fetched games instead of the 4 most recent.
    for g in reversed(games):
        is_home_game = g["home_team_id"] == team_id
        ps = g["home_score"] if is_home_game else g["away_score"]
        pa = g["away_score"] if is_home_game else g["home_score"]
        if ps is not None:
            points_scored.append(float(ps))
        if pa is not None:
            points_allowed.append(float(pa))
        if ps is not None and pa is not None:
            won.append(int(ps > pa))

        stats = g["stats"] or {}
        if "epa_per_play" in stats:
            epa_per_play.append(float(stats["epa_per_play"]))
        if "offensive_plays" in stats:
            offensive_plays.append(float(stats["offensive_plays"]))
        giveaways = (
            float(stats.get("interceptions", 0))
            + float(stats.get("sack_fumbles_lost", 0))
            + float(stats.get("rushing_fumbles_lost", 0))
            + float(stats.get("receiving_fumbles_lost", 0))
        )
        turnovers_committed.append(giveaways)

        game_dates.append(g["scheduled_utc"])

    for w in _WINDOWS:
        feats[f"points_scored_last{w}"] = rolling_mean(points_scored, w) or 21.5
        feats[f"points_allowed_last{w}"] = rolling_mean(points_allowed, w) or 21.5
        feats[f"point_diff_last{w}"] = (
            feats[f"points_scored_last{w}"] - feats[f"points_allowed_last{w}"]
        )
        feats[f"epa_per_play_last{w}"] = rolling_mean(epa_per_play, w) or 0.0
        feats[f"pace_last{w}"] = rolling_mean(offensive_plays, w) or 62.0
        feats[f"turnovers_committed_last{w}"] = rolling_mean(turnovers_committed, w) or 1.3

    feats["win_pct_season"] = float(np.mean(won)) if won else 0.5
    feats["streak"] = _compute_streak(won)

    # Rest / bye-week / short-week signals
    dates_desc = sorted(game_dates, reverse=True)
    most_recent = dates_desc[0]
    rest_days = (as_of_utc - most_recent).total_seconds() / 86400
    feats["rest_days"] = min(rest_days, 20.0)
    feats["bye_week_just_occurred"] = int(rest_days > 10.0)
    feats["short_week"] = int(rest_days < 5.5)

    return feats


def _fill_defaults(feats: dict[str, Any]) -> None:
    for w in _WINDOWS:
        feats[f"points_scored_last{w}"] = 21.5
        feats[f"points_allowed_last{w}"] = 21.5
        feats[f"point_diff_last{w}"] = 0.0
        feats[f"epa_per_play_last{w}"] = 0.0
        feats[f"pace_last{w}"] = 62.0
        feats[f"turnovers_committed_last{w}"] = 1.3
    feats["win_pct_season"] = 0.5
    feats["streak"] = 0
    feats["rest_days"] = 7.0
    feats["bye_week_just_occurred"] = 0
    feats["short_week"] = 0


def _compute_streak(won: list[int]) -> int:
    if not won:
        return 0
    streak = 0
    last = won[-1]
    for w in reversed(won):
        if w == last:
            streak += 1 if last == 1 else -1
        else:
            break
    return streak


def venue_is_dome(roof: str | None) -> bool:
    return roof in _DOME_ROOFS
