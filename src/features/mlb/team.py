"""MLB team-level feature engineering."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.features.common import load_team_game_stats_before, rolling_mean

_PARK_FACTORS: dict[str, float] = {
    "COL": 1.15,
    "CIN": 1.06,
    "TEX": 1.05,
    "HOU": 1.04,
    "BOS": 1.03,
    "CHC": 1.02,
    "MIL": 1.01,
    "ARI": 1.00,
    "NYY": 0.99,
    "PHI": 0.98,
    "ATL": 0.97,
    "LAD": 0.97,
    "STL": 0.97,
    "SD": 0.96,
    "SF": 0.95,
    "SEA": 0.94,
    "MIN": 0.97,
    "CWS": 1.00,
    "DET": 0.98,
    "CLE": 0.98,
    "TB": 0.98,
    "NYM": 0.98,
    "WSH": 0.99,
    "BAL": 1.00,
    "MIA": 0.96,
    "KC": 0.97,
    "OAK": 0.95,
    "PIT": 0.97,
    "TOR": 1.00,
    "LAA": 0.97,
}


def build_team_features(
    session: Session,
    team_id: int,
    as_of_utc: datetime,
    elo_rating: float,
    home_venue_abbrev: str | None = None,
    is_home: bool = True,
) -> dict[str, Any]:
    """Build MLB team features."""
    games = load_team_game_stats_before(session, team_id, as_of_utc, limit=30)

    feats: dict[str, Any] = {}
    feats["elo"] = elo_rating
    feats["is_home"] = int(is_home)
    feats["park_factor"] = _PARK_FACTORS.get(home_venue_abbrev or "", 1.0)

    if not games:
        _fill_defaults(feats)
        return feats

    runs_scored: list[float] = []
    runs_allowed: list[float] = []
    woba: list[float] = []
    k_pct: list[float] = []
    bb_pct: list[float] = []
    won: list[int] = []
    game_dates: list[datetime] = []

    for g in games:
        is_home_game = g["home_team_id"] == team_id
        rs = g["home_score"] if is_home_game else g["away_score"]
        ra = g["away_score"] if is_home_game else g["home_score"]

        if rs is not None:
            runs_scored.append(float(rs))
        if ra is not None:
            runs_allowed.append(float(ra))

        stats = g["stats"] or {}
        batting = stats.get("batting", {})
        ab = float(batting.get("atBats") or 0)
        if ab > 0:
            hits = float(batting.get("hits") or 0)
            bb = float(batting.get("baseOnBalls") or 0)
            ks = float(batting.get("strikeOuts") or 0)
            woba.append((hits + bb) / ab)
            k_pct.append(ks / ab)
            bb_pct.append(bb / ab)

        if rs is not None and ra is not None:
            won.append(int(rs > ra))
        game_dates.append(g["scheduled_utc"])

    for w in [5, 10, 15]:
        feats[f"runs_scored_last{w}"] = rolling_mean(runs_scored, w) or 4.5
        feats[f"runs_allowed_last{w}"] = rolling_mean(runs_allowed, w) or 4.5
        feats[f"run_diff_last{w}"] = feats[f"runs_scored_last{w}"] - feats[f"runs_allowed_last{w}"]
        feats[f"woba_last{w}"] = rolling_mean(woba, w) or 0.320
        feats[f"k_pct_last{w}"] = rolling_mean(k_pct, w) or 0.22
        feats[f"bb_pct_last{w}"] = rolling_mean(bb_pct, w) or 0.08

    # Win% at multiple windows
    for w in [3, 5, 10, 20]:
        feats[f"win_pct_last{w}"] = rolling_mean(won, w) or 0.5
    feats["win_pct_season"] = float(np.mean(won)) if won else 0.5

    # Streak
    feats["streak"] = _compute_streak(won)

    # Rest & schedule density
    dates_desc = sorted(game_dates, reverse=True)
    most_recent = dates_desc[0]
    rest_days = (as_of_utc - most_recent).total_seconds() / 86400
    feats["rest_days"] = min(rest_days, 5.0)
    feats["b2b"] = int(rest_days < 1.5)
    feats["three_in_four"] = int(_games_in_window(dates_desc, as_of_utc, days=4) >= 3)

    # Bullpen fatigue: IP and pitch count by relievers in last 3 days
    bp_ip, bp_pitches = _get_bullpen_usage(session, team_id, as_of_utc, days=3)
    feats["bullpen_ip_last3d"] = bp_ip
    feats["bullpen_pitches_last3d"] = bp_pitches

    # Opponent quality: avg win% of last 10 opponents (strength-of-schedule proxy)
    feats["opp_quality_last10"] = _get_opp_quality_last10(session, team_id, as_of_utc)

    return feats


def _get_bullpen_usage(
    session: Session, team_id: int, as_of_utc: datetime, days: int
) -> tuple[float, float]:
    """Total IP and pitches thrown by relievers in last `days` days.

    Identifies the starter per game as the pitcher with the most pitches thrown —
    more reliable than joining lineups (which are often absent for historical games).
    Returns (ip, pitches) for relievers only.
    """
    since = as_of_utc - timedelta(days=days)
    try:
        result = session.execute(
            text("""
                WITH pitcher_per_game AS (
                    SELECT
                        COALESCE((pgs.stats->'pitching'->>'numberOfPitches')::float, 0) AS pitches,
                        COALESCE((pgs.stats->'pitching'->>'inningsPitched')::float, 0) AS ip_raw,
                        ROW_NUMBER() OVER (
                            PARTITION BY pgs.game_id
                            ORDER BY
                                COALESCE((pgs.stats->'pitching'->>'numberOfPitches')::float, 0) DESC
                        ) AS rn
                    FROM player_game_stats pgs
                    JOIN games g ON g.id = pgs.game_id
                    WHERE pgs.team_id = :team_id
                      AND g.scheduled_utc >= :since
                      AND g.scheduled_utc < :as_of
                      AND g.status = 'final'
                      AND pgs.stats->'pitching' IS NOT NULL
                      AND pgs.stats->'pitching' != '{}'::jsonb
                      AND COALESCE((pgs.stats->'pitching'->>'numberOfPitches')::float, 0) > 0
                )
                SELECT
                    COALESCE(SUM(CASE WHEN rn > 1 THEN ip_raw  ELSE 0 END), 0) AS bp_ip,
                    COALESCE(SUM(CASE WHEN rn > 1 THEN pitches ELSE 0 END), 0) AS bp_pitches
                FROM pitcher_per_game
            """),
            {"team_id": team_id, "since": since, "as_of": as_of_utc},
        )
        row = result.first()
        ip = float(row.bp_ip) if row and row.bp_ip else 0.0
        pitches = float(row.bp_pitches) if row and row.bp_pitches else 0.0
        return ip, pitches
    except Exception:
        return 0.0, 0.0


def _fill_defaults(feats: dict[str, Any]) -> None:
    for w in [5, 10, 15]:
        feats[f"runs_scored_last{w}"] = 4.5
        feats[f"runs_allowed_last{w}"] = 4.5
        feats[f"run_diff_last{w}"] = 0.0
        feats[f"woba_last{w}"] = 0.320
        feats[f"k_pct_last{w}"] = 0.22
        feats[f"bb_pct_last{w}"] = 0.08
    for w in [3, 5, 10, 20]:
        feats[f"win_pct_last{w}"] = 0.5
    feats["win_pct_season"] = 0.5
    feats["streak"] = 0
    feats["rest_days"] = 2.0
    feats["b2b"] = 0
    feats["three_in_four"] = 0
    feats["bullpen_ip_last3d"] = 0.0
    feats["bullpen_pitches_last3d"] = 0.0
    feats["opp_quality_last10"] = 0.5


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


def _games_in_window(dates_desc: list[datetime], as_of: datetime, days: int) -> int:
    cutoff = as_of - timedelta(days=days)
    return sum(1 for d in dates_desc if d >= cutoff)


def _get_opp_quality_last10(session: Session, team_id: int, as_of_utc: datetime) -> float:
    """Avg win% of the team's last 10 opponents (60-day window before each matchup)."""
    try:
        result = session.execute(
            text("""
                WITH recent_games AS (
                    SELECT
                        CASE WHEN g.home_team_id = :tid THEN g.away_team_id
                             ELSE g.home_team_id END AS opp_id,
                        g.scheduled_utc AS game_date
                    FROM games g
                    WHERE (g.home_team_id = :tid OR g.away_team_id = :tid)
                      AND g.scheduled_utc < :as_of
                      AND g.status = 'final'
                      AND g.home_score IS NOT NULL
                    ORDER BY g.scheduled_utc DESC
                    LIMIT 10
                ),
                opp_win_pcts AS (
                    SELECT
                        AVG(CASE
                            WHEN (g2.home_team_id = rg.opp_id AND g2.home_score > g2.away_score)
                              OR (g2.away_team_id = rg.opp_id AND g2.away_score > g2.home_score)
                            THEN 1.0 ELSE 0.0 END) AS win_pct
                    FROM recent_games rg
                    JOIN games g2
                        ON (g2.home_team_id = rg.opp_id OR g2.away_team_id = rg.opp_id)
                    WHERE g2.scheduled_utc < rg.game_date
                      AND g2.scheduled_utc >= rg.game_date - INTERVAL '60 days'
                      AND g2.status = 'final'
                      AND g2.home_score IS NOT NULL
                    GROUP BY rg.opp_id
                )
                SELECT COALESCE(AVG(win_pct), 0.5) AS opp_quality
                FROM opp_win_pcts
            """),
            {"tid": team_id, "as_of": as_of_utc},
        ).scalar()
        return float(result) if result is not None else 0.5
    except Exception:
        return 0.5
