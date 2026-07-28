"""WNBA team-level feature engineering.

Ported from src/features/nba/team.py - same underlying game (four quarters, shot
clock, standard basketball box score), so the stat categories and formulas are a
direct port. Two deliberate deviations from a literal copy, per plans/sports-
integration/WNBA.md §4:

  - Rolling windows are scaled down for WNBA's ~40-game season (vs NBA's 82):
    NBA's 5/10/20 become 3/6/12, keeping roughly the same season-fraction rather
    than the same absolute game count.
  - Star availability gets its own top-2-specific feature (`star_availability`),
    separate from the existing top-8 `starter_availability`. In a 12/13-team,
    shallower-roster league, a single star's absence plausibly swings a team's
    expected performance more than in the NBA's deeper-bench league; giving the
    model a feature that isolates top-2 availability (rather than diluting it
    across the whole rotation) is how that gets more weight, in practice.

All features use data strictly before `as_of_utc` — enforced by
load_team_game_stats_before() which filters scheduled_utc < as_of.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.features.common import (
    load_team_game_stats_before,
    load_team_top_player_stats,
    rolling_mean,
)

log = get_logger(__name__)

_WINDOWS = [3, 6, 12]
_SHOOTING_WINDOWS = [3, 6]
_WIN_PCT_WINDOWS = [2, 3, 6, 12]


# Approximate UTC offset from longitude (15° = 1 hour).
def _lon_to_tz_offset(lon: float) -> float:
    return round(lon / 15.0)


def build_team_features(
    session: Session,
    team_id: int,
    as_of_utc: datetime,
    elo_rating: float,
    opponent_id: int | None = None,
    is_home: bool = True,
    current_venue_lon: float | None = None,
) -> dict[str, Any]:
    """Build team features for game-winner model."""
    games = load_team_game_stats_before(session, team_id, as_of_utc, limit=40)

    feats: dict[str, Any] = {}
    feats["elo"] = elo_rating
    feats["is_home"] = int(is_home)

    if not games:
        _fill_defaults(feats)
        return feats

    # ── Extract time-series ───────────────────────────────────────────────────
    game_dates: list[datetime] = []
    off_rtgs: list[float] = []
    def_rtgs: list[float] = []
    net_rtgs: list[float] = []
    paces: list[float] = []
    won: list[int] = []
    home_flags: list[int] = []
    ts_pcts: list[float] = []
    three_pars: list[float] = []
    tov_rates: list[float] = []
    oreb_pcts: list[float] = []

    for g in games:
        stats = g["stats"] or {}
        adv = stats.get("advanced", {})
        trad = stats.get("traditional", {})

        # Advanced: efficiency + pace
        off_rtg = _sf(adv.get("offensiveRating") or adv.get("E_OFF_RATING"))
        def_rtg = _sf(adv.get("defensiveRating") or adv.get("E_DEF_RATING"))
        pace = _sf(adv.get("pace") or adv.get("PACE"))

        if off_rtg is not None and def_rtg is not None:
            off_rtgs.append(off_rtg)
            def_rtgs.append(def_rtg)
            net_rtgs.append(off_rtg - def_rtg)
        if pace is not None:
            paces.append(pace)

        # Traditional: shooting efficiency
        pts = _sf(trad.get("points") or trad.get("PTS"))
        fga = _sf(trad.get("fieldGoalsAttempted") or trad.get("FGA"))
        fg3a = _sf(trad.get("threePointersAttempted") or trad.get("FG3A") or trad.get("fg3a"))
        fta = _sf(trad.get("freeThrowsAttempted") or trad.get("FTA"))
        tov = _sf(trad.get("turnovers") or trad.get("TOV"))
        oreb = _sf(trad.get("offensiveRebounds") or trad.get("OREB"))
        dreb = _sf(trad.get("defensiveRebounds") or trad.get("DREB"))

        if pts and fga and fta and fga > 0:
            ts = pts / (2.0 * (fga + 0.44 * fta))
            ts_pcts.append(ts)
        if fg3a is not None and fga and fga > 0:
            three_pars.append(fg3a / fga)
        if tov is not None and fga and fta and (fga + 0.44 * fta + tov) > 0:
            tov_rates.append(tov / (fga + 0.44 * fta + tov))
        if oreb is not None and dreb is not None and (oreb + dreb) > 0:
            oreb_pcts.append(oreb / (oreb + dreb))

        # Outcome
        is_home_game = g["home_team_id"] == team_id
        home_flags.append(int(is_home_game))
        hs = g["home_score"] or 0
        aw = g["away_score"] or 0
        won.append(int(hs > aw) if is_home_game else int(aw > hs))
        game_dates.append(g["scheduled_utc"])

    # ── Point differential (avg margin) ──────────────────────────────────────
    margins: list[float] = []
    for g in games:
        is_home_game = g["home_team_id"] == team_id
        hs = g["home_score"] or 0
        aw = g["away_score"] or 0
        margin = (hs - aw) if is_home_game else (aw - hs)
        margins.append(float(margin))

    # ── Rolling efficiency windows ────────────────────────────────────────────
    for w in _WINDOWS:
        feats[f"off_rtg_last{w}"] = rolling_mean(off_rtgs, w) or 100.0
        feats[f"def_rtg_last{w}"] = rolling_mean(def_rtgs, w) or 100.0
        feats[f"net_rtg_last{w}"] = rolling_mean(net_rtgs, w) or 0.0
        feats[f"pace_last{w}"] = rolling_mean(paces, w) or 95.0

    # ── Shooting efficiency windows ───────────────────────────────────────────
    for w in _SHOOTING_WINDOWS:
        feats[f"ts_pct_last{w}"] = rolling_mean(ts_pcts, w) or 0.530
        feats[f"three_par_last{w}"] = rolling_mean(three_pars, w) or 0.300
        feats[f"tov_rate_last{w}"] = rolling_mean(tov_rates, w) or 0.140
        feats[f"oreb_pct_last{w}"] = rolling_mean(oreb_pcts, w) or 0.260

    for w in _WINDOWS:
        feats[f"margin_last{w}"] = rolling_mean(margins, w) or 0.0

    # ── Win% at multiple windows ──────────────────────────────────────────────
    for w in _WIN_PCT_WINDOWS:
        feats[f"win_pct_last{w}"] = rolling_mean(won, w) or 0.5
    feats["win_pct_season"] = float(np.mean(won)) if won else 0.5

    # ── Home/away splits ──────────────────────────────────────────────────────
    home_games = [w for w, h in zip(won, home_flags, strict=False) if h == 1]
    away_games = [w for w, h in zip(won, home_flags, strict=False) if h == 0]
    feats["home_win_pct"] = float(np.mean(home_games)) if home_games else 0.5
    feats["away_win_pct"] = float(np.mean(away_games)) if away_games else 0.5

    # ── Winning/losing streak ─────────────────────────────────────────────────
    feats["streak"] = _compute_streak(won)

    # ── Rest & schedule density ───────────────────────────────────────────────
    most_recent = max(game_dates)
    rest_days = (as_of_utc - most_recent).total_seconds() / 86400
    feats["rest_days"] = min(rest_days, 10.0)

    dates_desc = sorted(game_dates, reverse=True)
    feats["b2b"] = int(rest_days < 1.5)
    feats["three_in_four"] = int(_games_in_window(dates_desc, as_of_utc, days=4) >= 3)
    feats["four_in_six"] = int(_games_in_window(dates_desc, as_of_utc, days=6) >= 4)

    # ── Travel placeholder (populated by matchup builder) ────────────────────
    feats["travel_km"] = 0.0

    # ── Timezone fatigue: consecutive road games + tz change ─────────────────
    road_streak = 0
    for h in reversed(home_flags):
        if h == 0:
            road_streak += 1
        else:
            break
    feats["road_game_streak"] = road_streak

    # Timezone change vs current venue (hours)
    feats["tz_hours_change"] = 0.0
    if current_venue_lon is not None and game_dates:
        last_game = max(game_dates)
        prev_venue_lon = _get_last_venue_lon(session, team_id, last_game)
        if prev_venue_lon is not None:
            feats["tz_hours_change"] = abs(
                _lon_to_tz_offset(current_venue_lon) - _lon_to_tz_offset(prev_venue_lon)
            )

    # ── Injury-adjusted starter / star availability ───────────────────────────
    feats["starter_availability"] = _get_rotation_availability(session, team_id, as_of_utc, top_n=8)
    feats["star_availability"] = _get_rotation_availability(session, team_id, as_of_utc, top_n=2)

    # ── Roster quality: player-level efficiency aggregates ────────────────────
    feats.update(_get_roster_quality(session, team_id, as_of_utc))

    return feats


def _get_last_venue_lon(session: Session, team_id: int, before_utc: datetime) -> float | None:
    """Return the longitude of the venue for the most recent game before before_utc."""
    try:
        result = session.execute(
            text("""
                SELECT v.lon FROM games g
                JOIN venues v ON v.id = g.venue_id
                WHERE (g.home_team_id = :tid OR g.away_team_id = :tid)
                  AND g.scheduled_utc < :before
                  AND g.status = 'final'
                  AND v.lon IS NOT NULL
                ORDER BY g.scheduled_utc DESC LIMIT 1
            """),
            {"tid": team_id, "before": before_utc},
        ).first()
        return float(result.lon) if result else None
    except Exception:
        return None


def _get_roster_quality(session: Session, team_id: int, as_of_utc: datetime) -> dict[str, Any]:
    """Aggregate player-level efficiency into roster-quality signals.

    n_games=5 (vs NBA's 10) to match the shorter-season proportional window.
    """
    players = load_team_top_player_stats(session, team_id, as_of_utc, n_games=5, top_n=8)
    if not players:
        return {
            "roster_star_pts": 16.0,
            "roster_star_ts_pct": 0.530,
            "roster_depth_score": 0.530,
            "star_usage_conc": 0.50,
        }

    pts = [p["avg_pts"] or 0.0 for p in players]
    ts = [p["avg_ts_pct"] or 0.530 for p in players]
    usage = [p["avg_usage"] or 0.20 for p in players]

    # Stars = top 2 by points
    top2_pts = sorted(pts, reverse=True)[:2]
    top2_ts = [ts[i] for i in sorted(range(len(pts)), key=lambda x: pts[x], reverse=True)[:2]]
    depth_ts = [ts[i] for i in sorted(range(len(pts)), key=lambda x: pts[x], reverse=True)[2:]]

    total_usage = sum(usage) or 1.0
    star_usage = sum(sorted(usage, reverse=True)[:2])

    return {
        "roster_star_pts": float(np.mean(top2_pts)) if top2_pts else 16.0,
        "roster_star_ts_pct": float(np.mean(top2_ts)) if top2_ts else 0.530,
        "roster_depth_score": float(np.mean(depth_ts)) if depth_ts else 0.530,
        "star_usage_conc": star_usage / total_usage,
    }


def _get_rotation_availability(
    session: Session, team_id: int, as_of_utc: datetime, top_n: int
) -> float:
    """Fraction of the top-`top_n` rotation players (by games played) not out/doubtful.

    top_n=8 gives the whole-rotation signal (mirrors NBA's starter_availability);
    top_n=2 isolates just the team's two highest-usage players, since a single
    star's absence matters proportionally more in a 12/13-team league.
    """
    try:
        result = session.execute(
            text("""
                WITH top_players AS (
                    SELECT pgs.player_id, COUNT(*) AS games_played
                    FROM player_game_stats pgs
                    JOIN games g ON g.id = pgs.game_id
                    WHERE pgs.team_id = :team_id
                      AND g.scheduled_utc < :as_of
                      AND g.status = 'final'
                    GROUP BY pgs.player_id
                    ORDER BY games_played DESC
                    LIMIT :top_n
                ),
                latest_injuries AS (
                    SELECT DISTINCT ON (player_id) player_id, status
                    FROM injuries
                    WHERE player_id IN (SELECT player_id FROM top_players)
                      AND reported_at < :as_of
                    ORDER BY player_id, reported_at DESC
                )
                SELECT tp.player_id, COALESCE(li.status, 'Active') AS injury_status
                FROM top_players tp
                LEFT JOIN latest_injuries li ON li.player_id = tp.player_id
            """),
            {"team_id": team_id, "as_of": as_of_utc, "top_n": top_n},
        )
        rows = list(result)
        if not rows:
            return 1.0
        injured = sum(1 for r in rows if r.injury_status in ("Out", "Doubtful"))
        return 1.0 - (injured / len(rows))
    except Exception:
        return 1.0


def _fill_defaults(feats: dict[str, Any]) -> None:
    for w in _WINDOWS:
        for stat in ["off_rtg", "def_rtg", "net_rtg", "pace"]:
            feats[f"{stat}_last{w}"] = (
                0.0 if stat == "net_rtg" else (95.0 if stat == "pace" else 100.0)
            )
    for w in _SHOOTING_WINDOWS:
        feats[f"ts_pct_last{w}"] = 0.530
        feats[f"three_par_last{w}"] = 0.300
        feats[f"tov_rate_last{w}"] = 0.140
        feats[f"oreb_pct_last{w}"] = 0.260
    for w in _WIN_PCT_WINDOWS:
        feats[f"win_pct_last{w}"] = 0.5
    feats["win_pct_season"] = 0.5
    for w in _WINDOWS:
        feats[f"margin_last{w}"] = 0.0
    feats["home_win_pct"] = 0.5
    feats["away_win_pct"] = 0.5
    feats["streak"] = 0
    feats["rest_days"] = 2.0
    feats["b2b"] = 0
    feats["three_in_four"] = 0
    feats["four_in_six"] = 0
    feats["travel_km"] = 0.0
    feats["road_game_streak"] = 0
    feats["tz_hours_change"] = 0.0
    feats["starter_availability"] = 1.0
    feats["star_availability"] = 1.0
    feats["roster_star_pts"] = 16.0
    feats["roster_star_ts_pct"] = 0.530
    feats["roster_depth_score"] = 0.530
    feats["star_usage_conc"] = 0.50


def _sf(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _compute_streak(won: list[int]) -> int:
    """Positive = win streak, negative = losing streak (from most recent game)."""
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
