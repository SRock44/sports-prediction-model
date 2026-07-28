"""WNBA matchup-level feature assembly.

Direct structural port of src/features/nba/matchup.py (see that file's docstring
and plans/sports-integration/WNBA.md §4) with three deliberate deviations:

  - Rolling-window cross-features reference team.py's scaled windows (3/6/12
    instead of 5/10/20) to match the season-fraction NBA's windows represent.
  - `_get_venue_hca`'s min_games threshold is lowered (30 -> 12) and its default
    home-win% is WNBA's league-average (~0.57) rather than NBA's (~0.60), since
    WNBA's smaller per-arena game counts would otherwise almost never clear a
    30-game bar within a season or two of the venue existing.
  - Star availability gets its own cross-features (star_availability_diff,
    star_pts_at_risk_diff) built from team.py's top-2-specific `star_availability`
    - see team.py's docstring for why this gets isolated rather than folded into
      the existing starter_availability_diff.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.features.common import (
    compute_elo_series,
    haversine_km,
    load_game_odds,
)
from src.features.wnba.team import build_team_features

log = get_logger(__name__)


def build_matchup_features(
    session: Session,
    game: Any,
    as_of: datetime,
) -> dict[str, Any]:
    """Assemble the full matchup feature vector for a single WNBA game.

    Args:
        game: A Game ORM instance.
        as_of: The cutoff timestamp for feature computation (typically scheduled_utc - 1h).
    """
    game_id: int = game.id
    home_team_id: int = game.home_team_id
    away_team_id: int = game.away_team_id
    sport_id: int = game.sport_id

    # ── Elo ratings (computed from all games before as_of) ────────────────────
    elo_ratings = _get_elo_ratings(session, sport_id, as_of)
    home_elo = elo_ratings.get(home_team_id, 1500.0)
    away_elo = elo_ratings.get(away_team_id, 1500.0)

    # ── Venue longitude (for timezone fatigue) ────────────────────────────────
    venue_lon = _get_venue_lon(session, game_id)

    # ── Team-level features ───────────────────────────────────────────────────
    home_feats = build_team_features(
        session,
        home_team_id,
        as_of,
        home_elo,
        opponent_id=away_team_id,
        is_home=True,
        current_venue_lon=venue_lon,
    )
    away_feats = build_team_features(
        session,
        away_team_id,
        as_of,
        away_elo,
        opponent_id=home_team_id,
        is_home=False,
        current_venue_lon=venue_lon,
    )

    # Prefix team features
    matchup: dict[str, Any] = {}
    for k, v in home_feats.items():
        matchup[f"home_{k}"] = v
    for k, v in away_feats.items():
        matchup[f"away_{k}"] = v

    # ── Cross features ────────────────────────────────────────────────────────
    matchup["elo_diff"] = home_elo - away_elo
    matchup["elo_diff_with_hca"] = (home_elo + 100.0) - away_elo
    matchup["elo_home_win_prob"] = 1.0 / (1.0 + 10.0 ** (-(matchup["elo_diff_with_hca"]) / 400.0))

    matchup["net_rtg_diff_last3"] = home_feats["net_rtg_last3"] - away_feats["net_rtg_last3"]
    matchup["net_rtg_diff_last6"] = home_feats["net_rtg_last6"] - away_feats["net_rtg_last6"]
    matchup["net_rtg_diff_last12"] = home_feats["net_rtg_last12"] - away_feats["net_rtg_last12"]
    matchup["rest_diff"] = home_feats["rest_days"] - away_feats["rest_days"]

    matchup["pace_diff_last3"] = home_feats["pace_last3"] - away_feats["pace_last3"]
    matchup["pace_diff_last6"] = home_feats["pace_last6"] - away_feats["pace_last6"]
    matchup["ts_pct_diff_last3"] = home_feats["ts_pct_last3"] - away_feats["ts_pct_last3"]
    matchup["ts_pct_diff_last6"] = home_feats["ts_pct_last6"] - away_feats["ts_pct_last6"]
    matchup["tov_rate_diff_last3"] = home_feats["tov_rate_last3"] - away_feats["tov_rate_last3"]
    matchup["oreb_pct_diff_last6"] = home_feats["oreb_pct_last6"] - away_feats["oreb_pct_last6"]

    matchup["win_pct_diff_last3"] = home_feats["win_pct_last3"] - away_feats["win_pct_last3"]
    matchup["win_pct_diff_last6"] = home_feats["win_pct_last6"] - away_feats["win_pct_last6"]
    matchup["win_pct_diff_last12"] = home_feats["win_pct_last12"] - away_feats["win_pct_last12"]
    matchup["win_pct_season_diff"] = home_feats["win_pct_season"] - away_feats["win_pct_season"]

    matchup["margin_diff_last3"] = home_feats["margin_last3"] - away_feats["margin_last3"]
    matchup["margin_diff_last6"] = home_feats["margin_last6"] - away_feats["margin_last6"]
    matchup["margin_diff_last12"] = home_feats["margin_last12"] - away_feats["margin_last12"]
    matchup["streak_diff"] = home_feats["streak"] - away_feats["streak"]
    matchup["b2b_diff"] = home_feats["b2b"] - away_feats["b2b"]
    matchup["schedule_load_diff"] = (
        home_feats["three_in_four"]
        + home_feats["four_in_six"]
        - away_feats["three_in_four"]
        - away_feats["four_in_six"]
    )
    matchup["starter_avail_diff"] = (
        home_feats["starter_availability"] - away_feats["starter_availability"]
    )

    # ── Star availability cross-features ──────────────────────────────────────
    # Isolated from starter_avail_diff above: "points at risk" scales the star-
    # availability gap by how much scoring those top-2 players actually carry,
    # so a missing star on a star-heavy team registers more than on a balanced one.
    matchup["star_availability_diff"] = (
        home_feats["star_availability"] - away_feats["star_availability"]
    )
    matchup["star_pts_at_risk_diff"] = (
        home_feats["roster_star_pts"] * (1.0 - home_feats["star_availability"])
    ) - (away_feats["roster_star_pts"] * (1.0 - away_feats["star_availability"]))

    # ── Head-to-head (last 5 regular season meetings) ─────────────────────────
    # With only 12-13 teams, H2H sample sizes are more reliable per-meeting than
    # in the NBA's 30-team league (teams face each other far more often per
    # season), so the same limit=5 window carries more signal here, not less.
    h2h = _get_h2h(session, home_team_id, away_team_id, as_of, limit=5)
    matchup["h2h_home_wins"] = h2h["home_wins"]
    matchup["h2h_total"] = h2h["total"]
    matchup["h2h_home_win_pct"] = h2h["home_wins"] / max(h2h["total"], 1)

    # ── Venue travel ──────────────────────────────────────────────────────────
    venue_feats = _get_travel_features(session, game_id, home_team_id, away_team_id, as_of)
    matchup.update(venue_feats)

    # ── Roster quality cross-features ─────────────────────────────────────────
    matchup["roster_star_pts_diff"] = home_feats["roster_star_pts"] - away_feats["roster_star_pts"]
    matchup["roster_star_ts_diff"] = (
        home_feats["roster_star_ts_pct"] - away_feats["roster_star_ts_pct"]
    )
    matchup["roster_depth_diff"] = (
        home_feats["roster_depth_score"] - away_feats["roster_depth_score"]
    )
    # Away team's road fatigue relative to home rest
    matchup["road_streak_away"] = away_feats["road_game_streak"]
    matchup["tz_change_away"] = away_feats["tz_hours_change"]

    # ── Strength of schedule (avg Elo of last-5 opponents) ────────────────────
    home_sos = _get_sos(session, home_team_id, as_of, sport_id, elo_ratings)
    away_sos = _get_sos(session, away_team_id, as_of, sport_id, elo_ratings)
    matchup["home_sos_last5"] = home_sos
    matchup["away_sos_last5"] = away_sos
    matchup["sos_diff"] = home_sos - away_sos

    # ── Venue home court advantage (historical HW% at this arena) ────────────
    matchup["venue_hca"] = _get_venue_hca(session, game_id, sport_id, as_of)

    # ── Market signals (odds) ─────────────────────────────────────────────────
    odds = load_game_odds(session, game_id)
    if odds:
        matchup.update(odds)
    else:
        matchup["odds_open_implied_home"] = 0.5
        matchup["odds_close_implied_home"] = 0.5
        matchup["odds_line_move"] = 0.0
        matchup["odds_sharp_move"] = 0
        matchup["odds_open_spread"] = 0.0

    # ── Referee tendencies ────────────────────────────────────────────────────
    matchup.update(_get_referee_features(session, game_id))

    # ── Clutch performance (last 12 games, close-game win rate) ──────────────
    matchup["home_clutch_win_pct"] = _clutch_win_pct(session, home_team_id, as_of, sport_id)
    matchup["away_clutch_win_pct"] = _clutch_win_pct(session, away_team_id, as_of, sport_id)
    matchup["clutch_win_pct_diff"] = matchup["home_clutch_win_pct"] - matchup["away_clutch_win_pct"]

    # ── Opponent-adjusted ratings (ortg/drtg relative to opponent quality) ───
    home_adj = _opp_adjusted_ratings(session, home_team_id, as_of, sport_id, elo_ratings)
    away_adj = _opp_adjusted_ratings(session, away_team_id, as_of, sport_id, elo_ratings)
    matchup["home_adj_ortg"] = home_adj["adj_ortg"]
    matchup["home_adj_drtg"] = home_adj["adj_drtg"]
    matchup["away_adj_ortg"] = away_adj["adj_ortg"]
    matchup["away_adj_drtg"] = away_adj["adj_drtg"]
    matchup["adj_net_rtg_diff"] = (home_adj["adj_ortg"] - home_adj["adj_drtg"]) - (
        away_adj["adj_ortg"] - away_adj["adj_drtg"]
    )

    # ── Scheme cross-feature: pace mismatch vs defensive quality ─────────────
    matchup["pace_scheme_advantage"] = (
        (home_feats["pace_last6"] - away_feats["pace_last6"])
        * (away_feats["def_rtg_last6"] - home_feats["def_rtg_last6"])
        / 100.0
    )

    return matchup


def _clutch_win_pct(
    session: Session,
    team_id: int,
    as_of: datetime,
    sport_id: int,
    window: int = 12,
) -> float:
    """Win rate in games decided by <=5 points (proxy for clutch performance)."""
    result = session.execute(
        text("""
            SELECT
                COUNT(*) AS total,
                SUM(
                    CASE
                        WHEN (home_team_id = :tid AND home_score > away_score)
                          OR (away_team_id = :tid AND away_score > home_score)
                        THEN 1 ELSE 0
                    END
                ) AS wins
            FROM (
                SELECT home_team_id, away_team_id, home_score, away_score
                FROM games
                WHERE (home_team_id = :tid OR away_team_id = :tid)
                  AND sport_id = :sid
                  AND scheduled_utc < :as_of
                  AND status = 'final'
                  AND home_score IS NOT NULL
                  AND ABS(home_score - away_score) <= 5
                ORDER BY scheduled_utc DESC
                LIMIT :window
            ) sub
        """),
        {"tid": team_id, "sid": sport_id, "as_of": as_of, "window": window},
    ).first()
    if result and result.total and result.total >= 3:
        return float(result.wins) / float(result.total)
    return 0.5


def _opp_adjusted_ratings(
    session: Session,
    team_id: int,
    as_of: datetime,
    sport_id: int,
    elo_ratings: dict[int, float],
    window: int = 8,
) -> dict[str, float]:
    """Opponent-adjusted offensive and defensive rating.

    window=8 (vs NBA's 15) to match the shorter-season proportional scaling.
    """
    defaults = {"adj_ortg": 100.0, "adj_drtg": 100.0}
    try:
        result = session.execute(
            text("""
                SELECT
                    CASE WHEN home_team_id = :tid THEN away_team_id ELSE home_team_id END AS opp_id,
                    CASE WHEN home_team_id = :tid
                         THEN (tgs.stats->'advanced'->>'offensiveRating')::float
                         ELSE (tgs.stats->'advanced'->>'defensiveRating')::float
                    END AS ortg,
                    CASE WHEN home_team_id = :tid
                         THEN (tgs.stats->'advanced'->>'defensiveRating')::float
                         ELSE (tgs.stats->'advanced'->>'offensiveRating')::float
                    END AS drtg
                FROM games g
                JOIN team_game_stats tgs ON tgs.game_id = g.id AND tgs.team_id = :tid
                WHERE (g.home_team_id = :tid OR g.away_team_id = :tid)
                  AND g.sport_id = :sid
                  AND g.scheduled_utc < :as_of
                  AND g.status = 'final'
                  AND tgs.stats->'advanced'->>'offensiveRating' IS NOT NULL
                ORDER BY g.scheduled_utc DESC
                LIMIT :window
            """),
            {"tid": team_id, "sid": sport_id, "as_of": as_of, "window": window},
        ).fetchall()

        if not result or len(result) < 3:
            return defaults

        adj_ortg_vals, adj_drtg_vals = [], []
        for row in result:
            opp_elo = elo_ratings.get(row.opp_id, 1500.0)
            quality_multiplier = opp_elo / 1500.0
            if row.ortg is not None:
                adj_ortg_vals.append(float(row.ortg) * quality_multiplier)
            if row.drtg is not None:
                adj_drtg_vals.append(float(row.drtg) / quality_multiplier)

        return {
            "adj_ortg": float(sum(adj_ortg_vals) / len(adj_ortg_vals)) if adj_ortg_vals else 100.0,
            "adj_drtg": float(sum(adj_drtg_vals) / len(adj_drtg_vals)) if adj_drtg_vals else 100.0,
        }
    except Exception:
        return defaults


def _get_elo_ratings(session: Session, sport_id: int, as_of: datetime) -> dict[int, float]:
    """Replay Elo from all completed games before as_of."""
    import pandas as pd

    result = session.execute(
        text("""
            SELECT id, home_team_id, away_team_id, scheduled_utc,
                   CASE WHEN home_score > away_score THEN 1 ELSE 0 END as home_won
            FROM games
            WHERE sport_id = :sport_id
              AND scheduled_utc < :as_of
              AND status = 'final'
              AND home_score IS NOT NULL
            ORDER BY scheduled_utc
        """),
        {"sport_id": sport_id, "as_of": as_of},
    )
    rows = [dict(r._mapping) for r in result]
    if not rows:
        return {}

    df = pd.DataFrame(rows)
    return compute_elo_series(df)


def _get_h2h(
    session: Session,
    home_team_id: int,
    away_team_id: int,
    as_of: datetime,
    limit: int = 5,
) -> dict[str, int]:
    result = session.execute(
        text("""
            SELECT home_score, away_score, home_team_id
            FROM games
            WHERE ((home_team_id = :home AND away_team_id = :away)
                OR (home_team_id = :away AND away_team_id = :home))
              AND scheduled_utc < :as_of
              AND status = 'final'
              AND home_score IS NOT NULL
            ORDER BY scheduled_utc DESC
            LIMIT :limit
        """),
        {"home": home_team_id, "away": away_team_id, "as_of": as_of, "limit": limit},
    )
    rows = list(result)
    home_wins = 0
    for row in rows:
        if (row.home_team_id == home_team_id and row.home_score > row.away_score) or (
            row.home_team_id == away_team_id and row.away_score > row.home_score
        ):
            home_wins += 1
    return {"home_wins": home_wins, "total": len(rows)}


def _get_venue_lon(session: Session, game_id: int) -> float | None:
    """Return longitude of a game's venue."""
    result = session.execute(
        text(
            "SELECT v.lon FROM games g JOIN venues v ON v.id = g.venue_id WHERE g.id = :gid AND v.lon IS NOT NULL"
        ),
        {"gid": game_id},
    ).first()
    return float(result.lon) if result else None


def _get_referee_features(session: Session, game_id: int) -> dict[str, Any]:
    """Aggregate referee crew tendencies for this game.

    Contingent on officials data being available (plan doc §2/§13, §4) - falls
    back to defaults gracefully, same as NBA's version, if games.meta['officials']
    isn't populated yet.
    """
    defaults = {
        "ref_foul_rate": 40.0,  # avg fouls called per game (WNBA games are 4x8min quarters)
        "ref_pace_factor": 1.0,  # relative pace vs league avg
        "ref_home_win_pct": 0.5,  # how often home team wins with this crew
    }
    try:
        game_meta = session.execute(
            text("SELECT meta FROM games WHERE id = :gid"), {"gid": game_id}
        ).scalar()
        officials = (game_meta or {}).get("officials", [])
        if not officials:
            return defaults

        result = session.execute(
            text("""
                SELECT
                    AVG(CASE WHEN (tgs.stats->'traditional'->>'personalFouls')::float IS NOT NULL
                             THEN (tgs.stats->'traditional'->>'personalFouls')::float ELSE NULL END) AS avg_fouls,
                    AVG(CASE WHEN (tgs.stats->'advanced'->>'pace')::float IS NOT NULL
                             THEN (tgs.stats->'advanced'->>'pace')::float ELSE NULL END) AS avg_pace,
                    AVG(CASE WHEN g.home_score > g.away_score THEN 1.0 ELSE 0.0 END) AS home_win_pct,
                    COUNT(DISTINCT g.id) AS n_games
                FROM games g
                JOIN team_game_stats tgs ON tgs.game_id = g.id
                WHERE g.status = 'final'
                  AND g.meta->'officials' ?| :officials
            """),
            {"officials": officials},
        ).first()

        if result and result.n_games and result.n_games >= 5:
            return {
                "ref_foul_rate": float(result.avg_fouls or 40.0),
                "ref_pace_factor": (result.avg_pace or 95.0) / 95.0,
                "ref_home_win_pct": float(result.home_win_pct or 0.5),
            }
    except Exception:
        pass
    return defaults


def _get_sos(
    session: Session,
    team_id: int,
    as_of: datetime,
    sport_id: int,
    elo_ratings: dict[int, float],
    window: int = 5,
) -> float:
    """Average Elo of last `window` opponents — proxy for strength of schedule.

    window=5 (vs NBA's 10) to match the shorter-season proportional scaling.
    """
    result = session.execute(
        text("""
            SELECT CASE WHEN home_team_id = :tid THEN away_team_id ELSE home_team_id END AS opp_id
            FROM games
            WHERE (home_team_id = :tid OR away_team_id = :tid)
              AND sport_id = :sid
              AND scheduled_utc < :as_of
              AND status = 'final'
            ORDER BY scheduled_utc DESC
            LIMIT :window
        """),
        {"tid": team_id, "sid": sport_id, "as_of": as_of, "window": window},
    )
    opp_ids = [r.opp_id for r in result]
    if not opp_ids:
        return 1500.0
    return sum(elo_ratings.get(oid, 1500.0) for oid in opp_ids) / len(opp_ids)


def _get_venue_hca(
    session: Session,
    game_id: int,
    sport_id: int,
    as_of: datetime,
    min_games: int = 12,
) -> float:
    """Historical home win% at this game's venue, using only games before as_of.

    min_games=12 (vs NBA's 30) and a lower WNBA-average default (0.57 vs NBA's
    0.60) - WNBA's smaller per-arena game counts would rarely clear a 30-game
    bar within a season or two of a venue's use (plan doc §4).
    """
    result = session.execute(
        text("""
            SELECT
                AVG(CASE WHEN g.home_score > g.away_score THEN 1.0 ELSE 0.0 END) AS hw_pct,
                COUNT(*) AS n_games
            FROM games g
            JOIN games this_game ON this_game.id = :gid
            WHERE g.venue_id = this_game.venue_id
              AND g.sport_id = :sid
              AND g.status = 'final'
              AND g.home_score IS NOT NULL
              AND g.scheduled_utc < :as_of
        """),
        {"gid": game_id, "sid": sport_id, "as_of": as_of},
    ).first()
    if result and result.n_games and result.n_games >= min_games:
        return float(result.hw_pct)
    return 0.57  # WNBA league-average home win rate


def _get_travel_features(
    session: Session,
    game_id: int,
    home_team_id: int,
    away_team_id: int,
    as_of: datetime,
) -> dict[str, Any]:
    """Compute travel distance for away team's most recent road trip."""
    result = session.execute(
        text("""
            SELECT v.lat, v.lon
            FROM games g
            JOIN venues v ON v.id = g.venue_id
            WHERE (g.home_team_id = :team OR g.away_team_id = :team)
              AND g.scheduled_utc < :as_of
              AND g.status = 'final'
              AND v.lat IS NOT NULL
            ORDER BY g.scheduled_utc DESC
            LIMIT 1
        """),
        {"team": away_team_id, "as_of": as_of},
    )
    prev = result.first()

    current_venue = session.execute(
        text("""
            SELECT v.lat, v.lon
            FROM games g
            JOIN venues v ON v.id = g.venue_id
            WHERE g.id = :game_id AND v.lat IS NOT NULL
        """),
        {"game_id": game_id},
    ).first()

    if prev and current_venue and prev.lat and current_venue.lat:
        away_travel = haversine_km(prev.lat, prev.lon, current_venue.lat, current_venue.lon)
    else:
        away_travel = 0.0

    return {"away_travel_km": away_travel}
