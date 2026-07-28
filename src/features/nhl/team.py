"""NHL team-level feature engineering.

All features use data strictly before `as_of_utc`. Corsi/Fenwick (shot-attempt
differentials, incl. a 5v5-only cut) are computed directly from the free,
first-party NHL API play-by-play — see src/ingest/nhl/games.py's
ingest_shot_attempts docstring for why this exists instead of a MoneyPuck
integration: MoneyPuck's own server redirects automated requests to a data-
license page, an explicit access-control decision this project isn't routing
around. True shot-quality-adjusted xG (MoneyPuck's actual specialty) is still
not included — Corsi/Fenwick are volume/possession proxies, not xG's shot-
quality adjustment. A homegrown xG model (shot distance/angle/type from the
same play-by-play coordinates) is a plausible future addition, not built yet.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.features.common import haversine_km, rolling_mean

_WINDOWS = [5, 10, 20]

# Approximate home-arena coordinates, keyed by venue name (matches Venue.name
# as populated by src/ingest/nhl/games.py, which doesn't itself store lat/lon
# — see module docstring rationale). Static geographic facts, same idiom as
# MLB's hardcoded park-factor dicts. Best-effort, not guaranteed to cover
# every arena name variant the API returns — _get_travel_km degrades safely
# to 0.0 on a miss rather than raising, so an incomplete table is a quality
# gap, not a correctness bug.
_ARENA_COORDS: dict[str, tuple[float, float]] = {
    "TD Garden": (42.3662, -71.0621),
    "KeyBank Center": (42.8750, -78.8765),
    "Canadian Tire Centre": (45.2969, -75.9269),
    "Madison Square Garden": (40.7505, -73.9934),
    "Prudential Center": (40.7336, -74.1711),
    "Wells Fargo Center": (39.9012, -75.1720),
    "PPG Paints Arena": (40.4395, -79.9895),
    "Capital One Arena": (38.8981, -77.0209),
    "PNC Arena": (35.8033, -78.7219),
    "Amerant Bank Arena": (26.1584, -80.3255),
    "Amalie Arena": (27.9427, -82.4518),
    "Bell Centre": (45.4961, -73.5693),
    "Scotiabank Arena": (43.6435, -79.3791),
    "Little Caesars Arena": (42.3411, -83.0553),
    "Nationwide Arena": (39.9695, -83.0061),
    "Rogers Place": (53.5469, -113.4977),
    "UBS Arena": (40.7229, -73.5904),
    "Xcel Energy Center": (44.9448, -93.1011),
    "United Center": (41.8807, -87.6742),
    "Enterprise Center": (38.6266, -90.2026),
    "Delta Center": (40.7683, -111.9011),
    "Ball Arena": (39.7487, -105.0077),
    "T-Mobile Arena": (36.1028, -115.1785),
    "Honda Center": (33.8078, -117.8765),
    "Crypto.com Arena": (34.0430, -118.2673),
    "Climate Pledge Arena": (47.6221, -122.3540),
    "SAP Center": (37.3327, -121.9012),
    "Rogers Arena": (49.2778, -123.1089),
    "Canada Life Centre": (49.8928, -97.1436),
}


def _sf(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_team_game_history(
    session: Session, team_id: int, as_of_utc: datetime, limit: int = 40
) -> list[dict[str, Any]]:
    """Per-game aggregates: own/opp goals, own/opp shots-on-goal, PP goals,
    faceoff%. One query so every rolling feature below stays chronologically
    aligned — team_game_stats.sog only has this team's own shots, so opponent
    SOG needs a same-game self-join, and PP-goals/faceoff% need a per-game
    aggregate over player_game_stats (only skaters have these fields; goalie
    rows simply have NULL, excluded via the WHERE guards below)."""
    rows = session.execute(
        text("""
            WITH team_games AS (
                SELECT g.id AS game_id, g.scheduled_utc,
                       (g.home_team_id = :tid) AS is_home,
                       CASE WHEN g.home_team_id = :tid THEN g.home_score ELSE g.away_score END AS own_score,
                       CASE WHEN g.home_team_id = :tid THEN g.away_score ELSE g.home_score END AS opp_score,
                       CASE WHEN g.home_team_id = :tid THEN g.away_team_id ELSE g.home_team_id END AS opp_id
                FROM games g
                WHERE (g.home_team_id = :tid OR g.away_team_id = :tid)
                  AND g.scheduled_utc < :as_of
                  AND g.status = 'final'
                ORDER BY g.scheduled_utc DESC
                LIMIT :limit
            )
            SELECT
                tg.game_id, tg.scheduled_utc, tg.is_home, tg.own_score, tg.opp_score,
                (own_tgs.stats->>'sog')::float AS own_sog,
                (opp_tgs.stats->>'sog')::float AS opp_sog,
                (own_tgs.stats->>'corsi')::float AS own_corsi,
                (opp_tgs.stats->>'corsi')::float AS opp_corsi,
                (own_tgs.stats->>'corsi_5v5')::float AS own_corsi_5v5,
                (opp_tgs.stats->>'corsi_5v5')::float AS opp_corsi_5v5,
                (own_tgs.stats->>'fenwick')::float AS own_fenwick,
                (opp_tgs.stats->>'fenwick')::float AS opp_fenwick,
                (SELECT COALESCE(SUM((pgs.stats->>'powerPlayGoals')::float), 0)
                   FROM player_game_stats pgs
                  WHERE pgs.game_id = tg.game_id AND pgs.team_id = :tid) AS pp_goals,
                (SELECT AVG((pgs.stats->>'faceoffWinningPctg')::float)
                   FROM player_game_stats pgs
                  WHERE pgs.game_id = tg.game_id AND pgs.team_id = :tid
                    AND (pgs.stats->>'faceoffWinningPctg') IS NOT NULL
                    AND (pgs.stats->>'faceoffWinningPctg')::float > 0) AS faceoff_pct
            FROM team_games tg
            LEFT JOIN team_game_stats own_tgs ON own_tgs.game_id = tg.game_id AND own_tgs.team_id = :tid
            LEFT JOIN team_game_stats opp_tgs ON opp_tgs.game_id = tg.game_id AND opp_tgs.team_id = tg.opp_id
            ORDER BY tg.scheduled_utc DESC
        """),
        {"tid": team_id, "as_of": as_of_utc, "limit": limit},
    )
    return [dict(r._mapping) for r in rows]


def build_team_features(
    session: Session,
    team_id: int,
    as_of_utc: datetime,
    elo_rating: float,
    is_home: bool = True,
    current_venue_name: str | None = None,
) -> dict[str, Any]:
    """Build NHL team features for the game-winner model."""
    games = _load_team_game_history(session, team_id, as_of_utc, limit=40)

    feats: dict[str, Any] = {}
    feats["elo"] = elo_rating
    feats["is_home"] = int(is_home)

    if not games:
        _fill_defaults(feats)
        return feats

    goals_for, goals_against, sog_for, sog_against = [], [], [], []
    corsi_pcts, corsi_5v5_pcts, fenwick_pcts = [], [], []
    pp_goals, faceoff_pcts, won, home_flags = [], [], [], []
    game_dates: list[datetime] = []

    for g in games:
        gf, ga = _sf(g["own_score"]), _sf(g["opp_score"])
        if gf is not None:
            goals_for.append(gf)
        if ga is not None:
            goals_against.append(ga)
        if gf is not None and ga is not None:
            won.append(int(gf > ga))

        own_sog, opp_sog = _sf(g["own_sog"]), _sf(g["opp_sog"])
        if own_sog is not None:
            sog_for.append(own_sog)
        if opp_sog is not None:
            sog_against.append(opp_sog)

        # Corsi/Fenwick %: share of shot attempts a team controlled — the
        # standard hockey-analytics "possession" metric, and this sport's
        # closest analog to MLB's wOBA / NFL's EPA (see NHL.md §4). 5v5 Corsi%
        # is the gold-standard cut (excludes PP/PK score-effect contamination);
        # all-situations Fenwick% is kept as a secondary volume-quality signal.
        own_c, opp_c = _sf(g["own_corsi"]), _sf(g["opp_corsi"])
        if own_c is not None and opp_c is not None and (own_c + opp_c) > 0:
            corsi_pcts.append(own_c / (own_c + opp_c))
        own_c5, opp_c5 = _sf(g["own_corsi_5v5"]), _sf(g["opp_corsi_5v5"])
        if own_c5 is not None and opp_c5 is not None and (own_c5 + opp_c5) > 0:
            corsi_5v5_pcts.append(own_c5 / (own_c5 + opp_c5))
        own_fw, opp_fw = _sf(g["own_fenwick"]), _sf(g["opp_fenwick"])
        if own_fw is not None and opp_fw is not None and (own_fw + opp_fw) > 0:
            fenwick_pcts.append(own_fw / (own_fw + opp_fw))

        pp_goals.append(_sf(g["pp_goals"]) or 0.0)
        fo = _sf(g["faceoff_pct"])
        if fo is not None:
            faceoff_pcts.append(fo)

        home_flags.append(int(g["is_home"]))
        game_dates.append(g["scheduled_utc"])

    for w in _WINDOWS:
        feats[f"goals_for_last{w}"] = rolling_mean(goals_for, w) or 2.9
        feats[f"goals_against_last{w}"] = rolling_mean(goals_against, w) or 2.9
        feats[f"goal_diff_last{w}"] = feats[f"goals_for_last{w}"] - feats[f"goals_against_last{w}"]
        feats[f"sog_for_last{w}"] = rolling_mean(sog_for, w) or 30.0
        feats[f"sog_against_last{w}"] = rolling_mean(sog_against, w) or 30.0
        feats[f"pp_goals_last{w}"] = rolling_mean(pp_goals, w) or 0.5

    for w in [10, 20]:
        feats[f"faceoff_pct_last{w}"] = rolling_mean(faceoff_pcts, w) or 0.500
        feats[f"corsi_pct_last{w}"] = rolling_mean(corsi_pcts, w) or 0.500
        feats[f"corsi_5v5_pct_last{w}"] = rolling_mean(corsi_5v5_pcts, w) or 0.500
        feats[f"fenwick_pct_last{w}"] = rolling_mean(fenwick_pcts, w) or 0.500

    for w in [3, 5, 10, 20]:
        feats[f"win_pct_last{w}"] = rolling_mean(won, w) or 0.5
    feats["win_pct_season"] = float(np.mean(won)) if won else 0.5

    home_games = [w for w, h in zip(won, home_flags, strict=False) if h == 1]
    away_games = [w for w, h in zip(won, home_flags, strict=False) if h == 0]
    feats["home_win_pct"] = float(np.mean(home_games)) if home_games else 0.5
    feats["away_win_pct"] = float(np.mean(away_games)) if away_games else 0.5

    feats["streak"] = _compute_streak(won)

    most_recent = max(game_dates)
    rest_days = (as_of_utc - most_recent).total_seconds() / 86400
    feats["rest_days"] = min(rest_days, 10.0)

    dates_desc = sorted(game_dates, reverse=True)
    feats["b2b"] = int(rest_days < 1.5)
    feats["three_in_four"] = int(_games_in_window(dates_desc, as_of_utc, days=4) >= 3)
    feats["four_in_six"] = int(_games_in_window(dates_desc, as_of_utc, days=6) >= 4)

    feats["travel_km"] = _get_travel_km(session, team_id, as_of_utc, current_venue_name)

    feats["starter_availability"] = _get_starter_availability(session, team_id, as_of_utc)

    return feats


def _get_travel_km(
    session: Session, team_id: int, as_of_utc: datetime, current_venue_name: str | None
) -> float:
    if not current_venue_name or current_venue_name not in _ARENA_COORDS:
        return 0.0
    dest_lat, dest_lon = _ARENA_COORDS[current_venue_name]

    try:
        prev_venue = session.execute(
            text("""
                SELECT v.name FROM games g
                JOIN venues v ON v.id = g.venue_id
                WHERE (g.home_team_id = :tid OR g.away_team_id = :tid)
                  AND g.scheduled_utc < :as_of
                  AND g.status = 'final'
                ORDER BY g.scheduled_utc DESC LIMIT 1
            """),
            {"tid": team_id, "as_of": as_of_utc},
        ).first()
    except Exception:
        return 0.0

    if not prev_venue or prev_venue.name not in _ARENA_COORDS:
        return 0.0
    prev_lat, prev_lon = _ARENA_COORDS[prev_venue.name]
    return haversine_km(prev_lat, prev_lon, dest_lat, dest_lon)


def _get_starter_availability(session: Session, team_id: int, as_of_utc: datetime) -> float:
    """Fraction of the team's top-12 recently-used skaters not ruled out/doubtful.

    Same pattern as NBA's rotation-availability check; NHL's 4-line rotation
    is deeper than NBA's, hence top-12 rather than top-8."""
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
                    LIMIT 12
                ),
                latest_injuries AS (
                    SELECT DISTINCT ON (player_id) player_id, status
                    FROM injuries
                    WHERE player_id IN (SELECT player_id FROM top_players)
                      AND reported_at < :as_of
                    ORDER BY player_id, reported_at DESC
                )
                SELECT tp.player_id, COALESCE(li.status, 'active') AS injury_status
                FROM top_players tp
                LEFT JOIN latest_injuries li ON li.player_id = tp.player_id
            """),
            {"team_id": team_id, "as_of": as_of_utc},
        )
        rows = list(result)
        if not rows:
            return 1.0
        injured = sum(1 for r in rows if r.injury_status in ("out", "doubtful"))
        return 1.0 - (injured / len(rows))
    except Exception:
        return 1.0


def _fill_defaults(feats: dict[str, Any]) -> None:
    for w in _WINDOWS:
        feats[f"goals_for_last{w}"] = 2.9
        feats[f"goals_against_last{w}"] = 2.9
        feats[f"goal_diff_last{w}"] = 0.0
        feats[f"sog_for_last{w}"] = 30.0
        feats[f"sog_against_last{w}"] = 30.0
        feats[f"pp_goals_last{w}"] = 0.5
    for w in [10, 20]:
        feats[f"faceoff_pct_last{w}"] = 0.500
        feats[f"corsi_pct_last{w}"] = 0.500
        feats[f"corsi_5v5_pct_last{w}"] = 0.500
        feats[f"fenwick_pct_last{w}"] = 0.500
    for w in [3, 5, 10, 20]:
        feats[f"win_pct_last{w}"] = 0.5
    feats["win_pct_season"] = 0.5
    feats["home_win_pct"] = 0.5
    feats["away_win_pct"] = 0.5
    feats["streak"] = 0
    feats["rest_days"] = 2.0
    feats["b2b"] = 0
    feats["three_in_four"] = 0
    feats["four_in_six"] = 0
    feats["travel_km"] = 0.0
    feats["starter_availability"] = 1.0


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
