"""Soccer team-level feature engineering (SOCCER.md §4).

Windows are 5/10/19 games, not NBA's 5/10/20 - a 38-match season makes 19 the natural half-season
mark (WNBA.md's own proportional-scaling logic for a shortened season, applied here for a
different reason: season length, not games-in-flight).

**Ordering bug avoided by construction**: `_load_team_game_history` below returns rows
most-recent-first (`ORDER BY g.scheduled_utc DESC`), matching every other sport's loader in this
repo, but `rolling_mean`/`_compute_streak` (src/features/common.py) both assume oldest-first input.
NFL fixed this locally with `reversed(games)`; NBA/MLB/WNBA still carry the bug (confirmed still
present in NHL's team.py too, while auditing this file - `_compute_streak(won)` there reads
`won[-1]` as "the current game" when `won[0]` is actually most recent, giving `_compute_streak` the
wrong sign; not fixed here, out of scope for a soccer-focused pass, but worth a dedicated fix
later). `build_team_features` below reverses immediately after loading, before any per-metric list
is built, so every downstream list here is genuinely oldest-first.

**Known gaps, not fixed here** (see plans/sports-integration/SOCCER.md for the write-up):
travel_km/tz_hours_change/manager tenure/manager change were dropped entirely (2026-07-19) rather
than shipped as always-0.0/365.0 placeholders - no per-stadium lat/lon table or manager-tenure data
source exists, and a column with ~zero variance across every row is dead weight for a tree model,
not a useful feature. PPDA could not be sourced (true PPDA needs FBref, which needs a real Chrome
browser via `soccerdata`'s selenium reader - confirmed 2026-07-19 too heavy for this container);
`pressing_actions_last10` (tackles + interceptions per match, from ESPN's boxscore) is the
substitute defensive-pressure proxy actually shipped here instead.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.features.common import rolling_mean

_WINDOWS = [5, 10, 19]
_DEFAULT_ELO_FALLBACK = 1500.0
_MIDWEEK_DAYS = {1, 2, 3}  # Tue/Wed/Thu (0=Mon)


def _sf(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_team_game_history(
    session: Session, team_id: int, as_of_utc: datetime, limit: int = 40
) -> list[dict[str, Any]]:
    """Own + opponent TeamGameStats.stats for a team's last `limit` completed games.

    Self-join for the opponent's row mirrors NHL's `_load_team_game_history` (own team_game_stats
    only has this team's own numbers - npxG-against and big-chances-conceded need the opponent's
    own-side row for the same game).
    """
    rows = session.execute(
        text("""
            WITH team_games AS (
                SELECT g.id AS game_id, g.scheduled_utc, g.meta->>'competition' AS competition,
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
                tg.game_id, tg.scheduled_utc, tg.competition, tg.is_home, tg.own_score, tg.opp_score,
                tg.opp_id,
                own_tgs.stats AS own_stats,
                opp_tgs.stats AS opp_stats
            FROM team_games tg
            LEFT JOIN team_game_stats own_tgs ON own_tgs.game_id = tg.game_id AND own_tgs.team_id = :tid
            LEFT JOIN team_game_stats opp_tgs ON opp_tgs.game_id = tg.game_id AND opp_tgs.team_id = tg.opp_id
            ORDER BY tg.scheduled_utc DESC
        """),
        {"tid": team_id, "as_of": as_of_utc, "limit": limit},
    )
    return [dict(r._mapping) for r in rows]


def _is_promoted(
    session: Session, team_id: int, competition_code: str, season_start_year: int
) -> bool:
    """True if this team has no completed game in this competition in the immediately
    preceding season - covers both a genuinely new club and one returning from relegation
    (whose older top-flight data, if any, is stale and shouldn't un-flag the cold start)."""
    prior_game = session.execute(
        text("""
            SELECT 1 FROM games g
            WHERE (g.home_team_id = :tid OR g.away_team_id = :tid)
              AND g.season = :prior_season
              AND g.meta->>'competition' = :competition
            LIMIT 1
        """),
        {"tid": team_id, "prior_season": season_start_year - 1, "competition": competition_code},
    ).first()
    return prior_game is None


def _league_standings(
    session: Session, competition_code: str, season_start_year: int, as_of_utc: datetime
) -> list[dict[str, Any]]:
    """Points/goal-difference table from every completed game so far this season, as of
    `as_of_utc` - used for league_position/points_to_safety. Sorted best-first."""
    rows = session.execute(
        text("""
            WITH results AS (
                SELECT home_team_id AS team_id, home_score AS gf, away_score AS ga,
                       CASE WHEN home_score > away_score THEN 3 WHEN home_score = away_score THEN 1 ELSE 0 END AS pts
                FROM games
                WHERE season = :season AND meta->>'competition' = :competition
                  AND status = 'final' AND scheduled_utc < :as_of
                UNION ALL
                SELECT away_team_id AS team_id, away_score AS gf, home_score AS ga,
                       CASE WHEN away_score > home_score THEN 3 WHEN away_score = home_score THEN 1 ELSE 0 END AS pts
                FROM games
                WHERE season = :season AND meta->>'competition' = :competition
                  AND status = 'final' AND scheduled_utc < :as_of
            )
            SELECT team_id, SUM(pts) AS points, SUM(gf) - SUM(ga) AS goal_diff
            FROM results
            GROUP BY team_id
            ORDER BY points DESC, goal_diff DESC
        """),
        {"season": season_start_year, "competition": competition_code, "as_of": as_of_utc},
    )
    return [dict(r._mapping) for r in rows]


def build_team_features(
    session: Session,
    team_id: int,
    as_of_utc: datetime,
    elo_rating: float,
    competition_code: str,
    season_start_year: int,
    is_home: bool = True,
    opponent_elo_by_game: dict[int, dict[int, float]] | None = None,
) -> dict[str, Any]:
    """Build soccer team features for the winner model. `is_home` is which side this team plays
    in the game being predicted, not a filter on history.

    `opponent_elo_by_game`: {game_id: {team_id: elo}} - the full Elo ratings table as it stood
    right before each historical game, from `elo.py::compute_soccer_elo_snapshots`. Powers
    `sos_opponent_elo_avg_last10` (strength-of-schedule - was named in SOCCER.md §4's prose but
    never built): a team padding its raw xG/form stats against weak opponents should be read
    differently than one doing the same against strong ones, and this is the only way to know how
    good an opponent *was* at the time they were actually played, since Elo isn't persisted
    per-game in the DB. Optional and defaults to a neutral value when not supplied (e.g. the
    live-scoring path, where recomputing a full ratings-table snapshot per historical game would
    reintroduce the O(n^2) cost `compute_soccer_elo_snapshots` exists to avoid in bulk - see that
    function's docstring).
    """
    games = list(reversed(_load_team_game_history(session, team_id, as_of_utc, limit=40)))

    feats: dict[str, Any] = {}
    feats["elo"] = elo_rating

    if not games:
        _fill_defaults(feats)
        feats["promoted_flag"] = int(
            _is_promoted(session, team_id, competition_code, season_start_year)
        )
        feats.update(
            _standings_features(session, team_id, competition_code, season_start_year, as_of_utc)
        )
        return feats

    xg_for, xg_against, npxg_for, npxg_against = [], [], [], []
    shots, shots_on_target, setpiece_share, possession = [], [], [], []
    pressing_actions, goals_for, goals_against, results, home_flags = [], [], [], [], []
    yellow_cards, red_cards, opponent_elos = [], [], []
    game_dates: list[datetime] = []

    for g in games:
        own = g["own_stats"] or {}
        opp = g["opp_stats"] or {}

        if opponent_elo_by_game is not None:
            game_ratings = opponent_elo_by_game.get(g["game_id"])
            if game_ratings is not None:
                opponent_elos.append(game_ratings.get(g["opp_id"], _DEFAULT_ELO_FALLBACK))

        xgf, xga = _sf(own.get("xg_for")), _sf(own.get("xg_against"))
        if xgf is not None:
            xg_for.append(xgf)
        if xga is not None:
            xg_against.append(xga)

        npxf, npxa = _sf(own.get("npxg_for")), _sf(opp.get("npxg_for"))
        if npxf is not None:
            npxg_for.append(npxf)
        if npxa is not None:
            npxg_against.append(npxa)

        sh, sot = _sf(own.get("shots")), _sf(own.get("shots_on_target"))
        if sh is not None:
            shots.append(sh)
        if sot is not None:
            shots_on_target.append(sot)

        sps = _sf(own.get("setpiece_xg_share"))
        if sps is not None:
            setpiece_share.append(sps)

        poss = _sf(own.get("possession_pct"))
        if poss is not None and poss > 0:
            possession.append(poss / 100.0)

        tackles, interceptions = _sf(own.get("tackles")), _sf(own.get("interceptions"))
        if tackles is not None and interceptions is not None:
            pressing_actions.append(tackles + interceptions)

        yc, rc = _sf(own.get("yellow_cards")), _sf(own.get("red_cards"))
        if yc is not None:
            yellow_cards.append(yc)
        if rc is not None:
            red_cards.append(rc)

        gf, ga = _sf(g["own_score"]), _sf(g["opp_score"])
        if gf is not None:
            goals_for.append(gf)
        if ga is not None:
            goals_against.append(ga)
        if gf is not None and ga is not None:
            results.append(1 if gf > ga else (0 if gf == ga else -1))

        home_flags.append(int(g["is_home"]))
        game_dates.append(g["scheduled_utc"])

    for w in _WINDOWS:
        feats[f"xg_for_last{w}"] = rolling_mean(xg_for, w) or 1.3
        feats[f"xg_against_last{w}"] = rolling_mean(xg_against, w) or 1.3
        feats[f"xgd_last{w}"] = feats[f"xg_for_last{w}"] - feats[f"xg_against_last{w}"]

    feats["npxg_for_last10"] = rolling_mean(npxg_for, 10) or 1.2
    feats["npxg_against_last10"] = rolling_mean(npxg_against, 10) or 1.2
    feats["xg_overperformance_last10"] = (rolling_mean(goals_for, 10) or 1.3) - (
        rolling_mean(xg_for, 10) or 1.3
    )
    feats["shots_last10"] = rolling_mean(shots, 10) or 11.0
    feats["shots_on_target_last10"] = rolling_mean(shots_on_target, 10) or 4.0
    feats["setpiece_xg_share_last10"] = rolling_mean(setpiece_share, 10) or 0.25
    feats["possession_pct_last10"] = rolling_mean(possession, 10) or 0.5
    # Strength of schedule: average Elo of the last 10 opponents *at the time they were played*
    # (not their current rating) - lets the model tell a team padding raw xG stats against weak
    # opposition apart from one doing the same against strong opposition, without needing a
    # hand-tuned adjustment formula (XGBoost's trees can learn the interaction with xgd_last10
    # directly). Falls back to a neutral 1500 when opponent_elo_by_game wasn't supplied.
    feats["sos_opponent_elo_avg_last10"] = rolling_mean(opponent_elos, 10) or _DEFAULT_ELO_FALLBACK
    feats["pressing_actions_last10"] = rolling_mean(pressing_actions, 10) or 30.0

    for w in [10, 19]:
        feats[f"goal_diff_last{w}"] = (rolling_mean(goals_for, w) or 1.3) - (
            rolling_mean(goals_against, w) or 1.3
        )
    feats["points_per_game_last10"] = (
        float(np.mean([3 if r == 1 else (1 if r == 0 else 0) for r in results[-10:]]))
        if results
        else 1.2
    )

    for w in [5, 10, 19]:
        feats[f"win_pct_last{w}"] = _rate(results, w, target=1)
    feats["win_pct_season"] = _rate(results, len(results), target=1) if results else 0.4
    feats["draw_pct_last10"] = _rate(results, 10, target=0)

    home_results = [r for r, h in zip(results, home_flags, strict=False) if h == 1]
    away_results = [r for r, h in zip(results, home_flags, strict=False) if h == 0]
    feats["venue_win_pct"] = _win_rate(home_results if is_home else away_results)
    feats["venue_draw_pct"] = _draw_rate(home_results if is_home else away_results)

    feats["streak"] = _compute_streak(results)

    most_recent = max(game_dates)
    rest_days = (as_of_utc - most_recent).total_seconds() / 86400
    feats["rest_days"] = min(rest_days, 14.0)
    feats["matches_last14d"] = _games_in_window(game_dates, as_of_utc, 14)
    feats["matches_last21d"] = _games_in_window(game_dates, as_of_utc, 21)
    feats["midweek_match_flag"] = int(
        any(
            d.weekday() in _MIDWEEK_DAYS
            for d in game_dates
            if (as_of_utc - d).total_seconds() / 86400 <= 7
        )
    )

    feats["yellow_cards_last10"] = rolling_mean(yellow_cards, 10) or 1.8
    feats["red_cards_last10"] = rolling_mean(red_cards, 10) or 0.05

    feats["promoted_flag"] = int(
        _is_promoted(session, team_id, competition_code, season_start_year)
    )
    feats.update(
        _standings_features(session, team_id, competition_code, season_start_year, as_of_utc)
    )

    return feats


def _standings_features(
    session: Session,
    team_id: int,
    competition_code: str,
    season_start_year: int,
    as_of_utc: datetime,
) -> dict[str, Any]:
    table = _league_standings(session, competition_code, season_start_year, as_of_utc)
    if not table:
        return {"league_position": 10.0, "points_to_safety": 0.0}

    positions = {row["team_id"]: i + 1 for i, row in enumerate(table)}
    points_by_team = {row["team_id"]: row["points"] for row in table}
    n_teams = len(table)
    relegation_idx = max(n_teams - 3, 0)
    safety_points = table[relegation_idx]["points"] if relegation_idx < n_teams else 0

    return {
        "league_position": float(positions.get(team_id, n_teams // 2)),
        "points_to_safety": float(points_by_team.get(team_id, 0) - safety_points),
    }


def _rate(results: list[int], window: int, target: int) -> float:
    recent = results[-window:] if window else results
    if not recent:
        return 0.4 if target == 1 else 0.26
    return float(np.mean([1 if r == target else 0 for r in recent]))


def _win_rate(results: list[int]) -> float:
    return float(np.mean([1 if r == 1 else 0 for r in results])) if results else 0.4


def _draw_rate(results: list[int]) -> float:
    return float(np.mean([1 if r == 0 else 0 for r in results])) if results else 0.26


def _fill_defaults(feats: dict[str, Any]) -> None:
    for w in _WINDOWS:
        feats[f"xg_for_last{w}"] = 1.3
        feats[f"xg_against_last{w}"] = 1.3
        feats[f"xgd_last{w}"] = 0.0
    feats["npxg_for_last10"] = 1.2
    feats["npxg_against_last10"] = 1.2
    feats["xg_overperformance_last10"] = 0.0
    feats["shots_last10"] = 11.0
    feats["shots_on_target_last10"] = 4.0
    feats["setpiece_xg_share_last10"] = 0.25
    feats["possession_pct_last10"] = 0.5
    feats["pressing_actions_last10"] = 30.0
    feats["sos_opponent_elo_avg_last10"] = _DEFAULT_ELO_FALLBACK
    for w in [10, 19]:
        feats[f"goal_diff_last{w}"] = 0.0
    feats["points_per_game_last10"] = 1.2
    for w in [5, 10, 19]:
        feats[f"win_pct_last{w}"] = 0.4
    feats["win_pct_season"] = 0.4
    feats["draw_pct_last10"] = 0.26
    feats["venue_win_pct"] = 0.4
    feats["venue_draw_pct"] = 0.26
    feats["streak"] = 0
    feats["rest_days"] = 7.0
    feats["matches_last14d"] = 0
    feats["matches_last21d"] = 0
    feats["midweek_match_flag"] = 0
    feats["yellow_cards_last10"] = 1.8
    feats["red_cards_last10"] = 0.05


def _compute_streak(results: list[int]) -> int:
    """+N for an N-game win streak, -N for an N-game loss streak, 0 if the most recent result
    was a draw (a draw ends any streak rather than extending one, unlike NHL's 2-outcome
    version)."""
    if not results:
        return 0
    last = results[-1]
    if last == 0:
        return 0
    streak = 0
    for r in reversed(results):
        if r == last:
            streak += 1
        else:
            break
    return streak * last


def _games_in_window(dates: list[datetime], as_of: datetime, days: int) -> int:
    cutoff = as_of - timedelta(days=days)
    return sum(1 for d in dates if d >= cutoff)
