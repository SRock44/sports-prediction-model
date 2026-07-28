"""Soccer matchup feature assembly: Elo, team features, cross-features, H2H, league context.

`ref_home_win_pct`/`ref_card_rate` are real, computed from Football-Data.co.uk's ingested referee
name + card counts (`ingest_football_data_co_uk`, `Game.meta['referee']`) - not placeholders.
`ref_penalty_rate` was dropped (2026-07-19): Football-Data.co.uk's CSV has no penalties-awarded
column, so it never had a real data source and always emitted the same constant. Market odds
default to Elo-implied probabilities when a game has no `game_odds` row (older seasons before
Football-Data.co.uk coverage, or a genuine gap).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.features.common import load_game_odds
from src.features.soccer.elo import compute_soccer_elo_series, elo_implied_probs
from src.features.soccer.team import _is_promoted, build_team_features

log = get_logger(__name__)


def build_matchup_features(
    session: Session,
    game: Any,
    as_of: datetime,
    elo_override: tuple[float, float] | None = None,
    opponent_elo_by_game: dict[int, dict[int, float]] | None = None,
) -> dict[str, Any]:
    """Full soccer game feature vector.

    Args:
        game: A Game ORM instance. `game.meta["competition"]` must be set (SOCCER.md §3).
        as_of: Cutoff timestamp for feature computation (typically scheduled_utc - 1h).
        elo_override: precomputed `(home_elo, away_elo)`, skipping the full-history DB
            recompute below. Live scoring (one game at a time) doesn't need this - the recompute
            is a single query. Bulk training-data assembly over ~20k historical games does:
            recomputing the *entire* Elo history from scratch on every call would be O(n^2)
            across the corpus. Callers doing bulk assembly should precompute every game's
            snapshot once via `elo.py::compute_soccer_elo_snapshots` and pass it in here.
        opponent_elo_by_game: the second element `compute_soccer_elo_snapshots` returns - powers
            `sos_opponent_elo_avg_last10` (team.py). Optional; omitted on the live-scoring path.
    """
    game_id: int = game.id
    home_team_id: int = game.home_team_id
    away_team_id: int = game.away_team_id
    sport_id: int = game.sport_id
    competition_code: str = game.meta["competition"]
    season_start_year: int = game.season

    if elo_override is not None:
        home_elo, away_elo = elo_override
    else:
        # ── Elo (own soccer engine - per-league HCA, goal-margin K, draws=0.5) ──
        rows = session.execute(
            text("""
                SELECT id, home_team_id, away_team_id, scheduled_utc, home_score, away_score,
                       meta->>'competition' AS competition
                FROM games
                WHERE sport_id = :sid AND scheduled_utc < :as_of AND status = 'final'
                  AND home_score IS NOT NULL
                ORDER BY scheduled_utc
            """),
            {"sid": sport_id, "as_of": as_of},
        )
        history_rows = [dict(r._mapping) for r in rows]
        promoted_ids: set[int] = set()
        if history_rows:
            df = pd.DataFrame(history_rows)
            team_ids = set(df["home_team_id"]) | set(df["away_team_id"])
            for tid in team_ids:
                if _is_promoted(session, tid, competition_code, season_start_year):
                    promoted_ids.add(tid)
            elo_ratings = compute_soccer_elo_series(df, promoted_team_ids=promoted_ids)
        else:
            elo_ratings = {}

        home_elo = elo_ratings.get(home_team_id, 1500.0)
        away_elo = elo_ratings.get(away_team_id, 1500.0)

    # ── Team-level features ───────────────────────────────────────────────────
    home_feats = build_team_features(
        session,
        home_team_id,
        as_of,
        home_elo,
        competition_code,
        season_start_year,
        is_home=True,
        opponent_elo_by_game=opponent_elo_by_game,
    )
    away_feats = build_team_features(
        session,
        away_team_id,
        as_of,
        away_elo,
        competition_code,
        season_start_year,
        is_home=False,
        opponent_elo_by_game=opponent_elo_by_game,
    )

    matchup: dict[str, Any] = {"competition_code": competition_code}
    for k, v in home_feats.items():
        matchup[f"home_{k}"] = v
    for k, v in away_feats.items():
        matchup[f"away_{k}"] = v

    # ── League context (as-of-safe) ───────────────────────────────────────────
    matchup.update(_league_context(session, competition_code, season_start_year, as_of))

    # ── Elo cross-features ────────────────────────────────────────────────────
    matchup["elo_diff"] = home_elo - away_elo
    p_home, p_draw, p_away = elo_implied_probs(home_elo, away_elo, competition_code)
    matchup["elo_home_win_prob"] = p_home
    matchup["elo_draw_prob"] = p_draw
    matchup["elo_away_win_prob"] = p_away

    # ── Style / form cross-features ───────────────────────────────────────────
    matchup["xgd_diff_last5"] = home_feats["xgd_last5"] - away_feats["xgd_last5"]
    matchup["xgd_diff_last10"] = home_feats["xgd_last10"] - away_feats["xgd_last10"]
    matchup["sos_diff_last10"] = (
        home_feats["sos_opponent_elo_avg_last10"] - away_feats["sos_opponent_elo_avg_last10"]
    )
    matchup["possession_diff_last10"] = (
        home_feats["possession_pct_last10"] - away_feats["possession_pct_last10"]
    )
    # Product of possession-style gap and pressing-intensity gap - a high-possession side against
    # a heavy-pressing side is a real, well-documented style interaction (NBA's
    # pace_scheme_advantage is the structural template: style-diff x quality-diff, not two
    # independent features). True PPDA wasn't sourceable (see team.py's module docstring); this
    # uses the pressing_actions proxy instead.
    matchup["style_matchup_advantage"] = (
        (home_feats["possession_pct_last10"] - away_feats["possession_pct_last10"])
        * (away_feats["pressing_actions_last10"] - home_feats["pressing_actions_last10"])
        / 100.0
    )
    matchup["win_pct_diff_last10"] = home_feats["win_pct_last10"] - away_feats["win_pct_last10"]
    matchup["rest_diff"] = home_feats["rest_days"] - away_feats["rest_days"]

    # ── Motivation / stakes (speculative - first candidate to cut, SOCCER.md §4) ─
    matchup["matchday_number"] = _matchday_number(
        session, home_team_id, competition_code, season_start_year, as_of
    )

    # ── Head-to-head (any competition - squads/managers turn over, weighted low) ─
    h2h = _get_h2h(session, home_team_id, away_team_id, as_of, limit=10)
    matchup["h2h_home_wins"] = h2h["home_wins"]
    matchup["h2h_draws"] = h2h["draws"]
    matchup["h2h_away_wins"] = h2h["away_wins"]
    matchup["h2h_home_win_pct"] = h2h["home_wins"] / max(h2h["total"], 1)

    # ── Referee tendencies (real, from Football-Data.co.uk's Game.meta['referee']) ──
    matchup.update(_referee_features(session, sport_id, game.meta.get("referee"), as_of))

    # ── Market odds (deferred - Football-Data.co.uk not wired into ingestion) ──
    odds = load_game_odds(session, game_id)
    if odds and "odds_open_implied_home" in odds:
        matchup["odds_open_implied_home"] = odds["odds_open_implied_home"]
        matchup["odds_open_implied_draw"] = odds.get("odds_open_implied_draw", 1.0 / 3)
        matchup["odds_open_implied_away"] = odds.get(
            "odds_open_implied_away", 1.0 - odds["odds_open_implied_home"]
        )
        matchup["odds_close_implied_home"] = odds.get(
            "odds_close_implied_home", odds["odds_open_implied_home"]
        )
        matchup["odds_close_implied_draw"] = odds.get("odds_close_implied_draw", 1.0 / 3)
        matchup["odds_close_implied_away"] = odds.get(
            "odds_close_implied_away", 1.0 - matchup["odds_close_implied_home"]
        )
        matchup["odds_line_move_home"] = (
            matchup["odds_close_implied_home"] - matchup["odds_open_implied_home"]
        )
    else:
        matchup["odds_open_implied_home"] = p_home
        matchup["odds_open_implied_draw"] = p_draw
        matchup["odds_open_implied_away"] = p_away
        matchup["odds_close_implied_home"] = p_home
        matchup["odds_close_implied_draw"] = p_draw
        matchup["odds_close_implied_away"] = p_away
        matchup["odds_line_move_home"] = 0.0

    return matchup


def _league_context(
    session: Session, competition_code: str, season_start_year: int, as_of: datetime
) -> dict[str, float]:
    row = session.execute(
        text("""
            SELECT
                AVG(home_score + away_score) AS avg_goals,
                AVG(CASE WHEN home_score > away_score THEN 1.0 ELSE 0.0 END) AS home_win_rate,
                AVG(CASE WHEN home_score = away_score THEN 1.0 ELSE 0.0 END) AS draw_rate
            FROM games
            WHERE season = :season AND meta->>'competition' = :competition
              AND status = 'final' AND scheduled_utc < :as_of
        """),
        {"season": season_start_year, "competition": competition_code, "as_of": as_of},
    ).first()
    if row is None or row.avg_goals is None:
        return {
            "league_avg_goals_per_match": 2.7,
            "league_avg_home_win_rate": 0.44,
            "league_avg_draw_rate": 0.25,
        }
    return {
        "league_avg_goals_per_match": float(row.avg_goals),
        "league_avg_home_win_rate": float(row.home_win_rate),
        "league_avg_draw_rate": float(row.draw_rate),
    }


def _referee_features(
    session: Session, sport_id: int, referee: str | None, as_of: datetime
) -> dict[str, float]:
    """Real, as-of-safe referee tendencies from Football-Data.co.uk's ingested referee name +
    card counts (`ingest_football_data_co_uk`). Falls back to neutral defaults when the referee
    is unknown (no Football-Data.co.uk coverage for this game/season) or has too few prior
    officiated games in our data to be a reliable signal - same graceful-degrade shape as every
    other cold-start default in this module, not a hard failure.
    """
    defaults = {"ref_home_win_pct": 0.44, "ref_card_rate": 4.0}
    if not referee:
        return defaults

    rows = session.execute(
        text("""
            SELECT g.home_score, g.away_score,
                   COALESCE(
                       (SELECT SUM((tgs.stats->>'yellow_cards')::float)
                             + SUM((tgs.stats->>'red_cards')::float)
                        FROM team_game_stats tgs WHERE tgs.game_id = g.id
                          AND tgs.stats ? 'yellow_cards'),
                       0
                   ) AS total_cards
            FROM games g
            WHERE g.sport_id = :sid AND g.meta->>'referee' = :referee
              AND g.status = 'final' AND g.scheduled_utc < :as_of
        """),
        {"sid": sport_id, "referee": referee, "as_of": as_of},
    )
    prior = list(rows)
    if len(prior) < 5:
        return defaults

    home_wins = sum(1 for r in prior if r.home_score > r.away_score)
    return {
        "ref_home_win_pct": home_wins / len(prior),
        "ref_card_rate": float(sum(r.total_cards for r in prior)) / len(prior),
    }


def _matchday_number(
    session: Session,
    home_team_id: int,
    competition_code: str,
    season_start_year: int,
    as_of: datetime,
) -> int:
    played = session.execute(
        text("""
            SELECT COUNT(*) FROM games
            WHERE (home_team_id = :tid OR away_team_id = :tid)
              AND season = :season AND meta->>'competition' = :competition
              AND status = 'final' AND scheduled_utc < :as_of
        """),
        {
            "tid": home_team_id,
            "season": season_start_year,
            "competition": competition_code,
            "as_of": as_of,
        },
    ).scalar()
    return int(played or 0) + 1


def _get_h2h(
    session: Session, home_id: int, away_id: int, as_of: datetime, limit: int
) -> dict[str, int]:
    result = session.execute(
        text("""
            SELECT home_score, away_score, home_team_id
            FROM games
            WHERE ((home_team_id = :h AND away_team_id = :a) OR (home_team_id = :a AND away_team_id = :h))
              AND scheduled_utc < :as_of AND status = 'final' AND home_score IS NOT NULL
            ORDER BY scheduled_utc DESC LIMIT :limit
        """),
        {"h": home_id, "a": away_id, "as_of": as_of, "limit": limit},
    )
    rows = list(result)
    home_wins = draws = away_wins = 0
    for r in rows:
        this_home_scored, this_away_scored = (
            (r.home_score, r.away_score)
            if r.home_team_id == home_id
            else (r.away_score, r.home_score)
        )
        if this_home_scored > this_away_scored:
            home_wins += 1
        elif this_home_scored == this_away_scored:
            draws += 1
        else:
            away_wins += 1
    return {"home_wins": home_wins, "draws": draws, "away_wins": away_wins, "total": len(rows)}
