"""NHL matchup feature assembly: combines team features, goalie confirmation,
Elo, H2H, division rivalry, and market odds.

Winner-target definition (plans/sports-integration/NHL.md §5): a regulation,
OT, or shootout win all count identically as "won the game" — this module
only supplies features, the target itself is defined at training time
(src/models/configs/nhl_winner.py), but it's noted here since a reader might
otherwise expect an OT/SO flag to change how any of these features are built.
It doesn't; Game.meta['last_period_type'] is metadata only (see
src/ingest/nhl/games.py's _upsert_game).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.features.common import compute_elo_series, load_game_odds
from src.features.nhl.team import build_team_features


def build_matchup_features(
    session: Session,
    game: Any,
    as_of: datetime,
) -> dict[str, Any]:
    """Full NHL game feature vector.

    Args:
        game: A Game ORM instance.
        as_of: Cutoff timestamp for feature computation (typically scheduled_utc - 1h).
    """
    import pandas as pd

    game_id: int = game.id
    home_team_id: int = game.home_team_id
    away_team_id: int = game.away_team_id
    sport_id: int = game.sport_id

    # ── Elo ───────────────────────────────────────────────────────────────────
    result = session.execute(
        text("""
            SELECT id, home_team_id, away_team_id, scheduled_utc,
                   CASE WHEN home_score > away_score THEN 1 ELSE 0 END as home_won
            FROM games
            WHERE sport_id = :sid AND scheduled_utc < :as_of AND status='final' AND home_score IS NOT NULL
            ORDER BY scheduled_utc
        """),
        {"sid": sport_id, "as_of": as_of},
    )
    rows = [dict(r._mapping) for r in result]
    if rows:
        df = pd.DataFrame(rows)
        elo_ratings = compute_elo_series(df)
    else:
        elo_ratings = {}

    home_elo = elo_ratings.get(home_team_id, 1500.0)
    away_elo = elo_ratings.get(away_team_id, 1500.0)

    # ── Venue name (for travel_km) ────────────────────────────────────────────
    venue_row = session.execute(
        text("SELECT v.name FROM games g JOIN venues v ON v.id=g.venue_id WHERE g.id=:gid"),
        {"gid": game_id},
    ).first()
    venue_name = venue_row.name if venue_row else None

    # ── Team features ─────────────────────────────────────────────────────────
    home_feats = build_team_features(
        session, home_team_id, as_of, home_elo, is_home=True, current_venue_name=venue_name
    )
    away_feats = build_team_features(
        session, away_team_id, as_of, away_elo, is_home=False, current_venue_name=venue_name
    )

    matchup: dict[str, Any] = {}
    for k, v in home_feats.items():
        matchup[f"home_{k}"] = v
    for k, v in away_feats.items():
        matchup[f"away_{k}"] = v

    # ── Cross features ────────────────────────────────────────────────────────
    matchup["elo_diff"] = home_elo - away_elo
    matchup["elo_home_win_prob"] = 1.0 / (1.0 + 10.0 ** (-(home_elo + 50.0 - away_elo) / 400.0))
    matchup["goal_diff_diff_last10"] = (
        home_feats["goal_diff_last10"] - away_feats["goal_diff_last10"]
    )
    matchup["sog_diff_last10"] = home_feats["sog_for_last10"] - away_feats["sog_for_last10"]
    matchup["rest_diff"] = home_feats["rest_days"] - away_feats["rest_days"]
    matchup["win_pct_diff_last10"] = home_feats["win_pct_last10"] - away_feats["win_pct_last10"]
    matchup["win_pct_season_diff"] = home_feats["win_pct_season"] - away_feats["win_pct_season"]
    matchup["streak_diff"] = home_feats["streak"] - away_feats["streak"]
    matchup["pp_goals_diff_last10"] = home_feats["pp_goals_last10"] - away_feats["pp_goals_last10"]
    matchup["faceoff_pct_diff"] = (
        home_feats["faceoff_pct_last10"] - away_feats["faceoff_pct_last10"]
    )
    matchup["corsi_pct_diff"] = home_feats["corsi_pct_last10"] - away_feats["corsi_pct_last10"]
    matchup["corsi_5v5_pct_diff"] = (
        home_feats["corsi_5v5_pct_last10"] - away_feats["corsi_5v5_pct_last10"]
    )
    matchup["fenwick_pct_diff"] = (
        home_feats["fenwick_pct_last10"] - away_feats["fenwick_pct_last10"]
    )

    # ── Goalie matchup — the dominant NHL signal (see NHL.md §2/§4) ──────────
    home_goalie = _get_confirmed_goalie(session, game_id, home_team_id, as_of)
    away_goalie = _get_confirmed_goalie(session, game_id, away_team_id, as_of)
    matchup.update(_goalie_features(home_goalie, prefix="home_goalie"))
    matchup.update(_goalie_features(away_goalie, prefix="away_goalie"))

    home_goalie_id = home_goalie.get("_internal_player_id") if home_goalie else None
    away_goalie_id = away_goalie.get("_internal_player_id") if away_goalie else None
    matchup.update(_goalie_rolling_form(session, home_goalie_id, as_of, prefix="home_goalie"))
    matchup.update(_goalie_rolling_form(session, away_goalie_id, as_of, prefix="away_goalie"))
    matchup["goalie_form_save_pct_diff"] = matchup.get(
        "home_goalie_form_save_pct", 0.900
    ) - matchup.get("away_goalie_form_save_pct", 0.900)
    matchup["goalie_rest_diff"] = matchup.get("home_goalie_rest_days", 3.0) - matchup.get(
        "away_goalie_rest_days", 3.0
    )

    # ── Division rivalry ───────────────────────────────────────────────────────
    divisions = session.execute(
        text("SELECT id, division FROM teams WHERE id IN (:h, :a)"),
        {"h": home_team_id, "a": away_team_id},
    ).fetchall()
    div_by_id = {r.id: r.division for r in divisions}
    matchup["division_rivalry"] = int(
        bool(div_by_id.get(home_team_id))
        and div_by_id.get(home_team_id) == div_by_id.get(away_team_id)
    )

    # ── Head-to-head ──────────────────────────────────────────────────────────
    h2h = _get_h2h(session, home_team_id, away_team_id, as_of, 5)
    matchup["h2h_home_win_pct"] = h2h["home_wins"] / max(h2h["total"], 1)
    matchup["h2h_total"] = h2h["total"]

    # ── Referee tendencies ─────────────────────────────────────────────────────
    # Deferred — Game.meta['officials'] isn't populated by ingestion yet (the
    # landing endpoint has this data; games.py doesn't fetch it). Safe default
    # only, same graceful-fallback shape as the real feature it'll become.
    matchup["ref_home_bias"] = 0.0
    matchup["ref_penalty_rate"] = 0.30

    # ── Market signals (odds) ─────────────────────────────────────────────────
    odds = load_game_odds(session, game_id)
    if odds:
        matchup.update(odds)
    else:
        matchup["odds_open_implied_home"] = 0.5
        matchup["odds_close_implied_home"] = 0.5
        matchup["odds_line_move"] = 0.0
        matchup["odds_sharp_move"] = 0

    return matchup


def _get_confirmed_goalie(
    session: Session, game_id: int, team_id: int, as_of: datetime
) -> dict[str, Any] | None:
    """Look up confirmed starting goalie, with box-score fallback.

    Tries the lineups table first (source='confirmed_goalie', written by
    src/ingest/nhl/players.py's sync_confirmed_goalie_lineup). For historical
    games where that patch never ran, falls back to the box score's explicit
    `starter: true` flag (see players.py's get_confirmed_goalies — a cleaner
    signal than MLB's innings-pitched proxy fallback, since the NHL API
    exposes starter status directly)."""
    row = session.execute(
        text("""
            SELECT players FROM lineups
            WHERE game_id = :gid AND team_id = :tid AND source = 'confirmed_goalie'
            ORDER BY fetched_at DESC LIMIT 1
        """),
        {"gid": game_id, "tid": team_id},
    ).first()
    if row and row.players:
        goalie = dict(row.players[0])
        # Lineup rows carry the raw box-score dict (external "playerId" only,
        # from players.py's get_confirmed_goalies) — resolve the internal
        # players.id the same way the box-score-fallback path below does,
        # since _goalie_rolling_form's query needs the internal FK, not the
        # external NHL API id.
        ext_id = str(goalie.get("playerId", ""))
        if ext_id:
            internal = session.execute(
                text("""
                    SELECT p.id FROM players p
                    JOIN teams t ON t.sport_id = p.sport_id
                    WHERE t.id = :tid AND p.external_id = :ext_id
                """),
                {"tid": team_id, "ext_id": ext_id},
            ).first()
            if internal:
                goalie["_internal_player_id"] = internal.id
        return goalie

    result = session.execute(
        text("""
            SELECT pgs.stats, pgs.player_id AS internal_player_id
            FROM player_game_stats pgs
            WHERE pgs.game_id = :gid
              AND pgs.team_id = :tid
              AND pgs.stats->>'role' = 'goalies'
              AND (pgs.stats->>'starter')::boolean IS TRUE
            LIMIT 1
        """),
        {"gid": game_id, "tid": team_id},
    ).first()
    if result is None:
        return None
    goalie = dict(result.stats)
    goalie["_internal_player_id"] = result.internal_player_id
    return goalie


def _goalie_features(goalie: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    if goalie is None:
        return {f"{prefix}_known": 0}
    return {f"{prefix}_known": 1}


def _goalie_rolling_form(
    session: Session,
    player_id: int | str | None,
    as_of: datetime,
    prefix: str,
    n_starts: int = 5,
) -> dict[str, Any]:
    """Last N starts save% and GAA for a goalie, plus workload (starts in
    last 7 days — back-to-back goalie starts are a real performance dip
    signal per NHL.md §4)."""
    defaults = {
        f"{prefix}_form_save_pct": 0.900,
        f"{prefix}_form_gaa": 2.90,
        f"{prefix}_form_known": 0,
        f"{prefix}_rest_days": 3.0,
        f"{prefix}_starts_last7d": 0,
    }
    if player_id is None:
        return defaults
    try:
        result = session.execute(
            text("""
                SELECT pgs.stats AS stats, g.scheduled_utc AS game_date
                FROM player_game_stats pgs
                JOIN games g ON g.id = pgs.game_id
                WHERE pgs.player_id = :pid
                  AND g.scheduled_utc < :as_of
                  AND g.status = 'final'
                  AND pgs.stats->>'role' = 'goalies'
                ORDER BY g.scheduled_utc DESC
                LIMIT :n
            """),
            {"pid": player_id, "as_of": as_of, "n": n_starts},
        )
        raw_rows = list(result)
        if not raw_rows:
            return defaults

        from datetime import UTC as _UTC
        from datetime import timedelta

        last_start = raw_rows[0].game_date
        if last_start.tzinfo is None:
            last_start = last_start.replace(tzinfo=_UTC)
        as_of_aware = as_of if as_of.tzinfo else as_of.replace(tzinfo=_UTC)
        rest_days = min((as_of_aware - last_start).total_seconds() / 86400, 10.0)

        recent_week = session.execute(
            text("""
                SELECT COUNT(*) AS n FROM player_game_stats pgs
                JOIN games g ON g.id = pgs.game_id
                WHERE pgs.player_id = :pid AND g.scheduled_utc < :as_of
                  AND g.scheduled_utc >= :since AND g.status = 'final'
                  AND pgs.stats->>'role' = 'goalies'
            """),
            {"pid": player_id, "as_of": as_of, "since": as_of - timedelta(days=7)},
        ).scalar()

        save_pcts, gaas = [], []
        for r in raw_rows:
            stats = r.stats
            sv_pct = stats.get("savePctg")
            if sv_pct is not None:
                save_pcts.append(float(sv_pct))
            ga = stats.get("goalsAgainst")
            toi = stats.get("toi")
            if ga is not None and toi:
                minutes = _toi_to_minutes(toi)
                if minutes > 0:
                    gaas.append(float(ga) * 60.0 / minutes)

        from src.features.common import rolling_mean

        return {
            f"{prefix}_form_save_pct": rolling_mean(save_pcts, n_starts) or 0.900,
            f"{prefix}_form_gaa": rolling_mean(gaas, n_starts) or 2.90,
            f"{prefix}_form_known": int(bool(save_pcts)),
            f"{prefix}_rest_days": rest_days,
            f"{prefix}_starts_last7d": int(recent_week or 0),
        }
    except Exception:
        return defaults


def _toi_to_minutes(toi: str) -> float:
    """NHL API reports TOI as "MM:SS" — convert to decimal minutes."""
    try:
        minutes, seconds = toi.split(":")
        return int(minutes) + int(seconds) / 60.0
    except (ValueError, AttributeError):
        return 0.0


def _get_h2h(
    session: Session, home_id: int, away_id: int, as_of: datetime, limit: int
) -> dict[str, int]:
    result = session.execute(
        text("""
            SELECT home_score, away_score, home_team_id
            FROM games
            WHERE ((home_team_id=:h AND away_team_id=:a) OR (home_team_id=:a AND away_team_id=:h))
              AND scheduled_utc < :as_of AND status='final' AND home_score IS NOT NULL
            ORDER BY scheduled_utc DESC LIMIT :limit
        """),
        {"h": home_id, "a": away_id, "as_of": as_of, "limit": limit},
    )
    rows = list(result)
    wins = sum(
        1
        for r in rows
        if (r.home_team_id == home_id and r.home_score > r.away_score)
        or (r.home_team_id == away_id and r.away_score > r.home_score)
    )
    return {"home_wins": wins, "total": len(rows)}
