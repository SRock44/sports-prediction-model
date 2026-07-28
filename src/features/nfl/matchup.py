"""NFL matchup feature assembly: Elo, team form, starting-QB signal, weather, odds, H2H.

Starting-QB resolution (NFL's single dominant signal, analogous to MLB's
confirmed starting pitcher - see NFL.md): for a game that has already been
played, nflverse's own schedule row records who actually started at QB
(captured into Game.meta['home_qb_id']/['away_qb_id'] by
src/ingest/nfl/games.py::ingest_season_schedule) - this is ground truth, not a
prediction, exactly like MLB's box-score-fallback starter. For a future game
with no recorded starter yet, this falls back to (a) the Lineup table (source
official/probable - no NFL ingest writes here yet, so this is a structural
hook for later, not dead code) then (b) the team's most recent known starting
QB, discounted if that player's most recent injury report says 'out'.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.features.common import (
    compute_elo_series,
    load_game_odds,
    load_injuries_before,
    load_player_game_stats_before,
    rolling_mean,
)
from src.features.nfl.team import build_team_features, venue_is_dome

_QB_FORM_WINDOWS = (4, 8)


def build_matchup_features(
    session: Session,
    game: Any,
    as_of: datetime,
) -> dict[str, Any]:
    """Full NFL game feature vector. Mirrors src/features/mlb/matchup.py's shape."""
    import pandas as pd

    game_id: int = game.id
    home_team_id: int = game.home_team_id
    away_team_id: int = game.away_team_id
    sport_id: int = game.sport_id
    meta = game.meta or {}

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
    elo_ratings = compute_elo_series(pd.DataFrame(rows)) if rows else {}
    home_elo = elo_ratings.get(home_team_id, 1500.0)
    away_elo = elo_ratings.get(away_team_id, 1500.0)

    # ── Team abbrevs (for travel_km) ─────────────────────────────────────────
    team_abbrevs: dict[int, str] = {
        row[0]: row[1]
        for row in session.execute(
            text("SELECT id, abbrev FROM teams WHERE id IN (:h, :a)"),
            {"h": home_team_id, "a": away_team_id},
        ).all()
    }
    home_abbrev = team_abbrevs.get(home_team_id)
    away_abbrev = team_abbrevs.get(away_team_id)

    # ── Team features ─────────────────────────────────────────────────────────
    home_feats = build_team_features(session, home_team_id, as_of, home_elo, is_home=True)
    away_feats = build_team_features(
        session, away_team_id, as_of, away_elo, away_abbrev, home_abbrev, is_home=False
    )

    matchup: dict[str, Any] = {}
    for k, v in home_feats.items():
        matchup[f"home_{k}"] = v
    for k, v in away_feats.items():
        matchup[f"away_{k}"] = v

    # ── Cross features ────────────────────────────────────────────────────────
    matchup["elo_diff"] = home_elo - away_elo
    matchup["elo_home_win_prob"] = 1.0 / (1.0 + 10.0 ** (-(home_elo + 100.0 - away_elo) / 400.0))
    matchup["point_diff_diff_last8"] = (
        home_feats["point_diff_last8"] - away_feats["point_diff_last8"]
    )
    matchup["epa_per_play_diff_last8"] = (
        home_feats["epa_per_play_last8"] - away_feats["epa_per_play_last8"]
    )
    matchup["pace_diff_last8"] = home_feats["pace_last8"] - away_feats["pace_last8"]
    # Turnover differential proxy: each side's own giveaway rate, not true
    # forced-turnovers (see src/features/nfl/team.py's module docstring for why).
    matchup["turnover_diff_last8"] = (
        away_feats["turnovers_committed_last8"] - home_feats["turnovers_committed_last8"]
    )
    matchup["rest_diff"] = home_feats["rest_days"] - away_feats["rest_days"]
    matchup["win_pct_season_diff"] = home_feats["win_pct_season"] - away_feats["win_pct_season"]
    matchup["streak_diff"] = home_feats["streak"] - away_feats["streak"]

    # ── Starting QB (the dominant signal) ─────────────────────────────────────
    home_qb_id, home_qb_confirmed = _resolve_starting_qb(
        session, game_id, home_team_id, meta.get("home_qb_id"), as_of
    )
    away_qb_id, away_qb_confirmed = _resolve_starting_qb(
        session, game_id, away_team_id, meta.get("away_qb_id"), as_of
    )
    matchup.update(_qb_form(session, home_qb_id, as_of, prefix="home_qb"))
    matchup.update(_qb_form(session, away_qb_id, as_of, prefix="away_qb"))
    matchup["home_qb_confirmed"] = int(home_qb_confirmed)
    matchup["away_qb_confirmed"] = int(away_qb_confirmed)
    matchup["qb_epa_diff_last8"] = matchup.get("home_qb_form_epa_last8", 0.0) - matchup.get(
        "away_qb_form_epa_last8", 0.0
    )

    # ── Head-to-head ──────────────────────────────────────────────────────────
    h2h = _get_h2h(session, home_team_id, away_team_id, as_of, 3)
    matchup["h2h_home_win_pct"] = h2h["home_wins"] / max(h2h["total"], 1)
    matchup["h2h_total"] = h2h["total"]

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

    # ── Weather (from Game.meta, populated at schedule-ingest time - no
    # separate weather ingest module needed since nflverse's schedule rows
    # already carry temp/wind for played games) ──────────────────────────────
    is_dome = venue_is_dome(meta.get("roof"))
    matchup["weather_is_dome"] = int(is_dome)
    matchup["weather_temp_f"] = float(meta["temp"]) if meta.get("temp") is not None else 60.0
    matchup["weather_wind_mph"] = (
        0.0 if is_dome else (float(meta["wind"]) if meta.get("wind") is not None else 7.0)
    )

    return matchup


# Injury designations that mean the most-recent-starter guess won't play.
# nflverse's NFL injury report only ever emits questionable/out/doubtful
# (verified against the real backfill: 27517/6375/920 rows) - there is no 'ir'
# for this sport, so listing one would be dead code. 'doubtful' is included
# because NFL's doubtful is a strong negative signal (~75% don't play), unlike
# 'questionable', which is close to a coin flip and stays in.
_QB_OUT_INJURY_STATUSES = frozenset({"out", "doubtful"})

# nflverse roster status codes that mean the player won't line up for this team
# at all - suspensions and roster moves, which the injury report does NOT
# carry. Verified against the real roster snapshot; the full observed set is
# ACT/CUT/DEV/RES/INA/UFA/RET/TRC/E14/TRD/TRT/RFA/NWT/EXE/SUS/RSR/PUP.
# Deliberately narrow: only codes that unambiguously mean "not playing for this
# team". RES/INA/PUP/DEV are excluded because they're either injury-related
# (already covered above) or ambiguous in a season-level snapshot.
_QB_DISQUALIFYING_ROSTER_STATUSES = frozenset(
    {
        "SUS",  # suspended
        "RET",  # retired
        "CUT",  # released
        "EXE",  # commissioner's exempt list
        "NWT",  # not with team
    }
)


def _roster_status_disqualifies(session: Session, player_id: int) -> bool:
    """True if the player's roster status rules them out of playing.

    Covers the gap the injury report leaves: a suspended or released QB is
    never 'out' on an injury report, so without this a suspended starter still
    looks available and the model would happily use his form.

    Only consulted from the most-recent-starter fallback below, which is
    serve-time only (a played game resolves its starter from Game.meta and
    returns before reaching it). That matters: Player.meta.status is a
    season-level snapshot of the player's *current* state, not an as-of value,
    so using it while building features for a historical game would leak.
    """
    status = session.execute(
        text("SELECT meta->>'status' FROM players WHERE id = :pid"),
        {"pid": player_id},
    ).scalar()
    return bool(status) and status in _QB_DISQUALIFYING_ROSTER_STATUSES


def _resolve_starting_qb(
    session: Session,
    game_id: int,
    team_id: int,
    meta_qb_id: str | None,
    as_of: datetime,
) -> tuple[int | None, bool]:
    """Return (Player.id, confirmed) for the team's starting QB.

    meta_qb_id is nflverse's own GSIS id string (e.g. '00-0033873') - resolve
    it to our internal Player.id via external_id. Confirmed=True means this is
    a known-played/known-lineup starter, not a most-recent-starter guess.
    """
    if meta_qb_id:
        row = session.execute(
            text(
                "SELECT id FROM players WHERE external_id = :ext AND sport_id = "
                "(SELECT sport_id FROM teams WHERE id = :tid)"
            ),
            {"ext": meta_qb_id, "tid": team_id},
        ).first()
        if row:
            return int(row.id), True

    # Lineup table fallback - no NFL ingest writes here yet (structural hook).
    row = session.execute(
        text("""
            SELECT players FROM lineups
            WHERE game_id = :gid AND team_id = :tid AND source IN ('official', 'probable')
            ORDER BY (source = 'official') DESC, fetched_at DESC LIMIT 1
        """),
        {"gid": game_id, "tid": team_id},
    ).first()
    if row and row.players:
        for p in row.players:
            if p.get("position") == "QB":
                pid = p.get("player_id") or p.get("playerId")
                if pid:
                    return int(pid), True

    # Last resort: this team's most recent known starting QB (from a past
    # game's meta), returned unconfirmed - or dropped entirely if an injury
    # designation or a roster move says he won't play.
    row = session.execute(
        text("""
            SELECT pgs.player_id
            FROM player_game_stats pgs
            JOIN games g ON g.id = pgs.game_id
            WHERE pgs.team_id = :tid
              AND g.scheduled_utc < :as_of
              AND g.status = 'final'
              AND (pgs.stats->>'attempts')::float > 5
            ORDER BY g.scheduled_utc DESC, (pgs.stats->>'attempts')::float DESC
            LIMIT 1
        """),
        {"tid": team_id, "as_of": as_of},
    ).first()
    if row:
        candidate_id = int(row.player_id)
        # A disqualifying status on the most-recent-starter guess means this
        # feature is actively misleading, not just uncertain - report unknown
        # (which falls back to generic QB form) rather than confidently
        # pointing at a QB who won't take the field.
        injuries = load_injuries_before(session, candidate_id, as_of)
        if injuries and injuries[0]["status"] in _QB_OUT_INJURY_STATUSES:
            return None, False
        if _roster_status_disqualifies(session, candidate_id):
            return None, False
        return candidate_id, False
    return None, False


def _qb_form(
    session: Session,
    player_id: int | None,
    as_of: datetime,
    prefix: str,
) -> dict[str, Any]:
    defaults = {
        f"{prefix}_form_epa_last8": 0.0,
        f"{prefix}_form_ypa_last8": 6.8,
        f"{prefix}_form_int_rate_last8": 0.02,
        f"{prefix}_form_rush_yds_last8": 5.0,
        f"{prefix}_known": 0,
    }
    if player_id is None:
        return defaults

    games = load_player_game_stats_before(session, player_id, as_of, limit=8)
    if not games:
        return defaults

    epa, ypa, int_rate, rush_yds = [], [], [], []
    # load_player_game_stats_before returns most-recent-first; rolling_mean
    # assumes oldest-first (see src/features/nfl/team.py's fix for the same
    # issue). Fetch limit==window here means this is currently a no-op in
    # terms of which games are included, but reversing keeps it correct if
    # that ever changes (e.g. adding a shorter window off the same fetch).
    for g in reversed(games):
        s = g["stats"] or {}
        attempts = float(s.get("attempts", 0) or 0)
        if attempts <= 0:
            continue
        epa.append(float(s.get("passing_epa", 0) or 0))
        ypa.append(float(s.get("passing_yards", 0) or 0) / attempts)
        int_rate.append(float(s.get("interceptions", 0) or 0) / attempts)
        rush_yds.append(float(s.get("rushing_yards", 0) or 0))

    if not epa:
        return defaults

    return {
        f"{prefix}_form_epa_last8": rolling_mean(epa, 8) or 0.0,
        f"{prefix}_form_ypa_last8": rolling_mean(ypa, 8) or 6.8,
        f"{prefix}_form_int_rate_last8": rolling_mean(int_rate, 8) or 0.02,
        f"{prefix}_form_rush_yds_last8": rolling_mean(rush_yds, 8) or 5.0,
        f"{prefix}_known": 1,
    }


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
