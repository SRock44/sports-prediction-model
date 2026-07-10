"""MLB matchup feature assembly: combines starter, bullpen, lineup, park, and Elo."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.features.common import (
    compute_elo_series,
    load_game_odds,
    load_game_weather,
)
from src.features.mlb.team import build_team_features


def build_matchup_features(
    session: Session,
    game: Any,
    as_of: datetime,
) -> dict[str, Any]:
    """Full MLB game feature vector.

    Args:
        game: A Game ORM instance.
        as_of: The cutoff timestamp for feature computation (typically scheduled_utc - 1h).
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

    # ── Venue/park factor ─────────────────────────────────────────────────────
    venue_row = session.execute(
        text("SELECT v.meta FROM games g JOIN venues v ON v.id=g.venue_id WHERE g.id=:gid"),
        {"gid": game_id},
    ).first()
    park_abbrev = (venue_row.meta or {}).get("team_abbrev") if venue_row else None

    # ── Team features ─────────────────────────────────────────────────────────
    home_feats = build_team_features(
        session, home_team_id, as_of, home_elo, park_abbrev, is_home=True
    )
    away_feats = build_team_features(
        session, away_team_id, as_of, away_elo, park_abbrev, is_home=False
    )

    matchup: dict[str, Any] = {}
    for k, v in home_feats.items():
        matchup[f"home_{k}"] = v
    for k, v in away_feats.items():
        matchup[f"away_{k}"] = v

    # ── Cross features ────────────────────────────────────────────────────────
    matchup["elo_diff"] = home_elo - away_elo
    matchup["elo_home_win_prob"] = 1.0 / (1.0 + 10.0 ** (-(home_elo + 50.0 - away_elo) / 400.0))
    matchup["run_diff_diff_last10"] = home_feats["run_diff_last10"] - away_feats["run_diff_last10"]
    matchup["opp_quality_diff"] = (
        home_feats["opp_quality_last10"] - away_feats["opp_quality_last10"]
    )
    matchup["run_diff_diff_last5"] = home_feats["run_diff_last5"] - away_feats["run_diff_last5"]
    matchup["woba_diff_last10"] = home_feats["woba_last10"] - away_feats["woba_last10"]
    matchup["rest_diff"] = home_feats["rest_days"] - away_feats["rest_days"]
    matchup["win_pct_diff_last10"] = home_feats["win_pct_last10"] - away_feats["win_pct_last10"]
    matchup["win_pct_season_diff"] = home_feats["win_pct_season"] - away_feats["win_pct_season"]
    matchup["streak_diff"] = home_feats["streak"] - away_feats["streak"]
    matchup["bullpen_rest_diff"] = away_feats["bullpen_ip_last3d"] - home_feats["bullpen_ip_last3d"]

    # ── Starting pitcher features ─────────────────────────────────────────────
    home_sp = _get_confirmed_starter(session, game_id, home_team_id, as_of)
    away_sp = _get_confirmed_starter(session, game_id, away_team_id, as_of)
    matchup.update(_pitcher_features(home_sp, prefix="home_sp"))
    matchup.update(_pitcher_features(away_sp, prefix="away_sp"))
    matchup["sp_xfip_diff"] = matchup.get("home_sp_xfip", 4.0) - matchup.get("away_sp_xfip", 4.0)

    # SP rolling form: last 5 starts ERA, WHIP, K/9, K%, BB%
    home_sp_id = home_sp.get("playerId") or home_sp.get("player_id") if home_sp else None
    away_sp_id = away_sp.get("playerId") or away_sp.get("player_id") if away_sp else None
    matchup.update(_sp_rolling_form(session, home_sp_id, as_of, prefix="home_sp"))
    matchup.update(_sp_rolling_form(session, away_sp_id, as_of, prefix="away_sp"))
    matchup["sp_form_era_diff"] = matchup.get("home_sp_form_era", 4.50) - matchup.get(
        "away_sp_form_era", 4.50
    )
    matchup["sp_form_whip_diff"] = matchup.get("home_sp_form_whip", 1.30) - matchup.get(
        "away_sp_form_whip", 1.30
    )
    matchup["sp_form_k9_diff"] = matchup.get("home_sp_form_k9", 8.0) - matchup.get(
        "away_sp_form_k9", 8.0
    )
    matchup["sp_form_k_pct_diff"] = matchup.get("home_sp_form_k_pct", 0.22) - matchup.get(
        "away_sp_form_k_pct", 0.22
    )
    matchup["sp_rest_diff"] = matchup.get("home_sp_rest_days", 5.0) - matchup.get(
        "away_sp_rest_days", 5.0
    )

    # Platoon splits: each team's avg runs vs LHP vs RHP starters
    home_vs_hand = _team_batting_vs_handedness(session, home_team_id, as_of)
    away_vs_hand = _team_batting_vs_handedness(session, away_team_id, as_of)
    # Positive platoon_adv = team scores MORE than average vs this game's opponent SP type
    away_sp_is_lhp = int(matchup.get("away_sp_handedness", 0))
    home_sp_is_lhp = int(matchup.get("home_sp_handedness", 0))
    home_expected = home_vs_hand["lhp_runs"] if away_sp_is_lhp else home_vs_hand["rhp_runs"]
    away_expected = away_vs_hand["lhp_runs"] if home_sp_is_lhp else away_vs_hand["rhp_runs"]
    matchup["home_platoon_adv"] = home_expected - home_vs_hand["avg_runs"]
    matchup["away_platoon_adv"] = away_expected - away_vs_hand["avg_runs"]
    matchup["platoon_adv_diff"] = matchup["home_platoon_adv"] - matchup["away_platoon_adv"]

    # Bullpen fatigue — individual team values + differential
    matchup["home_bullpen_ip_last3d"] = home_feats.get("bullpen_ip_last3d", 0.0)
    matchup["away_bullpen_ip_last3d"] = away_feats.get("bullpen_ip_last3d", 0.0)
    matchup["home_bullpen_pitches_last3d"] = home_feats.get("bullpen_pitches_last3d", 0.0)
    matchup["away_bullpen_pitches_last3d"] = away_feats.get("bullpen_pitches_last3d", 0.0)
    matchup["bullpen_pitch_diff"] = (
        matchup["away_bullpen_pitches_last3d"] - matchup["home_bullpen_pitches_last3d"]
    )

    # ── Head-to-head ──────────────────────────────────────────────────────────
    h2h = _get_h2h(session, home_team_id, away_team_id, as_of, 5)
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

    # ── Weather (MLB only — outdoor parks) ───────────────────────────────────
    weather = load_game_weather(session, game_id)
    if weather:
        matchup.update(weather)
        # Wind factor relative to ballpark orientation (+1 = blowing out, -1 = in)
        if park_abbrev:
            from src.ingest.mlb.weather import wind_out_factor

            matchup["weather_wind_out_factor"] = wind_out_factor(
                weather.get("weather_wind_mph", 5.0),
                weather.get("weather_wind_bearing", 180.0),
                park_abbrev,
            )
        else:
            matchup["weather_wind_out_factor"] = 0.0
    else:
        matchup["weather_temp_f"] = 72.0
        matchup["weather_wind_mph"] = 5.0
        matchup["weather_wind_bearing"] = 180.0
        matchup["weather_precip_prob"] = 0.0
        matchup["weather_wind_out_factor"] = 0.0

    # ── Bullpen quality (ERA + WHIP last 7d, not just fatigue) ───────────────
    matchup.update(_bullpen_quality(session, home_team_id, as_of, prefix="home"))
    matchup.update(_bullpen_quality(session, away_team_id, as_of, prefix="away"))
    matchup["bullpen_era_diff"] = matchup.get("away_bullpen_era_7d", 4.0) - matchup.get(
        "home_bullpen_era_7d", 4.0
    )
    matchup["bullpen_whip_diff"] = matchup.get("away_bullpen_whip_7d", 1.3) - matchup.get(
        "home_bullpen_whip_7d", 1.3
    )

    # ── HR park factor (separate from run factor — pitcher-friendly for HR) ──
    home_hr_factor = _HR_PARK_FACTORS.get(park_abbrev or "", 1.0)
    matchup["park_hr_factor"] = home_hr_factor

    # ── Umpire tendencies ─────────────────────────────────────────────────────
    matchup.update(_get_umpire_features(session, game_id))

    return matchup


# HR park factors: >1 = hitter-friendly for HR, <1 = pitcher-friendly
# Derived from multi-year Statcast HR/PA splits (home vs away at each park)
_HR_PARK_FACTORS: dict[str, float] = {
    "COL": 1.32,
    "CIN": 1.14,
    "TEX": 1.12,
    "PHI": 1.10,
    "BOS": 1.08,
    "NYY": 1.07,
    "MIL": 1.05,
    "HOU": 1.04,
    "CHC": 1.03,
    "ATL": 1.01,
    "ARI": 1.00,
    "STL": 0.99,
    "BAL": 0.99,
    "TOR": 0.98,
    "CWS": 0.98,
    "LAD": 0.97,
    "MIN": 0.97,
    "DET": 0.96,
    "CLE": 0.95,
    "WSH": 0.95,
    "NYM": 0.95,
    "KC": 0.94,
    "TB": 0.93,
    "LAA": 0.93,
    "PIT": 0.92,
    "MIA": 0.91,
    "SD": 0.90,
    "SF": 0.89,
    "SEA": 0.88,
    "OAK": 0.87,
}


def _bullpen_quality(
    session: Session,
    team_id: int,
    as_of: datetime,
    prefix: str,
    lookback_days: int = 7,
) -> dict[str, Any]:
    """ERA and WHIP for a team's bullpen over the last N days.

    Bullpen = pitchers who threw ≥1 IP but were NOT the starter (numberOfPitches < 75
    is a rough proxy; we exclude the highest-pitch pitcher per game as the SP proxy).
    """
    defaults = {
        f"{prefix}_bullpen_era_7d": 4.00,
        f"{prefix}_bullpen_whip_7d": 1.30,
        f"{prefix}_bullpen_k9_7d": 8.5,
    }
    from datetime import timedelta

    since = as_of - timedelta(days=lookback_days)
    try:
        rows = session.execute(
            text("""
                WITH game_starters AS (
                    SELECT DISTINCT ON (pgs.game_id)
                        pgs.game_id,
                        pgs.player_id AS sp_player_id
                    FROM player_game_stats pgs
                    WHERE pgs.team_id = :tid
                      AND pgs.stats->'pitching' IS NOT NULL
                      AND pgs.stats->'pitching' != '{}'::jsonb
                    ORDER BY pgs.game_id,
                             COALESCE((pgs.stats->'pitching'->>'numberOfPitches')::float, 0) DESC
                )
                SELECT
                    SUM((pgs.stats->'pitching'->>'earnedRuns')::float)       AS total_er,
                    SUM((pgs.stats->'pitching'->>'hits')::float)             AS total_h,
                    SUM((pgs.stats->'pitching'->>'baseOnBalls')::float)      AS total_bb,
                    SUM((pgs.stats->'pitching'->>'strikeOuts')::float)       AS total_k,
                    SUM(
                        FLOOR((pgs.stats->'pitching'->>'inningsPitched')::float)
                        + (((pgs.stats->'pitching'->>'inningsPitched')::float
                            - FLOOR((pgs.stats->'pitching'->>'inningsPitched')::float)) * 10.0 / 3.0)
                    ) AS total_ip
                FROM player_game_stats pgs
                JOIN games g ON g.id = pgs.game_id
                JOIN game_starters gs ON gs.game_id = pgs.game_id
                WHERE pgs.team_id = :tid
                  AND pgs.player_id != gs.sp_player_id
                  AND g.scheduled_utc >= :since
                  AND g.scheduled_utc < :as_of
                  AND g.status = 'final'
                  AND pgs.stats->'pitching' IS NOT NULL
                  AND pgs.stats->'pitching' != '{}'::jsonb
                  AND (pgs.stats->'pitching'->>'inningsPitched')::float >= 0.1
            """),
            {"tid": team_id, "since": since, "as_of": as_of},
        ).first()

        if not rows or not rows.total_ip or float(rows.total_ip) < 1.0:
            return defaults

        ip = float(rows.total_ip)
        er = float(rows.total_er or 0)
        h = float(rows.total_h or 0)
        bb = float(rows.total_bb or 0)
        k = float(rows.total_k or 0)
        return {
            f"{prefix}_bullpen_era_7d": min(er * 9.0 / ip, 18.0),
            f"{prefix}_bullpen_whip_7d": min((h + bb) / ip, 4.0),
            f"{prefix}_bullpen_k9_7d": k * 9.0 / ip,
        }
    except Exception:
        return defaults


def _get_confirmed_starter(
    session: Session, game_id: int, team_id: int, as_of: datetime
) -> dict[str, Any] | None:
    """Look up confirmed starting pitcher, with box-score fallback.

    Tries lineups table first (live games). For historical games where the lineups
    table is empty, falls back to identifying the starter as the pitcher with the
    most innings pitched for that team in the game's box score.
    """
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
            if p.get("position") in ("SP", "P") and p.get("batting_order") == 0:
                return p  # type: ignore[no-any-return]

    # Box-score fallback: pitcher with most pitches thrown (proxy for starter).
    # inningsPitched is missing from many rows; numberOfPitches covers 90%+ of games.
    result = session.execute(
        text("""
            SELECT pgs.player_id, pl.throws,
                   (pgs.stats->'pitching'->>'strikeOuts')::float AS k,
                   COALESCE((pgs.stats->'pitching'->>'baseOnBalls')::float,
                            (pgs.stats->'pitching'->>'walks')::float, 0) AS bb,
                   COALESCE((pgs.stats->'pitching'->>'battersFaced')::float, 1) AS bf
            FROM player_game_stats pgs
            JOIN players pl ON pl.id = pgs.player_id
            WHERE pgs.game_id = :gid
              AND pgs.team_id = :tid
              AND pgs.stats->'pitching' IS NOT NULL
              AND pgs.stats->'pitching' != '{}'::jsonb
              AND COALESCE(
                    (pgs.stats->'pitching'->>'numberOfPitches')::float,
                    (pgs.stats->'pitching'->>'inningsPitched')::float * 15,
                    0
                  ) > 0
            ORDER BY COALESCE(
                       (pgs.stats->'pitching'->>'numberOfPitches')::float,
                       (pgs.stats->'pitching'->>'inningsPitched')::float * 15,
                       0
                     ) DESC
            LIMIT 1
        """),
        {"gid": game_id, "tid": team_id},
    ).first()

    if result:
        k = float(result.k or 0)
        bb = float(result.bb or 0)
        bf = max(float(result.bf or 1), 1.0)
        return {
            "player_id": result.player_id,
            "throws": result.throws or "R",
            "k_bb_pct": (k - bb) / bf,
            "xfip": 4.50,  # not available from box score; rolling form fills this
        }
    return None


def _pitcher_features(pitcher: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    if pitcher is None:
        return {
            f"{prefix}_xfip": 4.50,
            f"{prefix}_k_bb_pct": 0.10,
            f"{prefix}_handedness": 0,
            f"{prefix}_known": 0,
        }
    return {
        f"{prefix}_xfip": float(pitcher.get("xfip", 4.50)),
        f"{prefix}_k_bb_pct": float(pitcher.get("k_bb_pct", 0.10)),
        f"{prefix}_handedness": int(pitcher.get("throws", "R") == "L"),
        f"{prefix}_known": 1,
    }


def _parse_ip(ip_str: float | str) -> float:
    """Convert baseball IP notation to decimal innings.

    Baseball uses base-3 fractions: "4.1" = 4⅓, "4.2" = 4⅔.
    Python float("4.1") = 4.1, which incorrectly inflates ERA calculations.
    """
    raw = float(ip_str)
    whole = int(raw)
    frac = round(raw - whole, 1)
    return whole + frac * 10.0 / 3.0


def _sp_rolling_form(
    session: Session,
    player_id: int | str | None,
    as_of: datetime,
    prefix: str,
    n_starts: int = 5,
) -> dict[str, Any]:
    """Last N starts ERA, WHIP, K/9, K%, BB% for a starting pitcher."""
    defaults = {
        f"{prefix}_form_era": 4.50,
        f"{prefix}_form_whip": 1.30,
        f"{prefix}_form_k9": 8.0,
        f"{prefix}_form_k_pct": 0.22,
        f"{prefix}_form_bb_pct": 0.08,
        f"{prefix}_form_known": 0,
        f"{prefix}_rest_days": 5.0,
    }
    if player_id is None:
        return defaults
    try:
        result = session.execute(
            text("""
                SELECT pgs.stats->'pitching' AS pit, g.scheduled_utc AS game_date
                FROM player_game_stats pgs
                JOIN games g ON g.id = pgs.game_id
                WHERE pgs.player_id = :pid
                  AND g.scheduled_utc < :as_of
                  AND g.status = 'final'
                  AND pgs.stats->'pitching' IS NOT NULL
                ORDER BY g.scheduled_utc DESC
                LIMIT :n
            """),
            {"pid": player_id, "as_of": as_of, "n": n_starts},
        )
        raw_rows = list(result)
        rows = [r.pit for r in raw_rows if r.pit]
        if not rows:
            return defaults

        # Rest days since most recent start
        from datetime import UTC as _UTC

        last_start = raw_rows[0].game_date
        if hasattr(last_start, "tzinfo") and last_start.tzinfo is None:
            last_start = last_start.replace(tzinfo=_UTC)
        as_of_aware = (
            as_of if (hasattr(as_of, "tzinfo") and as_of.tzinfo) else as_of.replace(tzinfo=_UTC)
        )
        sp_rest_days = min((as_of_aware - last_start).total_seconds() / 86400, 10.0)

        eras, whips, k9s, k_pcts, bb_pcts = [], [], [], [], []
        for pit in rows:
            er = pit.get("earnedRuns") or pit.get("earnedRunsAllowed")
            ip = pit.get("inningsPitched")
            k = pit.get("strikeOuts")
            bb = pit.get("baseOnBalls") or pit.get("walks")
            h = pit.get("hits") or pit.get("hitsAllowed")
            bf = pit.get("battersFaced") or pit.get("pitchesThrown")

            if ip:
                ip_actual = _parse_ip(ip)
                # Require at least 2 actual innings to avoid relief-appearance noise
                if ip_actual >= 2.0:
                    if er is not None:
                        era = float(er) * 9.0 / ip_actual
                        eras.append(min(era, 27.0))
                    if h is not None and bb is not None:
                        whip = (float(h) + float(bb)) / ip_actual
                        whips.append(min(whip, 4.0))
                    if k is not None:
                        k9s.append(float(k) * 9.0 / ip_actual)

            if k is not None and bf and float(bf) > 0:
                k_pcts.append(float(k) / float(bf))
            if bb is not None and bf and float(bf) > 0:
                bb_pcts.append(float(bb) / float(bf))

        from src.features.common import rolling_mean

        return {
            f"{prefix}_form_era": rolling_mean(eras, n_starts) or 4.50,
            f"{prefix}_form_whip": rolling_mean(whips, n_starts) or 1.30,
            f"{prefix}_form_k9": rolling_mean(k9s, n_starts) or 8.0,
            f"{prefix}_form_k_pct": rolling_mean(k_pcts, n_starts) or 0.22,
            f"{prefix}_form_bb_pct": rolling_mean(bb_pcts, n_starts) or 0.08,
            f"{prefix}_form_known": int(bool(eras)),
            f"{prefix}_rest_days": sp_rest_days,
        }
    except Exception:
        return defaults


def _team_batting_vs_handedness(
    session: Session,
    team_id: int,
    as_of: datetime,
    lookback_days: int = 365,
) -> dict[str, float]:
    """Avg runs scored per game vs LHP vs RHP starters over the past year.

    Returns {"lhp_runs": float, "rhp_runs": float, "avg_runs": float}.
    Starter identified as the opponent pitcher with >=50 pitches thrown (most thrown).
    Requires at least 5 games per split; falls back to 4.5 otherwise.
    """
    from datetime import timedelta

    since = as_of - timedelta(days=lookback_days)
    defaults = {"lhp_runs": 4.5, "rhp_runs": 4.5, "avg_runs": 4.5}
    try:
        rows = session.execute(
            text("""
                WITH opp_starters AS (
                    SELECT DISTINCT ON (pgs.game_id)
                        pgs.game_id,
                        pgs.team_id AS pitching_team_id,
                        pl.throws AS sp_hand
                    FROM player_game_stats pgs
                    JOIN players pl ON pl.id = pgs.player_id
                    WHERE pgs.stats->'pitching' IS NOT NULL
                      AND pgs.stats->'pitching' != '{}'::jsonb
                      AND pl.throws IS NOT NULL
                      AND COALESCE((pgs.stats->'pitching'->>'numberOfPitches')::float, 0) >= 50
                    ORDER BY
                        pgs.game_id,
                        (pgs.stats->'pitching'->>'numberOfPitches')::float DESC
                ),
                team_batting AS (
                    SELECT
                        g.id AS game_id,
                        CASE WHEN g.home_team_id = :tid THEN g.home_score
                             ELSE g.away_score END AS runs,
                        CASE WHEN g.home_team_id = :tid THEN g.away_team_id
                             ELSE g.home_team_id END AS opp_team_id
                    FROM games g
                    WHERE (g.home_team_id = :tid OR g.away_team_id = :tid)
                      AND g.status = 'final'
                      AND g.home_score IS NOT NULL
                      AND g.scheduled_utc < :as_of
                      AND g.scheduled_utc >= :since
                )
                SELECT os.sp_hand, AVG(tb.runs) AS avg_runs, COUNT(*) AS n
                FROM team_batting tb
                JOIN opp_starters os
                    ON os.game_id = tb.game_id
                   AND os.pitching_team_id = tb.opp_team_id
                GROUP BY os.sp_hand
            """),
            {"tid": team_id, "as_of": as_of, "since": since},
        ).fetchall()

        result = dict(defaults)
        total_runs, total_n = 0.0, 0
        for row in rows:
            n = int(row.n)
            avg = float(row.avg_runs)
            if n >= 5:
                if row.sp_hand == "L":
                    result["lhp_runs"] = avg
                elif row.sp_hand == "R":
                    result["rhp_runs"] = avg
            total_runs += avg * n
            total_n += n
        if total_n > 0:
            result["avg_runs"] = total_runs / total_n
        return result
    except Exception:
        return defaults


def _get_umpire_features(session: Session, game_id: int) -> dict[str, Any]:
    """Compute home plate umpire tendencies from historical games they worked.

    Umpires stored in games.meta['umpires'] as a list; first entry is HP ump.
    Graceful fallback to league averages if data unavailable.
    """
    defaults = {
        "ump_k_rate": 0.215,  # strikeouts per batter faced (league avg ~21.5%)
        "ump_bb_rate": 0.082,  # walks per batter faced
        "ump_home_bias": 0.0,  # deviation from 0.5 home win pct
    }
    try:
        game_meta = session.execute(
            text("SELECT meta FROM games WHERE id = :gid"), {"gid": game_id}
        ).scalar()
        umpires = (game_meta or {}).get("umpires", [])
        hp_ump = umpires[0] if umpires else None
        if not hp_ump:
            return defaults

        result = session.execute(
            text("""
                SELECT
                    AVG(
                        CASE WHEN (pgs.stats->'pitching'->>'strikeOuts')::float IS NOT NULL
                             AND (pgs.stats->'pitching'->>'battersFaced')::float > 0
                             THEN (pgs.stats->'pitching'->>'strikeOuts')::float /
                                  (pgs.stats->'pitching'->>'battersFaced')::float
                             ELSE NULL END
                    ) AS k_rate,
                    AVG(
                        CASE WHEN (pgs.stats->'pitching'->>'baseOnBalls')::float IS NOT NULL
                             AND (pgs.stats->'pitching'->>'battersFaced')::float > 0
                             THEN (pgs.stats->'pitching'->>'baseOnBalls')::float /
                                  (pgs.stats->'pitching'->>'battersFaced')::float
                             ELSE NULL END
                    ) AS bb_rate,
                    AVG(CASE WHEN g.home_score > g.away_score THEN 1.0 ELSE 0.0 END) AS home_win_pct,
                    COUNT(DISTINCT g.id) AS n_games
                FROM games g
                JOIN player_game_stats pgs ON pgs.game_id = g.id
                WHERE g.status = 'final'
                  AND g.meta->'umpires' @> :hp_ump_json::jsonb
                  AND pgs.stats->'pitching' IS NOT NULL
            """),
            {"hp_ump_json": f'["{hp_ump}"]'},
        ).first()

        if result and result.n_games and result.n_games >= 10:
            return {
                "ump_k_rate": float(result.k_rate or 0.215),
                "ump_bb_rate": float(result.bb_rate or 0.082),
                "ump_home_bias": float((result.home_win_pct or 0.5) - 0.5),
            }
    except Exception:
        pass
    return defaults


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
