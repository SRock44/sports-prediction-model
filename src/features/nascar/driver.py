"""NASCAR driver-level feature engineering.

All features use data strictly before `as_of_utc` — same as-of invariant every other
sport's feature module enforces (src/features/common.py). There is no "matchup" the
way team sports have one: a race has N entrants sharing one event, not two sides
facing each other, so there's no `matchup.py` here — see field.py for how per-driver
feature vectors get assembled into a full-field feature matrix for one race.

Elo: NASCAR has no home/away, so src/features/common.py's compute_elo_series (which
hardcodes home-field advantage into its replay loop) doesn't apply — same conclusion
UFC.md reaches for fighters. compute_driver_elo_ratings below is a NASCAR-specific
replay using the same underlying elo_expected/elo_update pairwise math, but doing an
all-pairs comparison within each historical race's field (driver A "beat" driver B if
A finished ahead of B) instead of a single 1v1 comparison per game.

v2 (2026-07-23, after the first-pass model evaluated at 68% top-5 hit rate on a 25-race
holdout, below the 75% target): adds loop-stats-derived features (average/closing
running position, quality passes, driver rating - already collected during ingestion
into RaceEntry.meta but not previously turned into rolling features), manufacturer-level
recent form, an explicit Next-Gen-car era flag (2022's rules package was a real
regulatory discontinuity), and one-hot track-type dummies (previously only a single
scalar "track_type_avg_finish", which can't let the model learn track-type-specific
feature interactions the way explicit dummies can).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.features.common import elo_update, rolling_mean

_WINDOWS = [5, 10, 20]
_DEFAULT_ELO = 1500.0
_K_FACTOR = 16.0  # NASCAR: ~36 races/season vs. NHL's 82 - lower per-comparison K than
# a dense team-sport schedule would use, since each race feeds ~35-40 pairwise
# comparisons into the rating already (much more signal per event than one game).

_NEXT_GEN_ERA_START_SEASON = 2022  # NASCAR's Next Gen car - a real rules-package
# discontinuity, not a gradual trend. Mixing pre/post data unflagged risks noise.

# Best-effort track-type classification, same idiom as NHL's arena-coords dict
# (src/features/nhl/team.py) - static, degrades safely to "intermediate" on a miss
# rather than raising. Not guaranteed to cover every track name variant cf.nascar.com
# returns.
_SUPERSPEEDWAYS = {"Daytona International Speedway", "Talladega Superspeedway"}
_SHORT_TRACKS = {
    "Martinsville Speedway",
    "Bristol Motor Speedway",
    "Richmond Raceway",
    "North Wilkesboro Speedway",
    "Iowa Speedway",
}
_ROAD_COURSES = {
    "Watkins Glen International",
    "Sonoma Raceway",
    "Circuit of the Americas",
    "Charlotte Motor Speedway",  # Roval configuration - same venue name as the oval race
    "Road America",
    "Chicago Street Course",
}
_TRACK_TYPES = ("superspeedway", "short_track", "road_course", "intermediate")


def _track_type(track_name: str | None) -> str:
    if not track_name:
        return "intermediate"
    if track_name in _SUPERSPEEDWAYS:
        return "superspeedway"
    if track_name in _SHORT_TRACKS:
        return "short_track"
    if track_name in _ROAD_COURSES:
        return "road_course"
    return "intermediate"


def _load_driver_race_history(
    session: Session, driver_id: int, as_of_utc: datetime, series: str, limit: int = 40
) -> list[dict[str, Any]]:
    """Per-race results for one driver, most recent first. One query, chronologically
    aligned, same pattern as NHL's _load_team_game_history.

    `series` filter is required, not optional: Cup and Truck share the same Sport row
    and Player namespace (a driver's identity persists across series - see races.py's
    module docstring), so without this filter a cross-series driver's rolling stats
    would blend two different competitive levels into one nonsensical history.
    """
    rows = session.execute(
        text("""
            SELECT r.id AS race_id, r.scheduled_utc, r.season, r.meta->>'track_name' AS track_name,
                   re.starting_position, re.finishing_position, re.laps_completed,
                   re.laps_led, re.status, r.laps_actual, re.meta AS entry_meta
            FROM race_entries re
            JOIN races r ON r.id = re.race_id
            WHERE re.driver_id = :did
              AND r.series = :series
              AND r.scheduled_utc < :as_of
              AND r.status = 'final'
              AND re.finishing_position IS NOT NULL
            ORDER BY r.scheduled_utc DESC
            LIMIT :limit
        """),
        {"did": driver_id, "series": series, "as_of": as_of_utc, "limit": limit},
    )
    return [dict(r._mapping) for r in rows]


def compute_driver_elo_ratings(
    session: Session, sport_id: int, as_of_utc: datetime, series: str
) -> dict[int, float]:
    """Replay every completed race before as_of_utc, doing an all-pairs comparison
    within each race's field (driver A "beat" B if A finished ahead of B). Returns
    {driver_id: elo_rating}. Older races first, same "final ratings after replay"
    contract as compute_elo_series. `series` filter: see _load_driver_race_history.
    """
    rows = session.execute(
        text("""
            SELECT r.id AS race_id, r.scheduled_utc, re.driver_id, re.finishing_position
            FROM race_entries re
            JOIN races r ON r.id = re.race_id
            WHERE r.sport_id = :sid
              AND r.series = :series
              AND r.scheduled_utc < :as_of
              AND r.status = 'final'
              AND re.finishing_position IS NOT NULL
            ORDER BY r.scheduled_utc
        """),
        {"sid": sport_id, "series": series, "as_of": as_of_utc},
    )

    races: dict[int, list[tuple[int, int]]] = {}
    order: list[int] = []
    for r in rows:
        race_id = r.race_id
        if race_id not in races:
            races[race_id] = []
            order.append(race_id)
        races[race_id].append((r.driver_id, r.finishing_position))

    ratings: dict[int, float] = {}
    for race_id in order:
        field = sorted(races[race_id], key=lambda t: t[1])  # by finishing position
        n = len(field)
        for i in range(n):
            driver_a, _ = field[i]
            rating_a = ratings.get(driver_a, _DEFAULT_ELO)
            for j in range(i + 1, n):
                driver_b, _ = field[j]
                rating_b = ratings.get(driver_b, _DEFAULT_ELO)
                # driver_a finished ahead of driver_b in this race.
                new_a, new_b = elo_update(rating_a, rating_b, score_a=1.0, k=_K_FACTOR)
                ratings[driver_a] = new_a
                ratings[driver_b] = new_b
                rating_a = new_a  # subsequent comparisons in this race use the updated rating

    return ratings


def compute_manufacturer_form(
    session: Session, sport_id: int, as_of_utc: datetime, series: str, lookback_days: int = 365
) -> dict[str, dict[str, float]]:
    """Recent competitiveness per manufacturer (Chevy/Ford/Toyota) over the trailing
    year - a new v2 feature (see module docstring): drafting/engineering competitiveness
    is a real, previously-unrepresented signal, especially at superspeedways.
    Returns {manufacturer: {"win_pct":..., "top5_pct":..., "avg_finish":...}}.
    `series` filter: see _load_driver_race_history - a manufacturer's Cup program and
    Truck program are different teams/budgets, not one blended competitiveness signal.
    """
    rows = session.execute(
        text("""
            SELECT re.manufacturer, re.finishing_position
            FROM race_entries re
            JOIN races r ON r.id = re.race_id
            WHERE r.sport_id = :sid
              AND r.series = :series
              AND r.scheduled_utc < :as_of
              AND r.scheduled_utc >= :cutoff
              AND r.status = 'final'
              AND re.finishing_position IS NOT NULL
              AND re.manufacturer IS NOT NULL
        """),
        {
            "sid": sport_id,
            "series": series,
            "as_of": as_of_utc,
            "cutoff": as_of_utc - timedelta(days=lookback_days),
        },
    )

    by_mfg: dict[str, list[int]] = {}
    for r in rows:
        by_mfg.setdefault(r.manufacturer, []).append(r.finishing_position)

    form: dict[str, dict[str, float]] = {}
    for mfg, finishes in by_mfg.items():
        n = len(finishes)
        form[mfg] = {
            "win_pct": sum(1 for f in finishes if f == 1) / n,
            "top5_pct": sum(1 for f in finishes if f <= 5) / n,
            "avg_finish": float(np.mean(finishes)),
        }
    return form


def build_driver_features(
    session: Session,
    driver_id: int,
    as_of_utc: datetime,
    elo_rating: float,
    starting_position: int | None,
    track_name: str | None,
    season: int,
    series: str,
    manufacturer: str | None = None,
    manufacturer_form: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Build NASCAR driver features for the race-winner model. One row per driver;
    see field.py for how these get assembled across a whole race field.
    """
    history = _load_driver_race_history(session, driver_id, as_of_utc, series, limit=40)

    feats: dict[str, Any] = {}
    feats["elo"] = elo_rating
    feats["starting_position"] = float(starting_position) if starting_position else 20.0
    feats["era_next_gen"] = 1.0 if season >= _NEXT_GEN_ERA_START_SEASON else 0.0

    track_type = _track_type(track_name)
    for tt in _TRACK_TYPES:
        feats[f"track_is_{tt}"] = 1.0 if tt == track_type else 0.0

    mfg_form = (manufacturer_form or {}).get(manufacturer or "", {})
    feats["manufacturer_win_pct"] = mfg_form.get("win_pct", 0.03)
    feats["manufacturer_top5_pct"] = mfg_form.get("top5_pct", 0.14)
    feats["manufacturer_avg_finish"] = mfg_form.get("avg_finish", 20.0)

    if not history:
        _fill_defaults(feats)
        return feats

    finishes, wins, top5s, top10s, dnfs, laps_led_pct = [], [], [], [], [], []
    track_type_finishes: list[float] = []
    race_dates: list[datetime] = []
    avg_ps_vals, closing_ps_vals, quality_passes_vals, rating_vals, qual_speed_vals = (
        [],
        [],
        [],
        [],
        [],
    )

    for h in history:
        fp = h["finishing_position"]
        finishes.append(float(fp))
        wins.append(int(fp == 1))
        top5s.append(int(fp <= 5))
        top10s.append(int(fp <= 10))
        dnfs.append(int(h["status"] in ("accident", "mechanical", "dnf")))

        laps_actual = h["laps_actual"] or 0
        if laps_actual > 0 and h["laps_led"] is not None:
            laps_led_pct.append(h["laps_led"] / laps_actual)

        if _track_type(h["track_name"]) == track_type:
            track_type_finishes.append(float(fp))

        race_dates.append(h["scheduled_utc"])

        entry_meta = h.get("entry_meta") or {}
        if entry_meta.get("avg_ps") is not None:
            avg_ps_vals.append(float(entry_meta["avg_ps"]))
        if entry_meta.get("closing_ps") is not None:
            closing_ps_vals.append(float(entry_meta["closing_ps"]))
        if entry_meta.get("quality_passes") is not None:
            quality_passes_vals.append(float(entry_meta["quality_passes"]))
        if entry_meta.get("rating") is not None:
            rating_vals.append(float(entry_meta["rating"]))
        if entry_meta.get("qualifying_speed") is not None:
            qual_speed_vals.append(float(entry_meta["qualifying_speed"]))

    for w in _WINDOWS:
        feats[f"avg_finish_last{w}"] = rolling_mean(finishes, w) or 20.0
        feats[f"win_pct_last{w}"] = rolling_mean(wins, w) or 0.03
        feats[f"top5_pct_last{w}"] = rolling_mean(top5s, w) or 0.14
        feats[f"top10_pct_last{w}"] = rolling_mean(top10s, w) or 0.28
        feats[f"dnf_pct_last{w}"] = rolling_mean(dnfs, w) or 0.1
        feats[f"laps_led_pct_last{w}"] = rolling_mean(laps_led_pct, w) or 0.0

    feats["avg_finish_season"] = float(np.mean(finishes))
    feats["races_run"] = float(len(history))

    feats["track_type_avg_finish"] = (
        float(np.mean(track_type_finishes)) if track_type_finishes else feats["avg_finish_last20"]
    )
    feats["track_type_races"] = float(len(track_type_finishes))

    # Loop-stats-derived rolling features (last 10) - see module docstring. Defaults
    # match a mid-pack driver profile, same fallback philosophy as every other default.
    feats["avg_running_pos_last10"] = rolling_mean(avg_ps_vals, 10) or 20.0
    feats["closing_pos_last10"] = rolling_mean(closing_ps_vals, 10) or 20.0
    feats["quality_passes_last10"] = rolling_mean(quality_passes_vals, 10) or 20.0
    feats["driver_rating_last10"] = rolling_mean(rating_vals, 10) or 75.0
    feats["qualifying_speed_last10"] = rolling_mean(qual_speed_vals, 10) or 0.0

    most_recent = max(race_dates)
    rest_days = (as_of_utc - most_recent).total_seconds() / 86400
    feats["rest_days"] = min(rest_days, 60.0)  # NASCAR's off-season gap dwarfs any
    # in-season rest signal - capped so one outlier doesn't blow up the feature's scale.

    return feats


def _fill_defaults(feats: dict[str, Any]) -> None:
    for w in _WINDOWS:
        feats[f"avg_finish_last{w}"] = 20.0
        feats[f"win_pct_last{w}"] = 0.03
        feats[f"top5_pct_last{w}"] = 0.14
        feats[f"top10_pct_last{w}"] = 0.28
        feats[f"dnf_pct_last{w}"] = 0.1
        feats[f"laps_led_pct_last{w}"] = 0.0
    feats["avg_finish_season"] = 20.0
    feats["races_run"] = 0.0
    feats["track_type_avg_finish"] = 20.0
    feats["track_type_races"] = 0.0
    feats["avg_running_pos_last10"] = 20.0
    feats["closing_pos_last10"] = 20.0
    feats["quality_passes_last10"] = 20.0
    feats["driver_rating_last10"] = 75.0
    feats["qualifying_speed_last10"] = 0.0
    feats["rest_days"] = 7.0
