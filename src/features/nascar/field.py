"""NASCAR full-field feature assembly: one row per driver-entrant for a single race.

This is the N-entrant analog of every other sport's matchup.py (which assembles
exactly 2 teams' features into one row per game). A race has no "matchup" - it's one
row per driver, all sharing the same race-level context (elo ratings computed once
for the whole field, not per-driver).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from src.core.time import as_of_for_game
from src.features.nascar.driver import (
    build_driver_features,
    compute_driver_elo_ratings,
    compute_manufacturer_form,
)


def build_field_features(
    session: Session, race: Any, as_of: datetime | None = None
) -> pd.DataFrame:
    """Full feature matrix for every entrant in `race`. One row per driver.

    Args:
        race: A Race ORM instance, with `entries` loaded (RaceEntry rows).
        as_of: Cutoff timestamp for feature computation; defaults to 1h before
            green flag (src.core.time.as_of_for_game, sport-agnostic despite the name).
    """
    if as_of is None:
        as_of = as_of_for_game(race.scheduled_utc)

    elo_ratings = compute_driver_elo_ratings(session, race.sport_id, as_of, race.series)
    manufacturer_form = compute_manufacturer_form(session, race.sport_id, as_of, race.series)
    track_name = (race.meta or {}).get("track_name")

    rows: list[dict[str, Any]] = []
    for entry in race.entries:
        feats = build_driver_features(
            session,
            driver_id=entry.driver_id,
            as_of_utc=as_of,
            elo_rating=elo_ratings.get(entry.driver_id, 1500.0),
            starting_position=entry.starting_position,
            track_name=track_name,
            season=race.season,
            series=race.series,
            manufacturer=entry.manufacturer,
            manufacturer_form=manufacturer_form,
        )
        feats["race_id"] = race.id
        feats["driver_id"] = entry.driver_id
        feats["race_entry_id"] = entry.id
        rows.append(feats)

    return pd.DataFrame(rows)
