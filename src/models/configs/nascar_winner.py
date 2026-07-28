"""Canonical feature list for the NASCAR race-winner model.

v2 (2026-07-23): expanded from a 25-feature first pass after it evaluated at 68%
top-5 hit rate on a 25-race holdout (below the 75% target - and that number turned
out to be small-sample noise, the real figure on a 66-race holdout was 54.5%). Added
loop-stats-derived rolling features (already collected during ingestion into
RaceEntry.meta but not previously turned into features), manufacturer-level recent
form, an explicit Next-Gen-car era flag, and one-hot track-type dummies. Shared by
both Cup and Truck Series models - see src/features/nascar/driver.py's docstring for
why the underlying feature *queries* need a series filter even though the feature
*names* here are series-agnostic.
"""

NASCAR_WINNER_FEATURES = [
    "elo",
    "starting_position",
    "avg_finish_last5",
    "avg_finish_last10",
    "avg_finish_last20",
    "win_pct_last5",
    "win_pct_last10",
    "win_pct_last20",
    "top5_pct_last5",
    "top5_pct_last10",
    "top5_pct_last20",
    "top10_pct_last5",
    "top10_pct_last10",
    "top10_pct_last20",
    "dnf_pct_last5",
    "dnf_pct_last10",
    "dnf_pct_last20",
    "laps_led_pct_last5",
    "laps_led_pct_last10",
    "laps_led_pct_last20",
    "avg_finish_season",
    "races_run",
    "track_type_avg_finish",
    "track_type_races",
    "rest_days",
    "era_next_gen",
    "track_is_superspeedway",
    "track_is_short_track",
    "track_is_road_course",
    "track_is_intermediate",
    "manufacturer_win_pct",
    "manufacturer_top5_pct",
    "manufacturer_avg_finish",
    "avg_running_pos_last10",
    "closing_pos_last10",
    "quality_passes_last10",
    "driver_rating_last10",
    "qualifying_speed_last10",
]
