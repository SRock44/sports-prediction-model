"""Canonical feature list for the NASCAR finishing-position quantile model.

Same feature set as the winner model (src/models/configs/nascar_winner.py) - nothing
props-specific needed, finishing position is predicted from the same driver-form
signal that predicts win probability. Kept as a separate list (not a re-export) since
the two models are trained independently and could legitimately diverge later.
"""

NASCAR_PROPS_FEATURES = [
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
