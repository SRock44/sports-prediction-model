"""Canonical feature list for the NFL game-winner model.

75 features - intentionally narrower than NBA's ~120/MLB's ~95, per NFL.md's
own caution that a 272-game season punishes high-dimensionality harder than
the other two sports. Odds features are computed by build_matchup_features
but deliberately excluded here, matching MLB_WINNER_FEATURES's convention
(game_odds is too sparsely populated to trust as a trained feature yet).

Four more features are computed by build_matchup_features but deliberately
excluded here, all for the same reason: they mean something different at
training time than at inference time. Measured on the real backfill
(1,693 final games) vs. real upcoming 2026 games:

  weather_temp_f     85 distinct values in training -> constant 60.0 when
  weather_wind_mph   scoring. Both are read from Game.meta, which nflverse
                     only populates *after* a game is played, so every
                     upcoming game falls back to matchup.py's 60.0F/7.0mph
                     defaults. The model would learn splits on a signal that
                     is constant at inference.

  home_qb_confirmed  1408x true / 285x false in training -> false on *every*
  away_qb_confirmed  upcoming game. _resolve_starting_qb can only return
                     confirmed=True from Game.meta's post-game QB record (or
                     the Lineup table, which no NFL ingest writes yet), so
                     this is structurally 0 when scoring. Worse than a
                     constant: the model had real 83/17 variation to split on
                     and would take the minority branch every single time.

weather_is_dome and qb_known both stay - is_dome comes from the venue's roof
metadata (known months ahead) and qb_known is ~97% true in training and true
when scoring, so both mean the same thing in both places.
"""

NFL_WINNER_FEATURES = [
    # ── Home team ─────────────────────────────────────────────────────────────
    "home_elo",
    "home_is_home",
    "home_travel_km",
    "home_points_scored_last4",
    "home_points_scored_last8",
    "home_points_scored_last17",
    "home_points_allowed_last4",
    "home_points_allowed_last8",
    "home_points_allowed_last17",
    "home_point_diff_last4",
    "home_point_diff_last8",
    "home_point_diff_last17",
    "home_epa_per_play_last4",
    "home_epa_per_play_last8",
    "home_epa_per_play_last17",
    "home_pace_last4",
    "home_pace_last8",
    "home_pace_last17",
    "home_turnovers_committed_last4",
    "home_turnovers_committed_last8",
    "home_turnovers_committed_last17",
    "home_win_pct_season",
    "home_streak",
    "home_rest_days",
    "home_bye_week_just_occurred",
    "home_short_week",
    # ── Away team ─────────────────────────────────────────────────────────────
    "away_elo",
    "away_is_home",
    "away_travel_km",
    "away_points_scored_last4",
    "away_points_scored_last8",
    "away_points_scored_last17",
    "away_points_allowed_last4",
    "away_points_allowed_last8",
    "away_points_allowed_last17",
    "away_point_diff_last4",
    "away_point_diff_last8",
    "away_point_diff_last17",
    "away_epa_per_play_last4",
    "away_epa_per_play_last8",
    "away_epa_per_play_last17",
    "away_pace_last4",
    "away_pace_last8",
    "away_pace_last17",
    "away_turnovers_committed_last4",
    "away_turnovers_committed_last8",
    "away_turnovers_committed_last17",
    "away_win_pct_season",
    "away_streak",
    "away_rest_days",
    "away_bye_week_just_occurred",
    "away_short_week",
    # ── Cross features ────────────────────────────────────────────────────────
    "elo_diff",
    "elo_home_win_prob",
    "point_diff_diff_last8",
    "epa_per_play_diff_last8",
    "pace_diff_last8",
    "turnover_diff_last8",
    "rest_diff",
    "win_pct_season_diff",
    "streak_diff",
    # ── Starting QB (the dominant signal) ─────────────────────────────────────
    "home_qb_form_epa_last8",
    "home_qb_form_ypa_last8",
    "home_qb_form_int_rate_last8",
    "home_qb_form_rush_yds_last8",
    "home_qb_known",
    "away_qb_form_epa_last8",
    "away_qb_form_ypa_last8",
    "away_qb_form_int_rate_last8",
    "away_qb_form_rush_yds_last8",
    "away_qb_known",
    "qb_epa_diff_last8",
    # ── Head-to-head ──────────────────────────────────────────────────────────
    "h2h_home_win_pct",
    "h2h_total",
    # ── Weather ───────────────────────────────────────────────────────────────
    # Only is_dome - see the module docstring for why temp/wind are excluded.
    "weather_is_dome",
]
