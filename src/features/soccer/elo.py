"""Soccer-specific Elo rating engine (SOCCER.md §4).

Not a direct reuse of `src/features/common.py`'s `compute_elo_series` - that function hardcodes a
single NBA-shaped home-court constant (`_HOME_ADVANTAGE = 100.0`), calls `elo_update` with the
module's fixed `_K_FACTOR`, treats `result_col` as strictly binary, and keeps one global rating
pool. Soccer needs:

- **Per-league home advantage** - soccer's HCA varies more by league than any US sport's (this is
  a *rating-update* constant, distinct from the `home_venue_win_pct` per-club feature in
  `SOCCER_WINNER_FEATURES`, which measures the same phenomenon a different way).
- **A goal-margin-scaled K** - the standard World Football Elo Ratings convention (eloratings.net):
  a 2-goal margin is worth 1.5x a 1-goal margin, evidence scales further past that. A 4-0 is more
  evidence of a real quality gap than a 1-0.
- **Draws as 0.5**, not skipped or coerced to a binary outcome.
- **Five separate rating pools, keyed implicitly by team** - clubs don't move between the five
  top-flight leagues this project tracks, so keying by `team_id` alone is sufficient; the
  per-competition behavior lives in the home-advantage lookup and the promoted-team cold start.
- **A promoted-team cold-start prior** - promoted sides systematically underperform their
  cold-start rating; starting them at the league-average 1500 overstates them. Same
  `_fill_defaults()`-style "degrade gracefully, never hard-fail" idiom used throughout this
  integration, applied to rating initialization instead of a missing-feature default.

`elo_implied_probs`' draw-probability curve is a documented heuristic (Hvattum & Arntzen 2010
"Using ELO ratings for match result prediction in association football" use a similar closeness-
based draw peak), not a calibrated model in its own right - these three numbers are *inputs* to
the gradient-boosted winner model, not the final prediction, so a defensible approximation is
enough; backtesting may reveal the constants below need tuning.
"""

from __future__ import annotations

import math

import pandas as pd

from src.features.common import elo_expected, elo_update

_DEFAULT_ELO = 1500.0
_BASE_K = 32.0
_PROMOTED_COLD_START_PENALTY = 80.0

# World-Football-Elo-style HCA point estimates per league (domestic top-flight averages skew
# lower than international-fixture HCA figures quoted elsewhere) - starting defaults, expected to
# need backtest calibration per league, same as _BASE_K.
_LEAGUE_HOME_ADVANTAGE: dict[str, float] = {
    "eng.1": 59.0,
    "esp.1": 55.0,
    "ger.1": 51.0,
    "ita.1": 57.0,
    "fra.1": 51.0,
}
_DEFAULT_HOME_ADVANTAGE = 55.0

# Peak draw probability (at elo_diff == 0) and how fast it decays as the gap widens - fit to
# roughly match top-5-league season-average draw rates (~24-27%), not backtested yet.
_DRAW_PEAK = 0.28
_DRAW_DECAY_SCALE = 500.0


def league_home_advantage(competition_code: str) -> float:
    return _LEAGUE_HOME_ADVANTAGE.get(competition_code, _DEFAULT_HOME_ADVANTAGE)


def _goal_margin_k(base_k: float, goal_diff: int) -> float:
    margin = abs(goal_diff)
    if margin <= 1:
        multiplier = 1.0
    elif margin == 2:
        multiplier = 1.5
    else:
        multiplier = (11.0 + margin) / 8.0
    return base_k * multiplier


def compute_soccer_elo_series(
    games_df: pd.DataFrame,
    competition_col: str = "competition",
    home_col: str = "home_team_id",
    away_col: str = "away_team_id",
    home_score_col: str = "home_score",
    away_score_col: str = "away_score",
    date_col: str = "scheduled_utc",
    promoted_team_ids: set[int] | None = None,
) -> dict[int, float]:
    """Replay a chronologically sorted sequence of games into final Elo ratings.

    `promoted_team_ids` should contain every (team_id) that is promoted into *some* season in
    `games_df` - the cold-start penalty only applies the first time that team is seen, so a team
    promoted in 2019 and still active in 2024 is unaffected by being in this set.

    Returns `{team_id: elo_rating}`, without home-advantage baked in - callers add
    `league_home_advantage(competition_code)` at prediction time, same convention as
    `src/features/common.py::compute_elo_series`.
    """
    ratings: dict[int, float] = {}
    promoted_team_ids = promoted_team_ids or set()
    seen: set[int] = set()

    def cold_start(team_id: int) -> float:
        if team_id in promoted_team_ids and team_id not in seen:
            return _DEFAULT_ELO - _PROMOTED_COLD_START_PENALTY
        return _DEFAULT_ELO

    for _, row in games_df.sort_values(date_col).iterrows():
        home_id = int(row[home_col])
        away_id = int(row[away_col])
        home_score = row[home_score_col]
        away_score = row[away_score_col]
        if pd.isna(home_score) or pd.isna(away_score):
            continue

        home_elo = ratings.get(home_id, cold_start(home_id))
        away_elo = ratings.get(away_id, cold_start(away_id))
        seen.add(home_id)
        seen.add(away_id)

        hca = league_home_advantage(row[competition_col])
        goal_diff = int(home_score) - int(away_score)
        score_a = 1.0 if goal_diff > 0 else (0.0 if goal_diff < 0 else 0.5)
        k = _goal_margin_k(_BASE_K, goal_diff)

        home_adj = home_elo + hca
        new_home_adj, new_away = elo_update(home_adj, away_elo, score_a, k=k)
        ratings[home_id] = new_home_adj - hca
        ratings[away_id] = new_away

    return ratings


def compute_soccer_elo_snapshots(
    games_df: pd.DataFrame,
    game_id_col: str = "id",
    competition_col: str = "competition",
    home_col: str = "home_team_id",
    away_col: str = "away_team_id",
    home_score_col: str = "home_score",
    away_score_col: str = "away_score",
    date_col: str = "scheduled_utc",
    promoted_team_ids: set[int] | None = None,
) -> tuple[dict[int, tuple[float, float]], dict[int, dict[int, float]]]:
    """Same one-pass replay as `compute_soccer_elo_series`, but returns each game's *pre-game*
    (home_elo, away_elo) rather than only the final ratings after the last game, plus a full
    ratings-table snapshot per game (every team's rating immediately before that game kicked off).

    Exists for bulk training-data assembly: calling `build_matchup_features` once per historical
    game (as live scoring does) would make each call re-fetch and re-replay the *entire* Elo
    history from scratch via a fresh DB query - fine for one game at a time, but O(n^2) across
    ~20k games. This computes every game's Elo inputs in a single O(n) pass instead; callers pass
    the per-game tuple into `build_matchup_features`'s `home_elo`/`away_elo` overrides to skip its
    own recompute.

    The full ratings-table snapshot exists for strength-of-schedule: to know how good a team's
    *opponent* was at the moment they actually played (not the opponent's current rating), you
    need the ratings table as it stood right before that specific historical game - which isn't
    otherwise recoverable, since Elo is computed in Python and never persisted per-game in the DB.
    ~21k games x ~150 teams x one float is a few hundred MB at most, not a real memory concern.
    """
    ratings: dict[int, float] = {}
    promoted_team_ids = promoted_team_ids or set()
    seen: set[int] = set()
    snapshots: dict[int, tuple[float, float]] = {}
    full_ratings_snapshots: dict[int, dict[int, float]] = {}

    def cold_start(team_id: int) -> float:
        if team_id in promoted_team_ids and team_id not in seen:
            return _DEFAULT_ELO - _PROMOTED_COLD_START_PENALTY
        return _DEFAULT_ELO

    for _, row in games_df.sort_values(date_col).iterrows():
        home_id = int(row[home_col])
        away_id = int(row[away_col])
        home_score = row[home_score_col]
        away_score = row[away_score_col]

        home_elo = ratings.get(home_id, cold_start(home_id))
        away_elo = ratings.get(away_id, cold_start(away_id))
        seen.add(home_id)
        seen.add(away_id)
        game_id = int(row[game_id_col])
        snapshots[game_id] = (home_elo, away_elo)
        full_ratings_snapshots[game_id] = dict(ratings)

        if pd.isna(home_score) or pd.isna(away_score):
            continue

        hca = league_home_advantage(row[competition_col])
        goal_diff = int(home_score) - int(away_score)
        score_a = 1.0 if goal_diff > 0 else (0.0 if goal_diff < 0 else 0.5)
        k = _goal_margin_k(_BASE_K, goal_diff)

        home_adj = home_elo + hca
        new_home_adj, new_away = elo_update(home_adj, away_elo, score_a, k=k)
        ratings[home_id] = new_home_adj - hca
        ratings[away_id] = new_away

    return snapshots, full_ratings_snapshots


def elo_implied_probs(
    home_elo: float, away_elo: float, competition_code: str
) -> tuple[float, float, float]:
    """(P(home win), P(draw), P(away win)) from Elo ratings - see module docstring for why the
    draw term is a documented heuristic, not a calibrated model."""
    hca = league_home_advantage(competition_code)
    diff = (home_elo + hca) - away_elo
    p_home_2way = elo_expected(home_elo + hca, away_elo)

    p_draw = _DRAW_PEAK * math.exp(-(diff**2) / (2 * _DRAW_DECAY_SCALE**2))
    remaining = 1.0 - p_draw
    p_home = remaining * p_home_2way
    p_away = remaining * (1.0 - p_home_2way)
    return p_home, p_draw, p_away
