# sports-prediction-model

Feature engineering, model training, and evaluation code for an NBA/MLB
sports prediction system. This is the open-source model core extracted
from a larger private application (API, web frontend, Discord bot,
billing, notifications) — this repo contains only the ML: how features
are built from raw box scores/schedules and how the game-winner and
player-prop models are trained and evaluated.

## What's here

- **`src/features/`** — feature engineering for NBA and MLB (team form,
  matchup context, player-level rolling stats) built from historical
  schedules, box scores, and rosters.
- **`src/models/train_winner.py`** — trains the game-winner model: an
  XGBoost + logistic-regression stacked ensemble, isotonic-calibrated.
- **`src/models/train_props.py`** — trains player-prop quantile models
  (LightGBM) producing a full predictive distribution (10th/25th/50th/
  75th/90th percentile) per stat.
- **`src/models/eval/`** — walk-forward cross-validation, calibration
  and ranking metrics (log-loss, Brier, ECE, pinball loss, coverage).
- **`src/models/configs/`** — hyperparameter search spaces and feature
  lists per sport/target.
- **`src/models/registry.py`** — MLflow experiment tracking wrapper.

## What's not here

This code is published for algorithmic transparency, not as a
turnkey, `pip install`-and-run package. It was extracted from a
monorepo where feature functions query a live Postgres database
(via SQLAlchemy `Session`) and pull odds/weather data from an ingest
layer — none of that DB schema, ingest layer, or serving/API code is
included. To run this end-to-end you'd need to supply your own
data-access layer that feeds `pandas.DataFrame`s shaped like what
these functions expect.

Updates are published every few days as the production models evolve.

## License

MIT — see [LICENSE](LICENSE).
