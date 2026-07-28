# sports-prediction-model

Feature engineering, model training, and evaluation code for a multi-sport
prediction system covering NBA, WNBA, MLB, NFL, NHL, top-5 European soccer
leagues, and NASCAR (Cup/Truck). This is the open-source model core
extracted from a larger private application (API, web frontend, Discord
bot, billing, notifications) — this repo contains only the ML: how
features are built from raw box scores/schedules and how the game-winner
and player-prop models are trained and evaluated.

## What's here

- **`src/features/`** — feature engineering per sport (team form, matchup
  context, player-level rolling stats) built from historical schedules,
  box scores, and rosters. Soccer additionally has its own Elo module
  (`src/features/soccer/elo.py`); NASCAR's per-driver/per-race shape
  lives in `src/features/nascar/driver.py` and `field.py`.
- **`src/models/train_winner.py`** — trains the game-winner model: an
  XGBoost + LightGBM + CatBoost stacked ensemble with a logistic
  meta-learner, isotonic-calibrated. Shared by every sport with a binary
  home/away target (NBA, WNBA, MLB, NFL, NHL, NASCAR) — small-sample
  sports (NFL, NASCAR) get tightened hyperparameter search bounds and a
  wider promotion margin, since a short season doesn't carry enough
  holdout games to trust a tight one.
- **`src/models/train_soccer_winner.py`** — soccer's winner model is a
  separate 3-way (home/draw/away) multi:softprob XGBoost classifier
  rather than the binary StackedEnsemble above, since a draw outcome
  doesn't fit a two-class shape.
- **`src/models/train_anytime_td.py`** — binary classifier for
  anytime-touchdown-scorer props (NFL); a probability-of-event target
  rather than the quantile-regression shape `train_props.py` uses for
  counting-stat props.
- **`src/models/train_props.py`** — trains player-prop quantile models
  (LightGBM) producing a full predictive distribution (10th/25th/50th/
  75th/90th percentile) per stat.
- **`src/models/eval/`** — walk-forward cross-validation, calibration
  and ranking metrics (log-loss, Brier, ECE, pinball loss, coverage),
  including a multiclass variant for soccer's 3-way market.
- **`src/models/configs/`** — hyperparameter search spaces and feature
  lists per sport/target.
- **`src/models/registry.py`** — MLflow experiment tracking wrapper.

## What's not here

This code is published for algorithmic transparency, not as a
turnkey, `pip install`-and-run package. It was extracted from a
monorepo where feature functions query a live Postgres database
(via SQLAlchemy `Session`) and pull odds/weather data from an ingest
layer — none of that DB schema, ingest layer, or serving/API code is
included. That also means the training-data-*assembly* step (querying
the database for every completed game/race and building the full
training `DataFrame`) isn't included for any sport; every `train_*`
entrypoint here picks up from an already-assembled `pandas.DataFrame`.
To run this end-to-end you'd need to supply your own data-access layer
that feeds `pandas.DataFrame`s shaped like what these functions expect.

Updates are published every few days as the production models evolve.

## License

MIT — see [LICENSE](LICENSE).
