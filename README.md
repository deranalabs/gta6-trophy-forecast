# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/deranalabs/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/deranalabs/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for make.fun's `gta6-trophy-rarity-forecast-01` bounty. For every Grand Theft Auto VI trophy, it forecasts the percent of players holding that trophy at 2026-11-26T00:00Z as five quantiles. The forecasts come from the Exophase refreshes the house recorded. Each case's history is read as distinct page refreshes: repeated reads of the same cached page count once, the tiny launch-night page is left out, and prints stuck at 0% or 100% are not used to fit trends.

The model fits a robust (Huber) line to each trophy's rate over the last 12 hours, against log-time (the log of one plus the hours since launch day), on the logit and probit scales. Each slope is pulled toward the median drift of its trophy group (story, strand, grind, misc, platinum), then decays as it is carried forward to the target. In-history calibration cuts the case's own history at earlier points and forecasts the last refresh from each cut. Those cuts choose the link and the decay rate, and they measure how fast the error grows. The spread comes from that observed error, with a safety margin, and its shape comes from the in-history errors. The platinum's forecast is capped at no more than any other trophy's. If the model fails on a case, the program falls back to the house baseline's formula, so every output stays valid.

There is no real multi-day history to backtest against. All backtest results in NOTES.md come from synthetic worlds built by `tests/simulate.py`, not from real data. NOTES.md covers the model, those results, what could not be verified, and the risks.

## Run it

Python 3.13, standard library only.

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<dir> MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
MAKEFUN_KIT_INPUT=<kit>/input python3 -m unittest discover -s tests -v
```

## Layout

- `main.py`: reads every `*.in` in name order and writes one validated `.out` each, with the house formula as fallback.
- `model.py`: the forecaster, with every setting in one `PARAMS` dict.
- `series.py`: turns the reads into distinct Exophase refreshes on the log-time axis.
- `groups.py`: sorts trophies into story, strand, grind, misc and platinum groups for pooling.
- `trend.py`: Huber line fits and empirical-Bayes slope shrinkage.
- `links.py`: the logit and probit scales.
- `quantiles.py`: builds quantile rows (clamped, rounded, monotone) and validates the output.
- `tests/backtest.py`: cuts a history into synthetic cases and scores the model against both house baselines.
- `tests/baselines.py`: the starter and a reading of the house curve baseline.
- `tests/simulate.py`: a synthetic launch-week world, not real data.
- `tests/sweep.py`: runs the backtest over a list of parameter settings.
- `tests/test_forecast.py`, `tests/cases.py`: unit tests and their synthetic cases.
- `NOTES.md`: the model, backtest results, limits and risks.

MIT licensed, see LICENSE.
