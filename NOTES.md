<!-- Explains the forecaster: the model, what the backtests showed, what could not be verified, and the risks. -->
# GTA VI trophy earn-rate forecast — notes

Entry for make.fun `gta6-trophy-rarity-forecast-01`. `python3 -P -s main.py`, Python 3.13 standard library only,
no network, deterministic (no randomness at all; same bytes under any `PYTHONHASHSEED`).

## Layout

| file | job |
|---|---|
| `main.py` | reads every `*.in` in name order, writes one validated `.out` each (house formula as fallback) |
| `model.py` | the forecaster (below); every setting in one `PARAMS` dict |
| `series.py` | reads into distinct Exophase refreshes; time axis x = ln(1 + hours since 2026-11-19T00:00Z) |
| `groups.py` | story / strand / grind / misc / platinum from tier and description, for pooling |
| `trend.py` | Huber line fits, empirical-Bayes slope shrinkage |
| `links.py` | logit and probit scales for trends |
| `quantiles.py` | quantile rows (clamped, rounded half up, monotone) and the output validator |
| `tests/backtest.py` | cuts a history into synthetic cases, scores model vs. both house baselines by the contest loss |
| `tests/baselines.py` | the starter and my reading of the house curve baseline |
| `tests/simulate.py` | a synthetic launch-week world (players, play, refreshes, reads), **not real data** |
| `tests/sweep.py` | backtest over a list of parameter settings |
| `tests/test_forecast.py`, `tests/cases.py` | unit tests and their synthetic cases |

Run: `mkdir -p out && MAKEFUN_INPUT_DIR=<dir> MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py`.
Tests: `MAKEFUN_KIT_INPUT=<kit>/input python3 -m unittest discover -s tests -v`.
Backtest: `python3 tests/backtest.py [--scheme inside|final|all] [--set key=value] FILE...`.

## The model

Per trophy, against x = ln(1 + hours since launch day 00:00Z), the house curve baseline's axis:

1. **Observations are refreshes, not reads.** Exophase holds a page for hours, so identical consecutive reads count
   once, placed halfway between the last old read and the first new one. The 638-player launch-night page (any
   refresh with under 25% of the latest players-tracked count) is left out of fits. Prints at the clamps (0.00/0.01%,
   99.99/100%) are censored out of fits: their logit only bounds the rate.
2. **Robust local trend.** Huber line of link(rate) on x over the last 12 hours of refreshes, on both the logit and
   the probit scale. Slope standard errors count one independent point per 3 hours (diurnal swings are correlated).
3. **Pooled drift across the cross-section.** Each slope is shrunk toward its group's median slope with weights
   1/se² and 1/τ², τ² the group's between-trophy slope variance net of sampling noise. Slopes are then clipped to the
   cross-section's 10th–90th percentile (a trophy that just appeared off the floor has a huge, short-lived slope).
   A trophy with no fit takes its group's slope: half of it, upward only, if it sits at the floor (grind trophies
   emerge over days); none at the ceiling (the 100% trophy stays put unless its own data says otherwise).
4. **Decaying slope.** b(x) = b·e^(−λ(x − x_mid)), x_mid the fit window's centre. Over dx from the last refresh the
   median moves b·e^(−λδ)·(1 − e^(−λdx))/λ, δ = x_last − x_mid. dx ends at the target less half the usual gap between
   refreshes (the answer shows the page as of Exophase's last refresh before 00:00Z).
5. **In-history calibration.** The case's history is cut at 6, 9, …, 36 hours before its last refresh; each cut
   forecasts the last refresh. On those cuts the model picks (link ∈ {logit, probit, blend}, λ ∈ {0 … 3}) by
   horizon-weighted absolute logit error, shrunk toward the prior (blend, λ = 0.75). λ is written in terms of dx,
   so a choice made on 0.1–0.5 units of log-time carries to the final's ~1.0 unit (2026-11-21T11:30Z → 11-26T00:00Z
   is ln(169/60.5) ≈ 1.03).
6. **Spread.** σ² = σ0² + (k·dx)² + 1/(N·p(1−p)): σ0 the fit's residual noise, k the in-history error growth per
   unit of log-time (per group, trimmed, and per trophy, shrunk 3:2 toward the group), the last term the binomial
   noise over the tracked pool. **Scaling rule:** error grows linearly in log-time (a slope error carried forward),
   which is more conservative than a random walk's √dx, from the backtests' short horizons to the final's. Floors:
   k ≥ 0.25, σ ≥ 0.10; then a **safety margin ×1.15** on every σ.
7. **Shape.** In-history errors divided by their predicted σ give an empirical shape (heavier tails, narrower
   quartiles than a normal). The five z values are half that shape and half the normal's.
8. **Structure.** The platinum's quantiles never exceed any other trophy's at the same level (a platinum holder
   holds every trophy). Rows are clamped to 0..100, rounded to 4 decimals half up and made monotone.

With no usable cut (the practice cases: one distinct page), λ and k take their priors (0.75 and 0.9 logit per
unit log-time), so the practice forecasts are wide: launch night is a long way in log-time.

## What the backtest showed

There is no real multi-day history to backtest on: the practice cases hold one distinct page each, so no cut inside
them has an answer (`tests/backtest.py` says so). All numbers below are on **synthetic worlds from
`tests/simulate.py`**, four scenarios (steady arrivals; a late wave of casual players; slow refreshes with a strong
diurnal rhythm; a fast hard-core pool), two seeds each for tuning (1–2) and two held out (3–4), never tuned on.

Scheme `final` is the final's shape (cut 59.5 h, target 168 h). Scheme `inside` cuts every 6 h from 12 h after launch,
with horizons of 6, 12 and 24 h, inside the first 59.5 h: what the real final's own history can check.

| worlds | scheme | starter | curve | model | model 90% coverage |
|---|---|---|---|---|---|
| tuning (8) | final | 1.398 | 0.885 | **0.475** | 0.87 |
| holdout (8) | final | 1.408 | 0.843 | **0.474** | 0.86 |
| tuning (8) | inside | 0.746 | 0.644 | **0.391** | 0.59 |
| holdout (8) | inside | 0.715 | 0.619 | **0.373** | 0.59 |

The model beats both baselines in every one of the 16 worlds at the final's horizon. On worlds where the curve
baseline's assumption holds exactly (pure log-linear logit trends, `tests/cases.py`), the selection picks λ = 0 and
the model scores 0.209 against the curve's 0.223. Most of its remaining loss there comes from the platinum cap, which
that test world breaks by construction: its "platinum" is a random trophy.

Ideas, one at a time (synthetic final loss; kept only when it helped):

| step | final | inside | kept |
|---|---|---|---|
| first version: 24 h window, damping picked in-history, bias correction, skew | 0.717 | 0.425 | — |
| 12 h window, anchor on last refresh, no bias correction / skew (both added error) | 0.588–0.616 | 0.40–0.41 | yes |
| fixed damping 0.5 / 0.75 / 1.0 instead of in-history choice | 0.697 / 0.588 / 0.728 | | no: too sharp |
| decaying slope, λ chosen in-history (prior 0.75) | 0.598 | 0.410 | yes |
| censor clamp prints out of fits | 0.566 | 0.388 | yes |
| floor trophies take ½ of their group's upward slope (1 = 0.541, 0 = 0.566) | 0.500 | 0.403 | yes, a prior |
| slope clip at the cross-section's 10–90% | 0.481 | 0.397 | yes |
| probit link alone / logit alone / blend | 0.553 / 0.542 / 0.540 | | in-history choice kept |
| platinum cap | ±0.000 | | yes: true structure, costs nothing |
| in-history error shape, half weight (full: 0.472 but coverage 0.80) | 0.475 | 0.382 | yes |
| margin 1.0 / 1.15 / 1.25 / 1.4 | 0.462 / 0.475 / 0.485 / 0.502 | | 1.15, see risks |

## What could not be verified

- **Anything at the real 4.5-day horizon.** The final's history ends 2026-11-21T11:30Z; the longest in-history
  backtest reaches about 0.5 units of log-time, half of the final's. The linear-in-log-time growth of the error, the
  decay of slopes beyond what the history shows, and the 1.15 margin are rules, not measurements.
- **The real dynamics.** The simulator is my own model of players (lognormal play intensity, diurnal rhythm, quits,
  thresholds). Every tuning number above is "good in that world". The choices I trust most are the ones that hold in
  every scenario and on held-out seeds, and that are set from the case's own history at run time (link, λ, k, shape).
- **The floor share (½), slope clip (10–90%), window (12 h) and priors (λ = 0.75, k = 0.9)** come from the simulator,
  not from real GTA VI data.
- **The house curve baseline** is my reading of RULES.txt's description; its real spread formula is not in the kit,
  so my baseline numbers for it are approximate.
- The practice cases only prove the program runs and writes valid output; their answers are not in the kit.

## Risks

- **Regime change after the cut.** A wave of new tracked players (Exophase crawl, a weekend surge, Thanksgiving on
  11-26) can pull early trophies down and bend every trend; the history cannot show it. The late-wave scenario is the
  stand-in, and the model still beats both baselines there, but its loss is higher.
- **Floor trophies** (platinum, 100% completion and the like) dominate the loss: they either emerge or stay at 0.00,
  and the history says little about which. Half the group drift is a compromise.
- **Saturating trophies near 100%** can rise faster on the logit scale than a decaying slope allows.
- **Exophase anomalies** (a partial page, a recount, a stale refresh at 00:00Z) are absorbed only through the
  Huber fit and the spread; a big recount right before the target is not modelled.
- **Validity** is guarded: every case runs through the output validator, and any failure falls back to the house
  baseline's formula, so a bad case scores as persistence rather than invalidating the entry. The run takes about
  0.1–0.6 s on 36–120 hour histories at a 10-minute cadence, far inside the 8-second limit.
