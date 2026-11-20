# Forecasts every trophy's earn-rate quantiles at a case's target from the case's own history.
"""The forecaster.

For each trophy, on the logit scale, against x = ln(1 + hours since 2026-11-19T00:00Z):

1. Observations are the distinct Exophase refreshes (series.py); the tiny pre-launch page is left out.
2. A Huber line is fitted on the last WINDOW hours of refreshes (trend.py). Its slope is shrunk toward the median
   slope of the trophy's group (groups.py) by empirical Bayes: the pooled drift of the cross-section.
3. The slope is taken to decay exponentially in log-time, b(x) = b * exp(-lam * (x - x_mid)), x_mid the window's
   centre (where a window's slope is measured). Carried forward from the last refresh x_last over dx, the move is
       b * exp(-lam * delta) * (1 - exp(-lam * dx)) / lam,   delta = x_last - x_mid
   (b * dx when lam = 0). dx runs to the target less half the usual refresh gap, since the answer shows the page as
   of Exophase's last refresh before the target. The median is the last refresh's logit plus that move.
4. lam and the spread come from rolling backtests inside the history: the history is cut at several earlier points,
   each cut forecasts the last refresh, and the logit errors are measured per trophy and per group. The lam with the
   least error (shrunk toward a prior) is kept; because the decay rule is written in terms of dx, a lam picked on
   short in-history horizons carries to the final's long one. The error spread is modelled as
   sigma^2 = sigma0^2 + (k * dx)^2: sigma0 the fit's own noise, k the error growth per unit of log-time, so the
   error grows linearly in log-time (a slope error carried forward). That is the rule that scales the backtests'
   short horizons up to the final's 4.5 days; a safety margin and floors are added on top, since no backtest
   inside the history reaches that far.
5. Quantiles are a normal on the logit scale (quantiles.py), clamped, rounded and made monotone.
"""

import math

import groups
import quantiles
import series
import trend

PARAMS = {
    # Hours of recent refreshes a trend is fitted on.
    "window_hours": 12.0,
    # Weight of the last refresh (vs. the fitted line at that moment) in the anchor.
    "anchor_weight": 1.0,
    # Slope decay rates (per unit log-time) the in-history backtest chooses from, and the prior it is shrunk toward.
    "decay_grid": (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    "decay_prior": 0.75,
    # Relative error penalty per unit of distance from the prior decay.
    "decay_penalty": 0.2,
    # Hours before the last refresh at which the history is cut for the in-history backtest.
    "cut_lags": (6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 30.0, 36.0),
    # Least hours of usable history a cut needs before it.
    "cut_min_hours": 6.0,
    # Error growth per unit log-time when no backtest can be run, and its floor otherwise.
    "k_prior": 0.9,
    "k_floor": 0.25,
    # Smallest spread (logit) at any horizon.
    "sigma_floor": 0.10,
    # Multiplier on every spread: the horizon is far beyond what the backtests verify.
    "margin": 1.15,
    # Spread for a trophy with no rate anywhere in the history.
    "unknown_sigma": 2.0,
}


class TrophyState:
    """What one trophy's history says at one cut: anchor, slope pieces, noise."""

    __slots__ = ("anchor", "x_last", "x_mid", "slope", "se", "scale", "last_pct", "players")

    def __init__(self):
        self.anchor = None
        self.x_last = 0.0
        self.x_mid = 0.0
        self.slope = None
        self.se = None
        self.scale = trend.MIN_SCALE
        self.last_pct = None
        self.players = 0.0


def _state_for(points: list, window_hours: float, anchor_weight: float) -> TrophyState:
    state = TrophyState()
    if not points:
        return state
    last = points[-1]
    state.x_last = last[1]
    state.x_mid = last[1]
    state.last_pct = last[3]
    state.players = last[4]
    state.anchor = last[2]
    end = last[0]
    recent = [p for p in points if p[0] >= end - window_hours]
    if len(recent) < 3 or recent[-1][0] - recent[0][0] < 3.0:
        recent = points
    span_hours = recent[-1][0] - recent[0][0]
    fit = trend.robust_line([p[1] for p in recent], [p[2] for p in recent], span_hours)
    if fit is None:
        return state
    state.slope = fit.slope
    state.x_mid = sum(p[1] for p in recent) / len(recent)
    state.se = fit.slope_se
    state.scale = fit.scale
    state.anchor = anchor_weight * last[2] + (1.0 - anchor_weight) * fit.at(last[1])
    return state


def trophy_states(refreshes: list, n: int, pools: list, params: dict) -> tuple:
    """States of every trophy over the given refreshes, with slopes shrunk across the cross-section."""
    kept = series.usable(refreshes)
    states = [
        _state_for(series.trophy_series(kept, i), params["window_hours"], params["anchor_weight"])
        for i in range(n)
    ]
    shrunk, _ = trend.shrink_slopes(
        [s.slope for s in states],
        [s.se if s.se is not None else 1.0 for s in states],
        pools,
    )
    return states, shrunk


def decay_factor(lam: float, dx: float, delta: float) -> float:
    """Share of slope * dx that a slope decaying at lam (from the window's centre) still delivers over dx."""
    if dx <= 1e-9:
        return 0.0
    if lam <= 1e-9:
        return 1.0
    return math.exp(-lam * max(delta, 0.0)) * (1.0 - math.exp(-lam * dx)) / (lam * dx)


class Calibration:
    """What the in-history backtest found."""

    def __init__(self):
        self.decay = None
        self.k_group = {}
        self.k_trophy = {}
        self.cuts = 0


def _cut_errors(refreshes: list, n: int, pools: list, params: dict) -> list:
    """For each usable cut, per trophy: (truth - anchor, shrunk slope * dx, dx, delta, noise) or None."""
    usable_all = series.usable(refreshes)
    if len(usable_all) < 4:
        return []
    end = usable_all[-1]
    start_hours = usable_all[0].hours
    out = []
    for lag in params["cut_lags"]:
        cut_hours = end.hours - lag
        if cut_hours - start_hours < params["cut_min_hours"]:
            continue
        sub = [r for r in refreshes if r.hours <= cut_hours]
        if len(series.usable(sub)) < 3:
            continue
        states, shrunk = trophy_states(sub, n, pools, params)
        pieces = []
        for i in range(n):
            st = states[i]
            truth = end.rates[i]
            if st.anchor is None or truth is None:
                pieces.append(None)
                continue
            dx = max(end.x - st.x_last, 0.0)
            gap = series.logit(truth) - st.anchor
            pieces.append((gap, shrunk[i] * dx, dx, st.x_last - st.x_mid, st.scale))
        out.append(pieces)
    return out


def _choose_decay(cuts: list, params: dict) -> float:
    """The decay with the least horizon-weighted absolute error over the cuts, penalised away from the prior."""
    best = None
    for lam in params["decay_grid"]:
        total = 0.0
        weight = 0.0
        for pieces in cuts:
            for item in pieces:
                if item is None:
                    continue
                gap, move, dx, delta, _ = item
                total += dx * abs(gap - move * decay_factor(lam, dx, delta))
                weight += dx
        if weight <= 0:
            continue
        score = (total / weight) * (1.0 + params["decay_penalty"] * abs(lam - params["decay_prior"]))
        if best is None or score < best[0] - 1e-12:
            best = (score, lam)
    return best[1] if best else params["decay_prior"]


def calibrate(refreshes: list, n: int, pools: list, params: dict) -> Calibration:
    cal = Calibration()
    cuts = _cut_errors(refreshes, n, pools, params)
    cal.cuts = len(cuts)
    if not cuts:
        return cal
    cal.decay = _choose_decay(cuts, params)
    # Squared errors at the chosen decay, net of each trophy's own noise, per unit of log-time squared.
    per_group = {}
    per_trophy = {}
    for pieces in cuts:
        for i, item in enumerate(pieces):
            if item is None:
                continue
            gap, move, dx, delta, scale = item
            if dx <= 1e-6:
                continue
            err = gap - move * decay_factor(cal.decay, dx, delta)
            k2 = max(err * err - scale * scale, 0.0) / (dx * dx)
            per_group.setdefault(pools[i], []).append(k2)
            per_trophy.setdefault(i, []).append(k2)
    for pool, values in per_group.items():
        # A trimmed mean: the top tenth of squared errors is capped, so one broken trophy does not set a group.
        ordered = sorted(values)
        cap = ordered[int(0.9 * (len(ordered) - 1))]
        cal.k_group[pool] = math.sqrt(sum(min(v, cap) for v in ordered) / len(ordered))
    for i, values in per_trophy.items():
        cal.k_trophy[i] = math.sqrt(sum(values) / len(values))
    return cal


def _count_variance(pct: float, players: float) -> float:
    """Binomial noise of a rate over the tracked pool, on the logit scale."""
    if players <= 0:
        return 0.0
    p = series.clamp(pct) / 100.0
    return 1.0 / max(players * p * (1.0 - p), 1.0)


def _growth(cal: Calibration, i: int, pool: str, params: dict) -> float:
    """Error growth per unit log-time for trophy i: its own backtest errors shrunk toward its group's."""
    if not cal.cuts:
        return params["k_prior"]
    k_pool = cal.k_group.get(pool, params["k_prior"])
    k_own = cal.k_trophy.get(i, k_pool)
    k2 = (2.0 * k_own * k_own + 3.0 * k_pool * k_pool) / 5.0
    return max(math.sqrt(k2), params["k_floor"])


def forecast_case(case: dict, params: dict = None) -> dict:
    params = dict(PARAMS if params is None else params)
    trophies = case["trophies"]
    n = len(trophies)
    pools = [groups.POOL_OF[g] for g in groups.classify_all(trophies)]
    refreshes = series.build_refreshes(case.get("history") or [], n)
    gap = series.typical_gap_hours(series.usable(refreshes))
    target_hours = series.hours_since_anchor(case["target"]) - 0.5 * gap
    x_target = series.log_time(target_hours)

    states, shrunk = trophy_states(refreshes, n, pools, params)
    cal = calibrate(refreshes, n, pools, params)
    lam = cal.decay if cal.decay is not None else params["decay_prior"]

    known = [s.anchor for s in states if s.anchor is not None]
    fallback_level = trend.median(known) if known else series.logit(1.0)
    rows = []
    for i, trophy in enumerate(trophies):
        st = states[i]
        if st.anchor is None:
            rows.append(quantiles.row(trophy["id"], fallback_level, params["unknown_sigma"], params["unknown_sigma"]))
            continue
        dx = max(x_target - st.x_last, 0.0)
        centre = st.anchor + shrunk[i] * dx * decay_factor(lam, dx, st.x_last - st.x_mid)
        k = _growth(cal, i, pools[i], params)
        var = st.scale ** 2 + (k * dx) ** 2 + _count_variance(st.last_pct, st.players)
        sigma = max(math.sqrt(var), params["sigma_floor"]) * params["margin"]
        rows.append(quantiles.row(trophy["id"], centre, sigma, sigma))
    return {"trophies": rows}
