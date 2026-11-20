# Forecasts every trophy's earn-rate quantiles at a case's target from the case's own history.
"""The forecaster.

For each trophy, against x = ln(1 + hours since 2026-11-19T00:00Z):

1. Observations are the distinct Exophase refreshes (series.py); the tiny pre-launch page is left out, and prints at
   the clamps (0.00 or 0.01%, 99.99 or 100%) are censored out of fits, since their logit says only "at most" or
   "at least".
2. On each link scale (links.py: logit and probit) a Huber line is fitted on the last WINDOW hours (trend.py). Its
   slope is shrunk toward the median slope of the trophy's group (groups.py) by empirical Bayes: the pooled drift of
   the cross-section. A trophy with no fit takes the pool's slope, upward only at the floor and none at the ceiling.
3. The slope decays exponentially in log-time, b(x) = b * exp(-lam * (x - x_mid)), x_mid the window's centre
   (where a window's slope is measured). Carried forward from the last refresh x_last over dx, the move is
       b * exp(-lam * delta) * (1 - exp(-lam * dx)) / lam,   delta = x_last - x_mid
   (b * dx when lam = 0). dx runs to the target less half the usual refresh gap, since the answer shows the page as
   of Exophase's last refresh before the target. The median is the last refresh plus that move, on the link's scale;
   the "blend" choice averages the logit and probit medians on the logit scale.
4. The link choice, lam and the spread come from rolling backtests inside the history: the history is cut at several
   earlier points, each cut forecasts the last refresh, and the logit errors are measured per trophy and per group.
   The (link, lam) with the least error, shrunk toward a prior, is kept; because the decay rule is written in terms of
   dx, a lam picked on short in-history horizons carries to the final's long one. The error spread is modelled as
   sigma^2 = sigma0^2 + (k * dx)^2: sigma0 the fit's own noise, k the error growth per unit of log-time, so the
   error grows linearly in log-time (a slope error carried forward). That is the rule that scales the backtests'
   short horizons up to the final's 4.5 days; a safety margin and floors are added on top, since no backtest
   inside the history reaches that far.
5. Quantiles are a normal on the logit scale (quantiles.py), rounded and made monotone.
"""

import math

import groups
import links
import quantiles
import series
import trend

PARAMS = {
    # Hours of recent refreshes a trend is fitted on.
    "window_hours": 12.0,
    # Weight of the last refresh (vs. the fitted line at that moment) in the anchor.
    "anchor_weight": 1.0,
    # Link choices the in-history backtest picks from, the prior one, and the relative penalty for leaving it.
    "link_choices": ("logit", "probit", "blend"),
    "link_prior": "blend",
    "link_penalty": 0.03,
    # Slope decay rates (per unit log-time) to choose from, the prior, and the relative penalty per unit away from it.
    "decay_grid": (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    "decay_prior": 0.75,
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
    # Prints at or below / at or above these percents are censored out of fits.
    "censor_low": 0.015,
    "censor_high": 99.985,
    # Share of its pool's (upward) slope a trophy stuck at the floor, with no fit of its own, is given.
    "floor_share": 0.5,
    # Shrunk slopes are clipped to this quantile of the cross-section's fitted slopes (and its mirror); 1 = no clip.
    "slope_cap_q": 0.9,
    # Whether the platinum's quantiles are held at or below every other trophy's (it needs them all).
    "platinum_cap": True,
}

LINKS = ("logit", "probit")


class TrophyState:
    """What one trophy's history says on one link scale at one cut."""

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


def _state_for(points: list, link: str, params: dict) -> TrophyState:
    """points are (hours, x, logit, percent, players), oldest first."""
    state = TrophyState()
    if not points:
        return state
    last = points[-1]
    state.x_last = last[1]
    state.x_mid = last[1]
    state.last_pct = last[3]
    state.players = last[4]
    last_value = links.to_link(link, last[3])
    state.anchor = last_value
    end = last[0]
    lo, hi = params["censor_low"], params["censor_high"]
    open_points = [p for p in points if lo < p[3] < hi]
    recent = [p for p in open_points if p[0] >= end - params["window_hours"]]
    if len(recent) < 3 or recent[-1][0] - recent[0][0] < 3.0:
        recent = open_points
    if len(recent) < 3:
        return state
    span_hours = recent[-1][0] - recent[0][0]
    ys = [links.to_link(link, p[3]) for p in recent]
    fit = trend.robust_line([p[1] for p in recent], ys, span_hours)
    if fit is None:
        return state
    state.slope = fit.slope
    state.x_mid = sum(p[1] for p in recent) / len(recent)
    state.se = fit.slope_se
    state.scale = fit.scale * links.logit_per_link(link, last_value)
    weight = params["anchor_weight"]
    state.anchor = weight * last_value + (1.0 - weight) * fit.at(last[1])
    return state


def trophy_states(refreshes: list, n: int, pools: list, params: dict) -> dict:
    """For each link: (states, shrunk slopes) of every trophy over the given refreshes."""
    kept = series.usable(refreshes)
    every = [series.trophy_series(kept, i) for i in range(n)]
    out = {}
    for link in LINKS:
        states = [_state_for(points, link, params) for points in every]
        shrunk, _ = trend.shrink_slopes(
            [s.slope for s in states],
            [s.se if s.se is not None else 1.0 for s in states],
            pools,
        )
        fitted = [s.slope for s in states if s.slope is not None]
        if len(fitted) >= 8 and params["slope_cap_q"] < 1.0:
            hi = trend.quantile(fitted, params["slope_cap_q"])
            lo = trend.quantile(fitted, 1.0 - params["slope_cap_q"])
            shrunk = [min(hi, max(lo, b)) for b in shrunk]
        for i, st in enumerate(states):
            if st.slope is None and st.last_pct is not None:
                if st.last_pct >= params["censor_high"]:
                    shrunk[i] = 0.0
                elif st.last_pct <= params["censor_low"]:
                    shrunk[i] = params["floor_share"] * max(shrunk[i], 0.0)
        out[link] = (states, shrunk)
    return out


def decay_factor(lam: float, dx: float, delta: float) -> float:
    """Share of slope * dx that a slope decaying at lam (from the window's centre) still delivers over dx."""
    if dx <= 1e-9:
        return 0.0
    if lam <= 1e-9:
        return 1.0
    return math.exp(-lam * max(delta, 0.0)) * (1.0 - math.exp(-lam * dx)) / (lam * dx)


def median_logit(by_link: dict, i: int, choice: str, lam: float, x_target: float) -> float:
    """Trophy i's median on the logit scale under a link choice and decay."""
    values = []
    for link in LINKS if choice == "blend" else (choice,):
        states, shrunk = by_link[link]
        st = states[i]
        dx = max(x_target - st.x_last, 0.0)
        moved = st.anchor + shrunk[i] * dx * decay_factor(lam, dx, st.x_last - st.x_mid)
        values.append(links.to_logit(link, moved))
    return sum(values) / len(values)


class Calibration:
    """What the in-history backtest found."""

    def __init__(self):
        self.link = None
        self.decay = None
        self.k_group = {}
        self.k_trophy = {}
        self.cuts = 0


def _cuts(refreshes: list, n: int, pools: list, params: dict) -> list:
    """For each usable cut: (states by link, x of the last refresh, truth logits by trophy)."""
    usable_all = series.usable(refreshes)
    if len(usable_all) < 4:
        return []
    end = usable_all[-1]
    start_hours = usable_all[0].hours
    truth = [series.logit(v) if v is not None else None for v in end.rates]
    out = []
    for lag in params["cut_lags"]:
        cut_hours = end.hours - lag
        if cut_hours - start_hours < params["cut_min_hours"]:
            continue
        sub = [r for r in refreshes if r.hours <= cut_hours]
        if len(series.usable(sub)) < 3:
            continue
        out.append((trophy_states(sub, n, pools, params), end.x, truth))
    return out


def _choose(cuts: list, n: int, params: dict) -> tuple:
    """(link choice, decay) with the least horizon-weighted absolute logit error, penalised away from the priors."""
    best = None
    for choice in params["link_choices"]:
        for lam in params["decay_grid"]:
            total = 0.0
            weight = 0.0
            for by_link, x_end, truth in cuts:
                states = by_link["logit"][0]
                for i in range(n):
                    if states[i].anchor is None or truth[i] is None:
                        continue
                    dx = max(x_end - states[i].x_last, 0.0)
                    total += dx * abs(truth[i] - median_logit(by_link, i, choice, lam, x_end))
                    weight += dx
            if weight <= 0:
                continue
            score = total / weight
            score *= 1.0 + params["decay_penalty"] * abs(lam - params["decay_prior"])
            score *= 1.0 + (params["link_penalty"] if choice != params["link_prior"] else 0.0)
            if best is None or score < best[0] - 1e-12:
                best = (score, choice, lam)
    if best is None:
        return params["link_prior"], params["decay_prior"]
    return best[1], best[2]


def calibrate(refreshes: list, n: int, pools: list, params: dict) -> Calibration:
    cal = Calibration()
    cuts = _cuts(refreshes, n, pools, params)
    cal.cuts = len(cuts)
    if not cuts:
        return cal
    cal.link, cal.decay = _choose(cuts, n, params)
    # Squared errors at the chosen settings, net of each trophy's own noise, per unit of log-time squared.
    per_group = {}
    per_trophy = {}
    for by_link, x_end, truth in cuts:
        states = by_link["logit"][0]
        for i in range(n):
            st = states[i]
            if st.anchor is None or truth[i] is None:
                continue
            dx = x_end - st.x_last
            if dx <= 1e-6:
                continue
            err = truth[i] - median_logit(by_link, i, cal.link, cal.decay, x_end)
            k2 = max(err * err - st.scale * st.scale, 0.0) / (dx * dx)
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

    by_link = trophy_states(refreshes, n, pools, params)
    cal = calibrate(refreshes, n, pools, params)
    choice = cal.link if cal.link is not None else params["link_prior"]
    lam = cal.decay if cal.decay is not None else params["decay_prior"]

    states = by_link["logit"][0]
    known = [s.anchor for s in states if s.anchor is not None]
    fallback_level = trend.median(known) if known else series.logit(1.0)
    rows = []
    for i, trophy in enumerate(trophies):
        st = states[i]
        if st.anchor is None:
            rows.append(quantiles.row(trophy["id"], fallback_level, params["unknown_sigma"], params["unknown_sigma"]))
            continue
        dx = max(x_target - st.x_last, 0.0)
        centre = median_logit(by_link, i, choice, lam, x_target)
        k = _growth(cal, i, pools[i], params)
        var = st.scale ** 2 + (k * dx) ** 2 + _count_variance(st.last_pct, st.players)
        sigma = max(math.sqrt(var), params["sigma_floor"]) * params["margin"]
        rows.append(quantiles.row(trophy["id"], centre, sigma, sigma))
    if params["platinum_cap"]:
        _cap_platinum(trophies, rows)
    return {"trophies": rows}


def _cap_platinum(trophies: list, rows: list) -> None:
    """A platinum holder holds every other trophy: no platinum quantile may exceed another trophy's at its level."""
    for i, trophy in enumerate(trophies):
        if str(trophy.get("tier") or "").lower() != "platinum":
            continue
        others = [rows[j]["q"] for j in range(len(rows)) if j != i]
        if not others:
            continue
        caps = [min(q[level] for q in others) for level in range(5)]
        rows[i]["q"] = quantiles.tidy([min(a, b) for a, b in zip(rows[i]["q"], caps)])
