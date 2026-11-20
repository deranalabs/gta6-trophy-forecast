# Robust straight-line fits of logit rate on log-time, and empirical-Bayes shrinkage of slopes toward a pooled drift.
"""Fitting helpers, pure arithmetic on lists.

robust_line fits y = a + b * x by iteratively reweighted least squares with Huber weights (tuning 1.345 times a MAD
scale), so a single odd refresh does not swing the slope. Its slope standard error is inflated for serial correlation:
refreshes a few hours apart share diurnal swings, so the fit counts one independent point per EFFECTIVE_HOURS hours.

shrink_slopes pulls every trophy's slope toward its pool's median slope with weights 1/se^2 and 1/tau^2, where tau^2
is the pool's between-trophy slope variance net of the trophies' own sampling variance.
"""

import math

HUBER_C = 1.345
# Hours of history that count as one independent observation for a slope's standard error.
EFFECTIVE_HOURS = 3.0
# Smallest residual scale a fit reports (logit units): the page's 2-decimal rounding alone is about this at 1%.
MIN_SCALE = 0.01


def median(values: list) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def quantile(values: list, tau: float) -> float:
    """Linear-interpolated sample quantile."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def mad_scale(values: list) -> float:
    if not values:
        return 0.0
    centre = median(values)
    return 1.4826 * median([abs(v - centre) for v in values])


class Fit:
    __slots__ = ("slope", "intercept", "scale", "slope_se", "n", "span_x")

    def __init__(self, slope, intercept, scale, slope_se, n, span_x):
        self.slope = slope
        self.intercept = intercept
        self.scale = scale
        self.slope_se = slope_se
        self.n = n
        self.span_x = span_x

    def at(self, x: float) -> float:
        return self.intercept + self.slope * x


def _weighted_line(xs, ys, ws):
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    return slope, my - slope * mx, sxx, sw


def robust_line(xs: list, ys: list, span_hours: float, iterations: int = 5):
    """Huber-weighted line; None when the points cannot fix a slope."""
    n = len(xs)
    if n < 3 or max(xs) - min(xs) < 1e-6:
        return None
    ws = [1.0] * n
    slope, intercept, sxx, sw = _weighted_line(xs, ys, ws)
    scale = MIN_SCALE
    for _ in range(iterations):
        resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        scale = max(mad_scale(resid), MIN_SCALE)
        limit = HUBER_C * scale
        ws = [1.0 if abs(r) <= limit else limit / abs(r) for r in resid]
        slope, intercept, sxx, sw = _weighted_line(xs, ys, ws)
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    rms = math.sqrt(sum(w * r * r for w, r in zip(ws, resid)) / max(sw - 2.0, 1.0))
    scale = max(scale, rms, MIN_SCALE)
    # Unweighted sxx over the effective number of points: correlated refreshes carry less information.
    n_eff = max(2.0, min(float(n), span_hours / EFFECTIVE_HOURS))
    mean_x = sum(xs) / n
    raw_sxx = sum((x - mean_x) ** 2 for x in xs) * (n_eff / n)
    slope_se = scale / math.sqrt(max(raw_sxx, 1e-12))
    return Fit(slope, intercept, scale, slope_se, n, max(xs) - min(xs))


def shrink_slopes(slopes: list, ses: list, pools: list) -> tuple:
    """Shrunk slopes, and each pool's (median, tau) prior; a None slope takes its pool's median outright."""
    by_pool = {}
    for b, se, pool in zip(slopes, ses, pools):
        if b is not None:
            by_pool.setdefault(pool, []).append((b, se))
    every = [b for b in slopes if b is not None]
    overall = median(every) if every else 0.0
    priors = {}
    for pool in set(pools):
        members = by_pool.get(pool, [])
        if len(members) >= 3:
            centre = median([b for b, _ in members])
            spread = mad_scale([b for b, _ in members]) ** 2
            noise = median([se * se for _, se in members])
            tau2 = max(spread - noise, 0.25 * spread, 1e-4)
        elif members:
            # Too few to stand alone: half-way to the whole cross-section's median.
            centre = 0.5 * (median([b for b, _ in members]) + overall)
            tau2 = max(mad_scale(every) ** 2, 1e-2) if len(every) >= 3 else 0.25
        else:
            centre = overall
            tau2 = max(mad_scale(every) ** 2, 1e-2) if len(every) >= 3 else 0.25
        priors[pool] = (centre, math.sqrt(tau2))
    shrunk = []
    for b, se, pool in zip(slopes, ses, pools):
        centre, tau = priors[pool]
        if b is None:
            shrunk.append(centre)
            continue
        w_own = 1.0 / max(se * se, 1e-8)
        w_pool = 1.0 / (tau * tau)
        shrunk.append((w_own * b + w_pool * centre) / (w_own + w_pool))
    return shrunk, priors
