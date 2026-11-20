# Turns a logit median and two side spreads into five valid percent quantiles, and validates whole forecasts.
"""Quantile rows.

A row is a two-piece normal on the logit scale: below the median the spread is sigma_down, above it sigma_up. The
quantiles are mapped back to percent, rounded to 4 decimals half up (as the starter rounds), held to 0..100, and made
non-decreasing after rounding.
"""

import math

import series

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536, -0.6744898, 0.0, 0.6744898, 1.6448536]


def round4(value: float) -> float | int:
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if out.is_integer() else out


def _safe(value: float, default: float) -> float:
    return value if isinstance(value, (int, float)) and math.isfinite(value) else default


def tidy(values: list) -> list:
    """Five finite, rounded, non-decreasing percents in 0..100."""
    out = []
    floor = 0.0
    for value in values:
        v = min(100.0, max(0.0, _safe(value, floor)))
        v = round4(v)
        if v < floor:
            v = floor
        out.append(v)
        floor = v
    return out


def row(trophy_id, centre: float, sigma_down: float, sigma_up: float) -> dict:
    centre = _safe(centre, 0.0)
    sigma_down = max(_safe(sigma_down, 1.0), 0.0)
    sigma_up = max(_safe(sigma_up, 1.0), 0.0)
    values = []
    for z in Z:
        spread = sigma_down if z < 0 else sigma_up
        values.append(series.inv_logit(centre + z * spread))
    return {"id": trophy_id, "q": tidy(values)}


def is_valid(forecast: dict, trophy_ids: list) -> bool:
    """The contest's shape: one row per id, exactly id and q, five finite non-decreasing values in 0..100."""
    if not isinstance(forecast, dict) or list(forecast.keys()) != ["trophies"]:
        return False
    rows = forecast["trophies"]
    if not isinstance(rows, list) or len(rows) != len(trophy_ids):
        return False
    seen = set()
    for item in rows:
        if not isinstance(item, dict) or sorted(item.keys()) != ["id", "q"]:
            return False
        q = item["q"]
        if not isinstance(q, list) or len(q) != 5:
            return False
        for a, b in zip(q, q[1:]):
            if b < a:
                return False
        for v in q:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 100:
                return False
        seen.add(item["id"])
    return seen == set(trophy_ids)
