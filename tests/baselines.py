# Re-implements the two house baselines (persistence and the log-time curve) from RULES.txt, for local comparison.
"""House baselines as RULES.txt describes them (the curve baseline's code is not in the kit, so this is a reading of
its description: least squares of logit rate on ln(1 + hours since 2026-11-19T00:00Z) over the case's last 24 hours of
reads, read at the target, spread sqrt(random-walk^2 + residual sd^2), and the persistence row with fewer than 3 reads).
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import quantiles  # noqa: E402
import series  # noqa: E402

Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA = 0.5


def _row(trophy_id, centre: float, spread: float) -> dict:
    return {"id": trophy_id, "q": [quantiles.round4(series.inv_logit(centre + z * spread)) for z in Z]}


def _last_rate(case: dict, i: int) -> float:
    for read in reversed(case["history"]):
        value = read["rates"][i] if i < len(read["rates"]) else None
        if value is not None:
            return value
    return 0.0


def starter(case: dict) -> dict:
    spread = SIGMA * math.sqrt(case["horizon_minutes"] / 1440.0)
    last = case["history"][-1]["rates"]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last[i] if i < len(last) and last[i] is not None else 0.0
        rows.append(_row(trophy["id"], series.logit(rate), spread))
    return {"trophies": rows}


def curve(case: dict) -> dict:
    rw = SIGMA * math.sqrt(case["horizon_minutes"] / 1440.0)
    history = case["history"]
    end = series.hours_since_anchor(history[-1]["at"])
    recent = [r for r in history if series.hours_since_anchor(r["at"]) >= end - 24.0]
    x_target = math.log(1.0 + max(series.hours_since_anchor(case["target"]), 0.0))
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        xs, ys = [], []
        for read in recent:
            value = read["rates"][i] if i < len(read["rates"]) else None
            if value is not None:
                xs.append(math.log(1.0 + max(series.hours_since_anchor(read["at"]), 0.0)))
                ys.append(series.logit(value))
        if len(xs) < 3:
            rows.append(_row(trophy["id"], series.logit(_last_rate(case, i)), rw))
            continue
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 1e-12 else 0.0
        intercept = my - slope * mx
        ssr = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys))
        resid = math.sqrt(ssr / (len(xs) - 2)) if len(xs) > 2 else 0.0
        rows.append(_row(trophy["id"], intercept + slope * x_target, math.sqrt(rw * rw + resid * resid)))
    return {"trophies": rows}
