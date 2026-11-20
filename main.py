# Entry point: reads every case in the input directory, in name order, and writes one validated forecast for each.
"""GTA VI trophy earn-rate forecast (make.fun gta6-trophy-rarity-forecast-01), Python 3.13 standard library only.

Run by the grader as `python3 -P -s /bot/main.py` with PYTHONHASHSEED=0. -P keeps this file's directory off sys.path,
so it is put back first, before the sibling modules are imported. Locally, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output.

Each case goes through model.forecast_case. If that raises or returns anything the contest would not accept, the case
falls back to persistence with a random-walk spread (the house baseline's formula), so every case file written is valid.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model  # noqa: E402
import quantiles  # noqa: E402
import series  # noqa: E402

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
# Random-walk spread of the fallback, logit units per square-root day.
FALLBACK_SIGMA = 0.5


def last_rates(case: dict, n: int) -> list:
    """Each trophy's latest non-null rate in the history, None where it never had one."""
    out = [None] * n
    for read in case.get("history") or []:
        for i, value in enumerate((read.get("rates") or [])[:n]):
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                out[i] = float(value)
    return out


def fallback(case: dict) -> dict:
    trophies = case.get("trophies") or []
    rates = last_rates(case, len(trophies))
    try:
        days = max(float(case.get("horizon_minutes", 0)) / 1440.0, 0.0)
    except (TypeError, ValueError):
        days = 0.0
    spread = FALLBACK_SIGMA * math.sqrt(days) if days > 0 else FALLBACK_SIGMA
    rows = []
    for trophy, rate in zip(trophies, rates):
        centre = series.logit(rate if rate is not None else 1.0)
        rows.append(quantiles.row(trophy["id"], centre, spread, spread))
    return {"trophies": rows}


def forecast(case: dict) -> dict:
    ids = [t["id"] for t in case.get("trophies") or []]
    try:
        result = model.forecast_case(case)
        if quantiles.is_valid(result, ids):
            return result
    except Exception:  # noqa: BLE001 - any failure of the model falls back to a valid forecast
        pass
    return fallback(case)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
