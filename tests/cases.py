# Builds synthetic case files of any size and shape (smooth trends, refresh holds, nulls) for the unit tests.
"""Synthetic cases for tests: 55 trophies like the real list's tiers (1 platinum, 2 gold, 6 silver, 46 bronze), a
history of reads on a fixed cadence, rates that follow smooth logit trends, and pages that Exophase holds for a few
reads at a time. Not meant to be realistic, only well-formed (or deliberately ragged, with nulls)."""

import math
import random
from datetime import timedelta

import series

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def trophy_list(n: int = 55) -> list:
    tiers = ["platinum"] + ["gold"] * 2 + ["silver"] * 6 + ["bronze"] * (n - 9)
    points = {"platinum": 300, "gold": 90, "silver": 30, "bronze": 15}
    descriptions = ["Complete Chapter 1.", "Complete the final mission for Andy.", "Complete 250 Stunts.",
                    "Rob a shop.", "Explore all of Leonida."]
    out = []
    for i in range(n):
        tier = tiers[i] if i < len(tiers) else "bronze"
        out.append({"id": i + 1, "name": f"Trophy {i + 1}", "description": descriptions[i % len(descriptions)],
                    "tier": tier, "points": points[tier], "secret": i % 3 == 0})
    return out


def _iso(hours: float) -> str:
    moment = series.ANCHOR + timedelta(hours=hours)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def build(hours: float = 36.0, step_minutes: float = 10.0, start: float = 23.5, hold: int = 6, seed: int = 7,
          null_share: float = 0.0, n: int = 55, horizon_minutes: int = 6510) -> dict:
    """A case whose history runs `hours` up to its checkpoint at `start + hours` hours after the anchor."""
    rng = random.Random(seed)
    trophies = trophy_list(n)
    levels = [rng.uniform(-8.0, 3.0) for _ in range(n)]
    slopes = [rng.uniform(-0.3, 2.0) for _ in range(n)]
    levels[1] = 9.5
    slopes[1] = 0.0
    steps = int(hours * 60 / step_minutes)
    history = []
    page = None
    for k in range(steps):
        h = start + k * step_minutes / 60.0
        if page is None or k % hold == 0:
            x = math.log(1.0 + h)
            rates = []
            for i in range(n):
                value = round(series.inv_logit(levels[i] + slopes[i] * (x - 3.5) + rng.gauss(0, 0.03)), 2)
                rates.append(value)
            page = {"players_tracked": 20000 + 300 * k, "trophies_earned": 100000 + 2000 * k, "rates": rates}
        rates = list(page["rates"])
        if null_share:
            rates = [None if rng.random() < null_share else v for v in rates]
        moment = _iso(h)
        history.append({"slot": moment[:16] + ":00Z", "at": moment, "players_tracked": page["players_tracked"],
                        "trophies_earned": page["trophies_earned"], "rates": rates})
    at_hours = start + hours
    return {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "t00",
        "at": _iso(at_hours)[:19] + "Z",
        "horizon_minutes": horizon_minutes,
        "target": _iso(at_hours + horizon_minutes / 60.0)[:19] + "Z",
        "levels": LEVELS,
        "game": {"name": "Synthetic", "platform": "PS5", "release_date": "2026-11-19", "source": "test"},
        "trophies": trophies,
        "history": history,
        "notes": "synthetic test case",
    }


def every_read_distinct(hours: float = 60.0) -> dict:
    """The slowest shape: a new page on every read, 10-minute cadence."""
    return build(hours=hours, step_minutes=10.0, start=0.0, hold=1, seed=11)
