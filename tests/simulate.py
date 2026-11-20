# Simulates a synthetic Exophase recording of a launch week (players, play, refreshes, reads) for local backtests.
"""A synthetic world: NOT real data, only a test bed whose dynamics are this file's assumptions.

Players become tracked over the week (an arrival curve, optionally a late wave of casual players). Each player has a
play intensity (play-hours per real hour, higher for early arrivals), a time zone that sets a diurnal rhythm, a story
pace and a total play time after which they stop. Trophies are earned when a player's cumulative play passes a
threshold:
  story    a fixed threshold in story order, scaled by the player's pace;
  strand   gated on story progress, done by a share of players after extra play;
  misc     a one-off with a per-player chance of ever happening and an exponential wait in play-hours;
  grind    only completionists, after a long play total;
  platinum after every other trophy.
Exophase refreshes the page at random intervals; the house reads it every 30 minutes, then every 10 minutes, and
every rate is printed to 2 decimals. The trophy list (names, tiers) is taken from a kit case file.

    python3 tests/simulate.py <kit case .in> <out world .json> [scenario] [seed]

The output has a case's fields and a `history` covering the whole week (to 2026-11-26T01:10Z).
"""

import bisect
import json
import math
import random
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import groups  # noqa: E402
import series  # noqa: E402

GRID_STEP = 1.0 / 12.0
GRID_START = -6.0
GRID_END = 175.0
END_HOURS = 169.2
FAST_READS_FROM = 22.0

SCENARIOS = {
    # Steady arrivals, moderate refresh gaps.
    "base": {"players": 24000, "arrive_tau": 50.0, "late_wave": 0.0, "refresh": (0.5, 2.5), "amp": 0.5,
             "intensity": 0.22, "quit_median": 45.0},
    # A big wave of casual players late in the week: early trophies fall, trends bend.
    "late_wave": {"players": 24000, "arrive_tau": 45.0, "late_wave": 0.6, "refresh": (0.5, 2.5), "amp": 0.5,
                  "intensity": 0.22, "quit_median": 45.0},
    # Exophase refreshes rarely; strong diurnal rhythm.
    "slow_refresh": {"players": 18000, "arrive_tau": 60.0, "late_wave": 0.0, "refresh": (2.0, 7.0), "amp": 0.8,
                     "intensity": 0.18, "quit_median": 35.0},
    # Hard-core pool: fast players, quick arrivals that saturate early.
    "hardcore": {"players": 15000, "arrive_tau": 25.0, "late_wave": 0.0, "refresh": (0.3, 1.5), "amp": 0.3,
                 "intensity": 0.35, "quit_median": 70.0},
}

STORY_THRESHOLDS = {2: 0.0, 8: 2.5, 3: 4.0, 4: 9.0, 9: 12.0, 5: 15.0, 10: 18.0, 6: 22.0, 11: 26.0, 12: 31.0,
                    13: 35.0, 7: 39.0}


def _diurnal_grids(amp: float) -> list:
    """Cumulative rhythm G(t) on a grid for each of three regions; G is increasing since amp < 1."""
    times = []
    t = GRID_START
    while t <= GRID_END:
        times.append(t)
        t += GRID_STEP
    out = []
    for phase in (2.0, 10.0, 17.0):
        values = [tt + amp * 24.0 / (2.0 * math.pi) * math.sin(2.0 * math.pi * (tt - phase) / 24.0) for tt in times]
        out.append(values)
    return [times, out]


def _time_at(grid_times: list, grid_values: list, target: float) -> float:
    """The time at which G reaches target, linear between grid points; inf beyond the grid."""
    j = bisect.bisect_left(grid_values, target)
    if j >= len(grid_values):
        return math.inf
    if j == 0:
        return grid_times[0]
    v0, v1 = grid_values[j - 1], grid_values[j]
    t0, t1 = grid_times[j - 1], grid_times[j]
    return t0 + (t1 - t0) * (target - v0) / max(v1 - v0, 1e-12)


def _value_at(grid_times: list, grid_values: list, t: float) -> float:
    j = min(max(int((t - GRID_START) / GRID_STEP), 0), len(grid_times) - 2)
    frac = (t - grid_times[j]) / GRID_STEP
    return grid_values[j] + (grid_values[j + 1] - grid_values[j]) * frac


def _trophy_specs(trophies: list, rng: random.Random) -> list:
    specs = []
    for trophy in trophies:
        kind = groups.classify(trophy)
        tid = trophy["id"]
        if kind == groups.STORY:
            specs.append(("story", STORY_THRESHOLDS.get(tid, 20.0) * rng.uniform(0.85, 1.15)))
        elif kind == groups.STRAND:
            specs.append(("strand", rng.uniform(8.0, 30.0), rng.uniform(0.3, 0.8), rng.uniform(3.0, 12.0)))
        elif kind == groups.GRIND:
            specs.append(("grind", rng.uniform(25.0, 140.0), rng.uniform(0.15, 0.6)))
        elif kind == groups.PLATINUM:
            specs.append(("platinum",))
        else:
            specs.append(("misc", rng.uniform(0.0, 25.0), rng.uniform(0.2, 0.95), rng.uniform(1.0, 30.0)))
    return specs


def _arrival(rng: random.Random, sc: dict) -> float:
    if rng.random() < 0.01:
        return rng.uniform(-3.0, 0.0)
    if sc["late_wave"] and rng.random() < sc["late_wave"] * 0.5:
        return 72.0 + rng.expovariate(1.0 / 30.0)
    return rng.expovariate(1.0 / sc["arrive_tau"]) + rng.uniform(0.0, 6.0)


def simulate(trophies: list, scenario: str, seed: int) -> dict:
    sc = SCENARIOS[scenario]
    rng = random.Random(seed)
    specs = _trophy_specs(trophies, rng)
    grid_times, regions = _diurnal_grids(sc["amp"])
    n = len(trophies)
    earned = [[] for _ in range(n)]
    arrivals = []
    for _ in range(sc["players"]):
        arrive = _arrival(rng, sc)
        if arrive > END_HOURS:
            continue
        arrivals.append(arrive)
        casual = arrive > 70.0 and sc["late_wave"] > 0
        rate = sc["intensity"] * math.exp(rng.gauss(0.0, 0.55)) * math.exp(-0.004 * max(arrive, 0.0))
        if casual:
            rate *= 0.5
        pace = math.exp(rng.gauss(0.0, 0.25))
        quit_after = sc["quit_median"] * math.exp(rng.gauss(0.0, 0.9))
        completionist = rng.random() < 0.18
        values = regions[rng.randrange(3)]
        base = _value_at(grid_times, values, max(arrive, GRID_START))
        own = []
        for spec in specs:
            kind = spec[0]
            if kind == "story":
                need = spec[1] * pace
            elif kind == "strand":
                need = spec[1] * pace + rng.expovariate(1.0 / spec[3]) if rng.random() < spec[2] else math.inf
            elif kind == "grind":
                ok = completionist and rng.random() < spec[2]
                need = spec[1] * math.exp(rng.gauss(0.0, 0.35)) if ok else math.inf
            elif kind == "misc":
                need = spec[1] * pace + rng.expovariate(1.0 / spec[3]) if rng.random() < spec[2] else math.inf
            else:
                own.append(None)
                continue
            if need > quit_after:
                own.append(math.inf)
                continue
            own.append(_time_at(grid_times, values, base + need / rate) if need > 0 else arrive)
        finite = [t for t in own if t is not None]
        plat = max(finite) if finite and all(math.isfinite(t) for t in finite) else math.inf
        for i, t in enumerate(own):
            t = plat if t is None else t
            if math.isfinite(t):
                earned[i].append(max(t, arrive))
    arrivals.sort()
    for times in earned:
        times.sort()
    return _record(trophies, arrivals, earned, sc, rng)


def _page(t: float, arrivals: list, earned: list) -> tuple:
    tracked = bisect.bisect_right(arrivals, t)
    if tracked == 0:
        return 0, 0, [0.0] * len(earned)
    counts = [bisect.bisect_right(times, t) for times in earned]
    rates = [round(100.0 * c / tracked, 2) for c in counts]
    return tracked, sum(counts), rates


def _record(trophies: list, arrivals: list, earned: list, sc: dict, rng: random.Random) -> dict:
    refreshes = [-1.0]
    t = 4.5
    while t < END_HOURS + 1:
        refreshes.append(t)
        t += rng.uniform(*sc["refresh"])
    pages = [_page(r, arrivals, earned) for r in refreshes]
    history = []
    read = -10.0 / 60.0
    while read <= END_HOURS:
        j = bisect.bisect_right(refreshes, read) - 1
        tracked, total, rates = pages[max(j, 0)]
        moment = series.ANCHOR + timedelta(hours=read)
        slot = moment.replace(second=0, microsecond=0)
        history.append({
            "slot": slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "at": moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z",
            "players_tracked": tracked,
            "trophies_earned": total,
            "rates": rates,
        })
        read += 0.5 if read < FAST_READS_FROM else 1.0 / 6.0
        read = round(read * 6.0) / 6.0 if read >= FAST_READS_FROM else read
    return {"history": history}


def main() -> None:
    kit_case = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    scenario = sys.argv[3] if len(sys.argv) > 3 else "base"
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    world = simulate(kit_case["trophies"], scenario, seed)
    out = {key: kit_case[key] for key in ("schema", "levels", "game", "trophies")}
    out["id"] = f"sim-{scenario}-{seed}"
    out["notes"] = f"Synthetic world from tests/simulate.py, scenario {scenario}, seed {seed}: not real data."
    out["history"] = world["history"]
    Path(sys.argv[2]).write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
