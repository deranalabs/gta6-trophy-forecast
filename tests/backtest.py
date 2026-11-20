# Cuts a recorded history into synthetic cases and scores the model against both house baselines by the contest loss.
"""Local backtest.

    python3 tests/backtest.py [--scheme inside|final|all] [--set key=value ...] FILE [FILE ...]

FILE is any JSON with `trophies` and `history` (a kit case, or a world from tests/simulate.py). Each synthetic case
keeps the reads whose slot is before its checkpoint; its answer is the first read in the hour from its target, as the
contest takes it. Trophies missing from the answer are not scored.

Schemes:
  inside  checkpoints every 6 hours from 12 hours after launch, horizons 6, 12 and 24 hours, every target inside
          the history: what the real final case's own history can verify.
  final   the final's shape: checkpoint 2026-11-21T11:30Z, target 2026-11-26T00:00Z (needs a simulated week).
  all     both.

--set overrides a model.PARAMS entry for the run (numbers, or comma lists for tuples), to compare ideas.
"""

import json
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baselines  # noqa: E402
import model  # noqa: E402
import series  # noqa: E402

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
FINAL_CUT = 59.5
FINAL_TARGET = 168.0


def _iso(hours: float) -> str:
    return (series.ANCHOR + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_case(world: dict, cut_hours: float, target_hours: float):
    """(case, answer rates) or None when the history has no answering read."""
    history = [r for r in world["history"] if series.hours_since_anchor(r["slot"]) < cut_hours]
    answer = None
    for read in world["history"]:
        slot = series.hours_since_anchor(read["slot"])
        if target_hours <= slot < target_hours + 1.0:
            answer = read["rates"]
            break
    if not history or answer is None:
        return None
    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": f"cut-{cut_hours:g}-{target_hours:g}",
        "at": _iso(cut_hours),
        "horizon_minutes": int(round((target_hours - cut_hours) * 60)),
        "target": _iso(target_hours),
        "levels": LEVELS,
        "game": world.get("game", {}),
        "trophies": world["trophies"],
        "history": history,
        "notes": "backtest cut",
    }
    return case, answer


def pinball(y: float, q: float, tau: float) -> float:
    d = series.logit(y) - series.logit(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score(forecast: dict, answer: list, trophies: list) -> tuple:
    """(mean loss, items, covered, scored trophies)."""
    by_id = {row["id"]: row["q"] for row in forecast["trophies"]}
    total = 0.0
    items = 0
    covered = 0
    scored = 0
    for i, trophy in enumerate(trophies):
        y = answer[i] if i < len(answer) else None
        if y is None:
            continue
        q = by_id[trophy["id"]]
        for tau, value in zip(LEVELS, q):
            total += pinball(y, value, tau)
            items += 1
        scored += 1
        covered += 1 if q[0] <= y <= q[4] else 0
    return total, items, covered, scored


def cuts_for(world: dict, scheme: str) -> list:
    history = world["history"]
    first = series.hours_since_anchor(history[0]["slot"])
    last = series.hours_since_anchor(history[-1]["slot"])
    pairs = []
    if scheme in ("inside", "all"):
        end = min(last, FINAL_CUT) if last > FINAL_TARGET else last
        cut = max(first, 0.0) + 12.0
        while cut < end:
            for horizon in (6.0, 12.0, 24.0):
                if cut + horizon <= end:
                    pairs.append((cut, cut + horizon))
            cut += 6.0
    if scheme in ("final", "all") and last >= FINAL_TARGET:
        pairs.append((FINAL_CUT, FINAL_TARGET))
    return pairs


def parse_sets(items: list) -> dict:
    params = dict(model.PARAMS)
    for item in items:
        key, _, raw = item.partition("=")
        if key not in params:
            raise SystemExit(f"unknown parameter {key}")
        old = params[key]
        if isinstance(old, tuple) and old and isinstance(old[0], str):
            params[key] = tuple(raw.split(","))
        elif isinstance(old, tuple):
            params[key] = tuple(float(v) for v in raw.split(","))
        elif isinstance(old, bool):
            params[key] = raw.lower() in ("1", "true", "yes")
        elif isinstance(old, str):
            params[key] = raw
        else:
            params[key] = float(raw)
    return params


def run(files: list, scheme: str, params: dict, verbose: bool = True) -> dict:
    forecasters = {
        "starter": baselines.starter,
        "curve": baselines.curve,
        "model": lambda case: model.forecast_case(case, params),
    }
    totals = {name: [0.0, 0, 0, 0] for name in forecasters}
    slowest = 0.0
    made_cases = 0
    for path in files:
        world = json.loads(Path(path).read_text(encoding="utf-8"))
        for cut, target in cuts_for(world, scheme):
            made = make_case(world, cut, target)
            if made is None:
                continue
            case, answer = made
            made_cases += 1
            line = [f"{Path(path).stem:>22} cut {cut:6.1f}h -> {target:6.1f}h"]
            for name, fn in forecasters.items():
                started = time.perf_counter()
                result = fn(case)
                if name == "model":
                    slowest = max(slowest, time.perf_counter() - started)
                loss, items, covered, scored = score(result, answer, world["trophies"])
                acc = totals[name]
                acc[0] += loss
                acc[1] += items
                acc[2] += covered
                acc[3] += scored
                line.append(f"{name} {loss / max(items, 1):.4f}")
            if verbose:
                print("  ".join(line))
    if not made_cases:
        print(f"scheme {scheme}: no cut of these histories has an answering read inside them; nothing scored")
        return {}
    summary = {}
    for name, (loss, items, covered, scored) in totals.items():
        summary[name] = (loss / max(items, 1), covered / max(scored, 1))
    print(f"scheme {scheme}: " + "  ".join(f"{k} loss {v[0]:.4f} cov90 {v[1]:.2f}" for k, v in summary.items())
          + f"  (slowest model run {slowest:.2f}s)")
    return summary


def main() -> None:
    args = sys.argv[1:]
    scheme = "all"
    sets = []
    quiet = False
    files = []
    while args:
        arg = args.pop(0)
        if arg == "--scheme":
            scheme = args.pop(0)
        elif arg == "--set":
            sets.append(args.pop(0))
        elif arg == "--quiet":
            quiet = True
        else:
            files.append(arg)
    if not files:
        raise SystemExit(__doc__)
    run(files, scheme, parse_sets(sets), verbose=not quiet)


if __name__ == "__main__":
    main()
