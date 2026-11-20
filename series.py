# Turns a case's history into distinct Exophase refreshes and per-trophy series on a log-time axis.
"""Parsing and de-duplication of the house's reads.

Exophase refreshes a game's counts on its own schedule, so many consecutive reads show the same page. A read is a new
observation only when its players-tracked count, trophies-earned count or any rate differs from the read before it.
The moment of a refresh is placed halfway between the last read of the old page and the first read of the new one.

Time is measured in hours since ANCHOR (2026-11-19T00:00Z, the release date, the origin the house curve baseline uses)
and mapped to x = ln(1 + max(hours, 0)), the axis every trend in this entry is fitted on.
"""

import math
from datetime import datetime, timezone

# Percent bounds the contest clamps to before taking a logit.
LO = 0.01
HI = 99.99
# The origin of the log-time axis.
ANCHOR = datetime(2026, 11, 19, tzinfo=timezone.utc)
# A refresh whose players-tracked count is below this share of the latest one is the pre-launch page: left out of fits.
MIN_PLAYER_SHARE = 0.25


def clamp(pct: float) -> float:
    return min(HI, max(LO, pct))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(value: float) -> float:
    """Percent for a logit; safe for any finite value."""
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def parse_time(text: str) -> datetime:
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def hours_since_anchor(text: str) -> float:
    return (parse_time(text) - ANCHOR).total_seconds() / 3600.0


def log_time(hours: float) -> float:
    return math.log(1.0 + max(hours, 0.0))


class Refresh:
    """One distinct page Exophase served: when it appeared and what it said."""

    __slots__ = ("hours", "x", "players", "rates")

    def __init__(self, hours: float, players: float, rates: list):
        self.hours = hours
        self.x = log_time(hours)
        self.players = players
        self.rates = rates


def _signature(read: dict) -> tuple:
    return (read.get("players_tracked"), read.get("trophies_earned"), tuple(read.get("rates") or ()))


def build_refreshes(history: list, n_trophies: int) -> list:
    """Every distinct page in the history, oldest first, rates padded or cut to n_trophies."""
    refreshes = []
    previous_sig = None
    previous_hours = None
    for read in history:
        try:
            hours = hours_since_anchor(read["at"])
        except (KeyError, TypeError, ValueError):
            continue
        sig = _signature(read)
        if sig == previous_sig:
            previous_hours = hours
            continue
        seen_at = hours if previous_hours is None else 0.5 * (previous_hours + hours)
        rates = list(read.get("rates") or [])
        rates = (rates + [None] * n_trophies)[:n_trophies]
        clean = []
        for value in rates:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            clean.append(float(value) if ok else None)
        players = read.get("players_tracked")
        players = float(players) if isinstance(players, (int, float)) and players and players > 0 else 0.0
        refreshes.append(Refresh(seen_at, players, clean))
        previous_sig = sig
        previous_hours = hours
    return refreshes


def usable(refreshes: list) -> list:
    """The refreshes a fit may use: the pre-launch page with a tiny player pool is dropped when later pages exist."""
    if not refreshes:
        return []
    latest = refreshes[-1].players
    if latest <= 0:
        return list(refreshes)
    kept = [r for r in refreshes if r.players >= MIN_PLAYER_SHARE * latest]
    return kept if kept else list(refreshes)


def trophy_series(refreshes: list, index: int) -> list:
    """(hours, x, logit rate, raw percent, players) for every refresh that lists the trophy."""
    out = []
    for r in refreshes:
        value = r.rates[index]
        if value is not None:
            out.append((r.hours, r.x, logit(value), value, r.players))
    return out


def typical_gap_hours(refreshes: list, window_hours: float = 24.0) -> float:
    """Median hours between refreshes over the last window; 0 when there are fewer than three refreshes."""
    if len(refreshes) < 3:
        return 0.0
    end = refreshes[-1].hours
    recent = [r.hours for r in refreshes if r.hours >= end - window_hours]
    if len(recent) < 3:
        recent = [r.hours for r in refreshes[-3:]]
    gaps = sorted(b - a for a, b in zip(recent, recent[1:]))
    return gaps[len(gaps) // 2]
