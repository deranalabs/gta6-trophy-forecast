# Link functions (logit, probit) that map a percent rate to the scale a trend is extrapolated on, and back.
"""A trend is a straight line of link(rate) against log-time.

  logit   ln(p / (1 - p)): a log-logistic spread of players' times to earn a trophy.
  probit  the standard normal quantile of p: a lognormal spread of times, whose logit slope falls as a rare trophy
          rises, so a trophy that has just appeared is not carried off at its first, steep slope.

Both clamp the percent to 0.01..99.99 first, as the contest does.
"""

import math
from statistics import NormalDist

import series

_NORMAL = NormalDist()


def to_link(name: str, pct: float) -> float:
    if name == "probit":
        return _NORMAL.inv_cdf(series.clamp(pct) / 100.0)
    return series.logit(pct)


def from_link(name: str, value: float) -> float:
    """Percent for a link value."""
    if name == "probit":
        value = max(-8.0, min(8.0, value))
        return 100.0 * _NORMAL.cdf(value)
    return series.inv_logit(value)


def to_logit(name: str, value: float) -> float:
    """The contest's logit of the percent a link value stands for, without the 0.01..99.99 clamp."""
    if name == "probit":
        pct = from_link(name, value)
        pct = min(100.0 - 1e-9, max(1e-9, pct))
        p = pct / 100.0
        return math.log(p / (1.0 - p))
    return value


def logit_per_link(name: str, value: float) -> float:
    """d logit / d link at a link value: converts a fit's noise to the logit scale."""
    if name == "probit":
        p = min(1.0 - 1e-12, max(1e-12, _NORMAL.cdf(value)))
        return _NORMAL.pdf(value) / (p * (1.0 - p))
    return 1.0
