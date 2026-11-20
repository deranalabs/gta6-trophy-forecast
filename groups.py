# Sorts a case's trophies into groups that progress alike, for pooling drifts and spreads across the cross-section.
"""Trophy groups from the fields a case gives (tier, secret flag, description).

  story     - main-story missions and chapters ("Complete Chapter 2.", "Complete Breakfast Run.", the final mission):
              held by players in order of story progress, they rise as the pool plays on.
  strand    - side-character strands ("Complete the final mission for Andy.", "... for Bret."): mid-game content.
  grind     - counts and completion ("Complete 250 Stunts.", "every location", "all", "100%", money totals): very few
              players in week one, slow to move.
  platinum  - the platinum: needs every other trophy; pooled with grind for drifts.
  misc      - one-off actions and stunts (rob a shop, play an arcade game), secret story beats.

The grouping only steers pooling (a drift prior and a spread per group): a misplaced trophy still keeps its own
trend and its own backtest errors.
"""

import re

STORY = "story"
STRAND = "strand"
GRIND = "grind"
PLATINUM = "platinum"
MISC = "misc"

# Which groups share a pooled drift prior.
POOL_OF = {STORY: STORY, STRAND: STRAND, GRIND: GRIND, PLATINUM: GRIND, MISC: MISC}

_STORY_RE = re.compile(r"^Complete (?:Chapter \d+|the final mission|[A-Z][^.]*)\.?$")
_STRAND_RE = re.compile(r"\bfor [A-Z][\w'’]*\.?$")
_GRIND_WORDS = re.compile(r"\b(?:every|all|each|100%|completion|collectible)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\$?(\d[\d,]*)")


def _largest_number(text: str) -> int:
    best = 0
    for match in _NUMBER_RE.finditer(text):
        digits = match.group(1).replace(",", "")
        if digits.isdigit():
            best = max(best, int(digits))
    return best


def classify(trophy: dict) -> str:
    tier = str(trophy.get("tier") or "").lower()
    if tier == "platinum":
        return PLATINUM
    text = str(trophy.get("description") or "").strip()
    if _STRAND_RE.search(text):
        return STRAND
    if _STORY_RE.match(text):
        return STORY
    if _GRIND_WORDS.search(text) or "$" in text or _largest_number(text) >= 8:
        return GRIND
    return MISC


def classify_all(trophies: list) -> list:
    return [classify(t) for t in trophies]
