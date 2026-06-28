"""Round detection + pool scoring rules.

Pool points for a correct pick, by round:
  Group Stage:   Win 1, Draw 2
  Round of 32:   Win 2
  Round of 16:   Win 2
  Quarter-Final: Win 3
  Semi-Final:    Win 5
  Third Place:   Win 5   (not in the official list; mirrors the Semi-Final tier)
  Final:         Win 10

Knockout ties decided on penalties read as a level score from the feed, so the
advancing side is unknown. Such matches stay unscored (None) until a manual
"winner" ("home"/"away") is added to the fixture's .results_store.json entry.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

# A match belongs to the latest round whose start date is <= its tournament-local
# date. We shift kickoff by -8h so a late US-evening game (which lands in the early
# hours of the next UTC day) maps back to its US calendar day before comparison.
_DAY_SHIFT = timedelta(hours=8)
_ROUND_STARTS: list[tuple[str, date]] = [
    ("group", date(1900, 1, 1)),
    ("r32", date(2026, 6, 28)),
    ("r16", date(2026, 7, 4)),
    ("qf", date(2026, 7, 9)),
    ("sf", date(2026, 7, 14)),
    ("third", date(2026, 7, 18)),
    ("final", date(2026, 7, 19)),
]

WIN_POINTS = {"group": 1, "r32": 2, "r16": 2, "qf": 3, "sf": 5, "third": 5, "final": 10}
DRAW_POINTS = 2  # group stage only
_LABELS = {
    "group": "Group Stage",
    "r32": "Round of 32",
    "r16": "Round of 16",
    "qf": "Quarter-Final",
    "sf": "Semi-Final",
    "third": "Third Place",
    "final": "Final",
}


def round_of(commence_time: str | None) -> str:
    """Map an ISO kickoff time to its tournament round key."""
    if not commence_time:
        return "group"
    try:
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    except ValueError:
        return "group"
    d = (dt.astimezone(timezone.utc) - _DAY_SHIFT).date()
    rnd = "group"
    for name, start in _ROUND_STARTS:  # ascending; keep the latest match
        if d >= start:
            rnd = name
    return rnd


def round_label(rnd: str) -> str:
    return _LABELS.get(rnd, rnd)


def win_points(rnd: str) -> int:
    return WIN_POINTS.get(rnd, 1)


def points_earned(
    rnd: str, pick: str | None, actual: str | None, winner: str | None = None
) -> int | None:
    """Pool points for a pick.

    Returns None when not yet scorable: no result, or a knockout tie (penalties)
    with no manual ``winner`` recorded. ``pick`` / ``actual`` are
    "home" | "away" | "draw"; ``winner`` is "home" | "away".
    """
    if pick is None or actual is None:
        return None
    if rnd == "group":
        if pick == actual:
            return DRAW_POINTS if pick == "draw" else WIN_POINTS["group"]
        return 0
    # Knockout: one side advances. A level score means penalties decided it, so
    # fall back to the manually recorded winner; until then it is unscorable.
    effective = actual
    if actual == "draw":
        if winner not in ("home", "away"):
            return None
        effective = winner
    return win_points(rnd) if pick == effective else 0
