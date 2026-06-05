"""Poisson scoreline model: vig-stripped market odds + Elevenify blend -> score grid.

Per fixture:
  1. De-vig 1X2 -> fair P(home/draw/away); de-vig totals -> fair P(over).
  2. Invert P(total > line) under Poisson(T) to recover the total mean T.
  3. Split T into lam_home / lam_away with the Asian-handicap line; if no
     handicap is quoted, fall back to inverting 1X2 supremacy under Skellam.
  4. If an Elevenify fixture matches, blend the odds goal-difference with
     Elevenify's predicted goal difference, re-split T, and apply a clean-sheet
     Dixon-Coles tilt.
  5. Build the 0..8 Poisson grid; return the most likely scorelines.

The de-vig / market-inversion core (steps 1-3) is lifted from a prior odds
project and is provider-agnostic: it only needs decimal odds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq
from scipy.stats import poisson, skellam

MAX_GOALS = 8
LAMBDA_FLOOR = 0.15
ELEVENIFY_WEIGHT = 0.4  # blend: 0.6 odds / 0.4 Elevenify on the goal difference
CS_TILT = 0.3  # clean-sheet Dixon-Coles strength
_T_MIN, _T_MAX = 1e-6, 20.0


# --------------------------------------------------------------------------- #
# De-vig + market inversion (salvaged, provider-agnostic)                      #
# --------------------------------------------------------------------------- #
def implied_prob(decimal_odds: float) -> float:
    """Raw (vig-inclusive) implied probability from decimal odds."""
    return 1.0 / decimal_odds


def devig_two_way(over_odds: float, under_odds: float) -> float:
    """Fair (vig-free) P(over) from an over/under price pair."""
    po, pu = implied_prob(over_odds), implied_prob(under_odds)
    return po / (po + pu)


def devig_1x2(home: float, draw: float, away: float) -> tuple[float, float, float]:
    """Fair (vig-free) (P(home), P(draw), P(away)) from a 1X2 price triple."""
    ph, pd, pa = implied_prob(home), implied_prob(draw), implied_prob(away)
    t = ph + pd + pa
    return ph / t, pd / t, pa / t


def total_mean_from_ou(over_odds: float, under_odds: float, line: float) -> float:
    """Poisson total mean T such that P(total > line) == de-vigged P(over).

    Assumes a half-line (x.5): 'over' == total >= floor(line) + 1, so
    P(over) = 1 - poisson.cdf(floor(line), T). brentq is exact; the spec's
    binary search converges to the same root.
    """
    fair_over = devig_two_way(over_odds, under_odds)
    k = math.floor(line)
    return brentq(lambda t: (1.0 - poisson.cdf(k, t)) - fair_over, _T_MIN, _T_MAX)


def supremacy_from_1x2(p_home: float, total_mean: float) -> float:
    """Skellam supremacy S such that P(diff > 0) == p_home, given total mean T.

    diff = home - away ~ Skellam(lam_h, lam_a), lam_h=(T+S)/2, lam_a=(T-S)/2.
    P(home win) is monotonic in S, so brentq finds a unique root. If p_home is
    beyond the achievable range for this T, clamp to the nearest feasible S.
    """

    def p_home_win(s: float) -> float:
        lam_h = max((total_mean + s) / 2, _T_MIN)
        lam_a = max((total_mean - s) / 2, _T_MIN)
        return 1.0 - skellam.cdf(0, lam_h, lam_a)

    lo, hi = -total_mean + _T_MIN, total_mean - _T_MIN
    f_lo, f_hi = p_home_win(lo) - p_home, p_home_win(hi) - p_home
    if f_lo * f_hi > 0:  # no sign change -> p_home outside achievable range
        return lo if abs(f_lo) < abs(f_hi) else hi
    return brentq(lambda s: p_home_win(s) - p_home, lo, hi)


# --------------------------------------------------------------------------- #
# Lambda split, grid, prediction                                              #
# --------------------------------------------------------------------------- #
@dataclass
class Scoreline:
    score: str
    probability: float  # percent, 2dp
    outcome: str  # "Home Win" / "Draw" / "Away Win"


@dataclass
class Prediction:
    lambda_home: float
    lambda_away: float
    blend_applied: bool
    top_scorelines: list[Scoreline]


def split_lambda(
    total_mean: float,
    *,
    handicap: float | None = None,
    p_home: float | None = None,
) -> tuple[float, float]:
    """Split T into (lam_home, lam_away). Prefer the Asian handicap; else 1X2.

    handicap is the *home* handicap point: -0.5 means home favoured, so
    lam_home = (T - handicap) / 2 lifts the favourite. Both clamped to a floor.
    """
    if handicap is not None:
        lam_home = (total_mean - handicap) / 2
        lam_away = (total_mean + handicap) / 2
    elif p_home is not None:
        s = supremacy_from_1x2(p_home, total_mean)
        lam_home, lam_away = (total_mean + s) / 2, (total_mean - s) / 2
    else:
        lam_home = lam_away = total_mean / 2
    return max(lam_home, LAMBDA_FLOOR), max(lam_away, LAMBDA_FLOOR)


def _outcome(i: int, j: int) -> str:
    return "Home Win" if i > j else "Away Win" if j > i else "Draw"


def poisson_grid(
    lam_home: float,
    lam_away: float,
    *,
    home_cs: float | None = None,
    away_cs: float | None = None,
) -> list[list[float]]:
    """P(home=i, away=j) for i,j in 0..MAX_GOALS, with optional clean-sheet tilt.

    A strong home clean-sheet rate lifts the away=0 column; a strong away CS
    rate lifts the home=0 row (Dixon-Coles-style). Grid is renormalised after.
    """
    hp = [poisson.pmf(i, lam_home) for i in range(MAX_GOALS + 1)]
    ap = [poisson.pmf(j, lam_away) for j in range(MAX_GOALS + 1)]
    grid = [[hp[i] * ap[j] for j in range(MAX_GOALS + 1)] for i in range(MAX_GOALS + 1)]
    if home_cs:  # P(away = 0): every cell with j == 0
        for i in range(MAX_GOALS + 1):
            grid[i][0] *= 1.0 + home_cs * CS_TILT
    if away_cs:  # P(home = 0): every cell with i == 0
        for j in range(MAX_GOALS + 1):
            grid[0][j] *= 1.0 + away_cs * CS_TILT
    total = sum(sum(row) for row in grid)
    return [[c / total for c in row] for row in grid]


def top_scorelines(grid: list[list[float]], n: int = 10) -> list[Scoreline]:
    cells = [
        (i, j, grid[i][j]) for i in range(len(grid)) for j in range(len(grid))
    ]
    cells.sort(key=lambda c: c[2], reverse=True)
    return [
        Scoreline(f"{i}-{j}", round(p * 100, 2), _outcome(i, j))
        for i, j, p in cells[:n]
    ]


def predict(
    *,
    ml: dict,
    totals: dict,
    handicap: float | None = None,
    elevenify: dict | None = None,
) -> Prediction:
    """Full per-fixture prediction.

    ml:        {"home", "draw", "away"} decimal odds
    totals:    {"over", "under", "line"}
    handicap:  home handicap point, or None to split via 1X2 supremacy
    elevenify: {"home_goals", "away_goals", "home_cs", "away_cs"} or None
    """
    total_mean = total_mean_from_ou(totals["over"], totals["under"], totals["line"])
    p_home, _, _ = devig_1x2(ml["home"], ml["draw"], ml["away"])
    lam_home, lam_away = split_lambda(total_mean, handicap=handicap, p_home=p_home)

    blend_applied = False
    home_cs = away_cs = None
    if elevenify and elevenify.get("home_goals") is not None:
        odds_diff = lam_home - lam_away
        elev_diff = elevenify["home_goals"] - elevenify["away_goals"]
        blended = (1 - ELEVENIFY_WEIGHT) * odds_diff + ELEVENIFY_WEIGHT * elev_diff
        lam_home = max((total_mean + blended) / 2, LAMBDA_FLOOR)
        lam_away = max((total_mean - blended) / 2, LAMBDA_FLOOR)
        home_cs, away_cs = elevenify.get("home_cs"), elevenify.get("away_cs")
        blend_applied = True

    grid = poisson_grid(lam_home, lam_away, home_cs=home_cs, away_cs=away_cs)
    return Prediction(
        lambda_home=round(lam_home, 3),
        lambda_away=round(lam_away, 3),
        blend_applied=blend_applied,
        top_scorelines=top_scorelines(grid),
    )
