"""Poisson scoreline model: vig-stripped market odds + Elevenify blend -> score grid.

Per fixture:
  1. De-vig 1X2 -> fair P(home/draw/away); de-vig totals -> fair P(over).
  2. Invert P(total > line) under Poisson(T) to recover the total mean T.
  3. Split T into lam_home / lam_away with the Asian-handicap line; if no
     handicap is quoted, fall back to inverting 1X2 supremacy under Skellam.
  4. If an Elevenify fixture matches, blend both the total goals mean and the
     goal difference with Elevenify's predictions, re-split, and apply a
     clean-sheet Dixon-Coles tilt.
  5. Build the 0..8 Poisson grid; return the most likely scorelines.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from scipy.optimize import brentq
from scipy.stats import poisson, skellam

# Knockout stage: draws resolve to one team via ET/penalties, so "Draw" is never
# a valid pick. When set, the draw probability is redistributed into the two
# teams' "to advance" probabilities. Group stage is fully finished for WC26.
KNOCKOUT_STAGE = os.environ.get("WC_KNOCKOUT", "1") == "1"

MAX_GOALS = 8
LAMBDA_FLOOR = 0.15
ELEVENIFY_WEIGHT = 0.2  # blend: 0.8 odds / 0.2 Elevenify on both total goals and goal difference
DC_RHO = 0.1  # Dixon-Coles rho magnitude; applied as negative (inflates 0-0 and 1-1, deflates 1-0 and 0-1)
_T_MIN, _T_MAX = 1e-6, 20.0


# --------------------------------------------------------------------------- #
# De-vig + market inversion                                                    #
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
class OutcomeProbs:
    home: float          # percent, 1dp
    draw: float
    away: float
    optimal_pick: str    # "Home" | "Draw" | "Away"
    expected_points: float  # under 1pt-win / 2pt-draw scoring rule


@dataclass
class Prediction:
    lambda_home: float
    lambda_away: float
    blend_applied: bool
    knockout: bool  # draw redistributed into to-advance probs; pick is a team
    top_scorelines: list[Scoreline]
    outcomes: OutcomeProbs


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


def _dc_tau(i: int, j: int, lam_h: float, lam_a: float) -> float:
    """Dixon-Coles correction factor for the four correlated low-score cells."""
    rho = -DC_RHO
    if i == 0 and j == 0:
        return 1 - rho * lam_h * lam_a
    if i == 1 and j == 0:
        return 1 + rho * lam_a
    if i == 0 and j == 1:
        return 1 + rho * lam_h
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0


def poisson_grid(
    lam_home: float,
    lam_away: float,
    *,
    home_cs: float | None = None,
    away_cs: float | None = None,
) -> list[list[float]]:
    """P(home=i, away=j) for i,j in 0..MAX_GOALS, with Dixon-Coles low-score correction.

    When Elevenify CS data is present, applies DC tau to the four correlated
    low-score cells only: (0,0), (1,0), (0,1), (1,1). Inflates 0-0 and 1-1
    draws; deflates 1-0 and 0-1. Grid is renormalised after.
    """
    hp = [poisson.pmf(i, lam_home) for i in range(MAX_GOALS + 1)]
    ap = [poisson.pmf(j, lam_away) for j in range(MAX_GOALS + 1)]
    grid = [[hp[i] * ap[j] for j in range(MAX_GOALS + 1)] for i in range(MAX_GOALS + 1)]
    if home_cs or away_cs:
        for i in range(2):
            for j in range(2):
                grid[i][j] *= _dc_tau(i, j, lam_home, lam_away)
    total = sum(sum(row) for row in grid)
    return [[c / total for c in row] for row in grid]


def _outcome_probs(
    grid: list[list[float]], knockout: bool = False, win_points: int = 1
) -> OutcomeProbs:
    n = len(grid)
    p_home = float(sum(grid[i][j] for i in range(n) for j in range(n) if i > j))
    p_draw = float(sum(grid[i][i] for i in range(n)))
    p_away = float(sum(grid[i][j] for i in range(n) for j in range(n) if j > i))

    if knockout:
        # The tie resolves to one team; fold the draw mass into each side in
        # proportion to its regulation win prob (keeps the favorite favored).
        denom = max(p_home + p_away, 1e-9)
        home_adv = p_home + p_draw * p_home / denom
        away_adv = p_away + p_draw * p_away / denom
        pick = "Home" if home_adv >= away_adv else "Away"
        # Both sides advancing are worth the same round points, so the pick is
        # the favorite; expected points scale that prob by the round's value.
        return OutcomeProbs(
            home=round(home_adv * 100, 1),
            draw=0.0,
            away=round(away_adv * 100, 1),
            optimal_pick=pick,
            expected_points=round(win_points * max(home_adv, away_adv), 3),
        )

    choices = {"Home": p_home, "Draw": 2.0 * p_draw, "Away": p_away}
    pick = max(choices, key=choices.get)
    return OutcomeProbs(
        home=round(p_home * 100, 1),
        draw=round(p_draw * 100, 1),
        away=round(p_away * 100, 1),
        optimal_pick=pick,
        expected_points=round(choices[pick], 3),
    )


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
    knockout: bool = False,
    win_points: int = 1,
) -> Prediction:
    """Full per-fixture prediction.

    ml:         {"home", "draw", "away"} decimal odds
    totals:     {"over", "under", "line"}
    handicap:   home handicap point, or None to split via 1X2 supremacy
    elevenify:  {"home_goals", "away_goals", "home_cs", "away_cs"} or None
    knockout:   redistribute draw into to-advance probs; pick a team, not Draw
    win_points: pool points for a correct pick this round (scales knockout exp pts)
    """
    total_mean = total_mean_from_ou(totals["over"], totals["under"], totals["line"])
    p_home, _, _ = devig_1x2(ml["home"], ml["draw"], ml["away"])
    lam_home, lam_away = split_lambda(total_mean, handicap=handicap, p_home=p_home)

    blend_applied = False
    home_cs = away_cs = None
    if elevenify and elevenify.get("home_goals") is not None:
        elev_home = elevenify["home_goals"]
        elev_away = elevenify["away_goals"]
        blended_total = (1 - ELEVENIFY_WEIGHT) * total_mean + ELEVENIFY_WEIGHT * (elev_home + elev_away)
        blended_diff = (1 - ELEVENIFY_WEIGHT) * (lam_home - lam_away) + ELEVENIFY_WEIGHT * (elev_home - elev_away)
        lam_home = max((blended_total + blended_diff) / 2, LAMBDA_FLOOR)
        lam_away = max((blended_total - blended_diff) / 2, LAMBDA_FLOOR)
        home_cs, away_cs = elevenify.get("home_cs"), elevenify.get("away_cs")
        blend_applied = True

    grid = poisson_grid(lam_home, lam_away, home_cs=home_cs, away_cs=away_cs)
    return Prediction(
        lambda_home=round(lam_home, 3),
        lambda_away=round(lam_away, 3),
        blend_applied=blend_applied,
        knockout=knockout,
        top_scorelines=top_scorelines(grid),
        outcomes=_outcome_probs(grid, knockout=knockout, win_points=win_points),
    )
