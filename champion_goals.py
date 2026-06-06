#!/usr/bin/env python3
"""
Estimate total goals the WC26 champion will score over 7 games.

Pulls outright winner odds from The Odds API, de-vigs them into win
probabilities, then weights each team's expected goal output (based on
historical WC winner distributions by playing style) to produce a
mixture distribution and a recommended "closest without going over" bid.

Usage:
    python champion_goals.py
"""
from __future__ import annotations

import math
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Historical WC winners — 32-team era (1986-2022), all 7 games each.
# WC26 champion also plays 7 games (2 group + R32/R16/QF/SF/F), so this
# transfers directly. Verify goal totals before citing externally.
# ---------------------------------------------------------------------------
HISTORICAL = [
    # (year, team, goals, style)
    (1986, "Argentina", 14, "balanced"),
    (1990, "Germany",   15, "attacking"),
    (1994, "Brazil",    11, "balanced"),
    (1998, "France",    15, "balanced"),
    (2002, "Brazil",    18, "attacking"),
    (2006, "Italy",     12, "defensive"),
    (2010, "Spain",      8, "defensive"),
    (2014, "Germany",   18, "attacking"),
    (2018, "France",    12, "defensive"),
    (2022, "Argentina", 15, "balanced"),
]

# ---------------------------------------------------------------------------
# Style classifications for WC26 contenders. Edit freely.
# attacking  — high press, high goal output expected
# balanced   — mixed approach
# defensive  — low-block or pragmatic; win ugly
# ---------------------------------------------------------------------------
TEAM_STYLE: dict[str, str] = {
    "France":             "defensive",   # Deschamps always pragmatic
    "Brazil":             "attacking",
    "England":            "balanced",
    "Spain":              "balanced",    # more vertical post-Xavi than 2010
    "Argentina":          "balanced",
    "Germany":            "attacking",
    "Portugal":           "attacking",
    "Netherlands":        "balanced",
    "Morocco":            "defensive",
    "Uruguay":            "defensive",
    "United States":      "balanced",
    "USA":                "balanced",
    "Mexico":             "defensive",
    "Colombia":           "balanced",
    "Belgium":            "balanced",
    "Croatia":            "defensive",
    "Serbia":             "balanced",
    "Senegal":            "defensive",
}

OUTRIGHTS_URL = (
    "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup_winner/odds"
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_win_probs() -> dict[str, float]:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sys.exit("ODDS_API_KEY not set. Export it and re-run.")

    r = requests.get(
        OUTRIGHTS_URL,
        params={"apiKey": key, "regions": "eu", "oddsFormat": "decimal"},
        timeout=20,
    )
    r.raise_for_status()
    print(f"API requests remaining: {r.headers.get('x-requests-remaining', '?')}\n")

    events = r.json()
    if not events:
        sys.exit("No outright events returned — market may not be listed yet.")

    # Collect per-team implied probabilities across all bookmakers, then average.
    raw_probs: dict[str, list[float]] = {}
    for event in events:
        for bk in event.get("bookmakers", []):
            for market in bk.get("markets", []):
                if market.get("key") != "outrights":
                    continue
                outcomes = market["outcomes"]
                overround = sum(1.0 / o["price"] for o in outcomes)
                for o in outcomes:
                    imp = (1.0 / o["price"]) / overround  # de-vigged
                    raw_probs.setdefault(o["name"], []).append(imp)

    if not raw_probs:
        sys.exit("Outrights market found but no outcomes parsed — check market key.")

    # Average across bookmakers; re-normalise to sum to 1.
    avg = {team: sum(ps) / len(ps) for team, ps in raw_probs.items()}
    total = sum(avg.values())
    return {team: p / total for team, p in avg.items()}


# ---------------------------------------------------------------------------
# Style stats from historical data
# ---------------------------------------------------------------------------

def style_stats() -> dict[str, tuple[float, float]]:
    """Return {style: (mean, std)} from HISTORICAL."""
    buckets: dict[str, list[int]] = {}
    for _, _, goals, style in HISTORICAL:
        buckets.setdefault(style, []).append(goals)

    stats: dict[str, tuple[float, float]] = {}
    for style, gs in buckets.items():
        mu = sum(gs) / len(gs)
        if len(gs) > 1:
            variance = sum((g - mu) ** 2 for g in gs) / (len(gs) - 1)
            sigma = math.sqrt(variance)
        else:
            sigma = 2.5  # fallback for single-sample styles
        stats[style] = (mu, sigma)
    return stats


# ---------------------------------------------------------------------------
# Distribution math — no scipy, stdlib only
# ---------------------------------------------------------------------------

def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def normal_pmf(k: int, mu: float, sigma: float) -> float:
    """P(X = k) approximated as area of Normal(mu, sigma) over [k-0.5, k+0.5]."""
    return _phi((k + 0.5 - mu) / sigma) - _phi((k - 0.5 - mu) / sigma)


def build_mixture(
    win_probs: dict[str, float],
    sstats: dict[str, tuple[float, float]],
    k_range: range,
) -> dict[int, float]:
    pmf: dict[int, float] = {k: 0.0 for k in k_range}
    unknown: list[str] = []

    for team, p_win in win_probs.items():
        style = TEAM_STYLE.get(team)
        if style is None:
            unknown.append(team)
            style = "balanced"
        mu, sigma = sstats[style]
        for k in k_range:
            pmf[k] += p_win * normal_pmf(k, mu, sigma)

    if unknown:
        print(f"[warn] No style entry for: {', '.join(unknown)} — defaulting to 'balanced'.")
        print("       Add them to TEAM_STYLE in champion_goals.py for better accuracy.\n")

    # Re-normalise (normal tails may leak outside k_range).
    total = sum(pmf.values())
    return {k: v / total for k, v in pmf.items()}


def cdf_percentile(pmf: dict[int, float], pct: float) -> int:
    """Smallest k with CDF >= pct."""
    cumulative = 0.0
    for k in sorted(pmf):
        cumulative += pmf[k]
        if cumulative >= pct:
            return k
    return max(pmf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Fetching WC26 outright winner odds...")
    win_probs = fetch_win_probs()

    sstats = style_stats()

    # Filter to teams with at least 1% win probability (trim noise).
    relevant = {t: p for t, p in win_probs.items() if p >= 0.01}
    total_p = sum(relevant.values())
    relevant = {t: p / total_p for t, p in relevant.items()}  # re-normalise

    print("Top contenders (de-vigged, re-normalised to shown teams):")
    print(f"  {'Team':<22} {'Win%':>6}  Style        Hist μ (goals)")
    print(f"  {'-'*22} {'-'*6}  {'-'*11}  {'-'*14}")
    for team, p in sorted(relevant.items(), key=lambda x: -x[1])[:12]:
        style = TEAM_STYLE.get(team, "balanced")
        mu, _ = sstats.get(style, (0.0, 0.0))
        print(f"  {team:<22} {p*100:5.1f}%  {style:<11}  {mu:.1f}")

    k_range = range(4, 26)
    pmf = build_mixture(relevant, sstats, k_range)

    mean_val = sum(k * v for k, v in pmf.items())
    median   = cdf_percentile(pmf, 0.50)
    p40      = cdf_percentile(pmf, 0.40)  # CWOGO: ~60% chance actual >= bid

    print(f"\nGoal distribution (probability-weighted across contenders):")
    for k in k_range:
        if pmf[k] < 0.005:
            continue
        bar = "█" * int(pmf[k] * 300)
        tag = ""
        if k == p40:
            tag = " ← recommended bid"
        elif k == median:
            tag = " ← median"
        print(f"  {k:2d} goals: {pmf[k]*100:4.1f}%  {bar}{tag}")

    print(f"\n  Expected (mean): {mean_val:.1f} goals")
    print(f"  Median:          {median} goals")
    print()
    print(f"  Recommended bid (40th pct): {p40} goals")
    print(f"  — bidding at the 40th percentile means ~60% chance the actual")
    print(f"    exceeds your bid, so you don't go over. Adjust up if you think")
    print(f"    others in the pool will bid conservatively.")


if __name__ == "__main__":
    main()
