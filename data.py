"""Fetch + cache the two data sources: The Odds API and Elevenify (Datawrapper).

Elevenify CSV layout: paired team rows ("Mexico | :mx: | Group A | 1.76 | 48% | 62%",
then a "v" separator row, then the away team). Parsed with the stdlib csv module —
pandas adds no leverage on this irregular layout. Each team row gives predicted goals
directly, which is a stronger blend signal than a win%-derived proxy.
"""

from __future__ import annotations

import csv
import difflib
import io
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger("wc.data")

# The Odds API (the-odds-api.com). Override via env for a different provider.
ODDS_BASE = os.environ.get(
    "ODDS_API_BASE",
    "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds",
)
# Elevenify republishes this same Datawrapper chart each round. The unversioned
# static CSV (static.dwcdn.net/data/<id>.csv) freezes on an old version, so we
# resolve the latest published version from the embed and fetch its dataset.
ELEVENIFY_CHART_ID = "8CjZ3"
ELEVENIFY_EMBED = f"https://datawrapper.dwcdn.net/{ELEVENIFY_CHART_ID}/"

_HTTP_TIMEOUT = 20
_ODDS_CACHE = Path(__file__).parent / ".odds_cache.json"
_ELEVENIFY_CACHE = Path(__file__).parent / ".elevenify_cache.json"
_ELEVENIFY_TTL = 4 * 3600  # seconds; Elevenify updates between rounds, not continuously

# Quota min-maxxing. The Odds API bills ~3 credits per odds call (3 markets x 1
# region) and ~2 per scores call (daysFrom variant). Two guards protect a forced
# refresh from wasting credits; both are bypassable per-call via override=True.
#   - debounce: ignore a forced refresh if the cache is younger than this.
#   - quota floor: refuse a forced refresh once remaining credits hit this reserve
#     (0 disables). Stops the last credits leaking into a silent sample-data fallback.
_ODDS_MIN_INTERVAL = int(os.environ.get("ODDS_MIN_REFRESH_SECONDS", "60"))
_ODDS_QUOTA_FLOOR = int(os.environ.get("ODDS_QUOTA_FLOOR", "0"))

_odds_quota: int | None = None
_last_odds_action: str = "init"  # fetched | cache | debounced | quota_floor | fallback


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def odds_fetched_at() -> str | None:
    """Return the ISO timestamp when odds were last fetched, or None if no cache."""
    if not _ODDS_CACHE.exists():
        return None
    try:
        return json.loads(_ODDS_CACHE.read_text()).get("fetched_at")
    except Exception:
        return None


def odds_quota() -> int | None:
    """Return the most recently observed x-requests-remaining from The Odds API."""
    return _odds_quota


def last_odds_action() -> str:
    """Outcome of the last fetch_odds call (fetched/cache/debounced/quota_floor/fallback)."""
    return _last_odds_action


def _read_odds_cache() -> dict | None:
    if not _ODDS_CACHE.exists():
        return None
    try:
        return json.loads(_ODDS_CACHE.read_text())
    except Exception:
        return None


def _iso_age(ts: str | None) -> float | None:
    """Seconds since an ISO timestamp, or None if unparseable."""
    if not ts:
        return None
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# The Odds API                                                                #
# --------------------------------------------------------------------------- #
def fetch_odds(force: bool = False, override: bool = False) -> list[dict]:
    """Return cached odds if available, otherwise fetch from The Odds API.

    Pass force=True to bypass the cache and hit the API (~3 credits). A forced
    fetch is still suppressed by the debounce and quota-floor guards unless
    override=True. Inspect last_odds_action() to see what actually happened.
    """
    global _odds_quota, _last_odds_action
    cached = _read_odds_cache()

    if cached is not None and _odds_quota is None and cached.get("quota") is not None:
        _odds_quota = cached["quota"]

    if not force and cached is not None:
        log.info("Loading odds from cache (%s).", _ODDS_CACHE)
        _last_odds_action = "cache"
        return cached["events"]

    # Forced refresh: apply credit-saving guards (skippable via override).
    if force and cached is not None and not override:
        age = _iso_age(cached.get("fetched_at"))
        if age is not None and age < _ODDS_MIN_INTERVAL:
            log.info("Odds refresh debounced (%.0fs < %ds) - serving cache.", age, _ODDS_MIN_INTERVAL)
            _last_odds_action = "debounced"
            return cached["events"]
        if _ODDS_QUOTA_FLOOR and _odds_quota is not None and _odds_quota <= _ODDS_QUOTA_FLOOR:
            log.warning(
                "Odds refresh blocked: %d credits left <= floor %d - serving cache.",
                _odds_quota, _ODDS_QUOTA_FLOOR,
            )
            _last_odds_action = "quota_floor"
            return cached["events"]

    key = os.environ.get("ODDS_API_KEY")
    if not key:
        log.warning("ODDS_API_KEY not set - serving sample fixtures.")
        _last_odds_action = "fallback"
        return _sample_odds()
    # Pinnacle is the sharpest book (no-vig, limits winners); betfair_ex_eu is the
    # exchange-derived market. Both are far more efficient than soft EU retail books.
    params = {
        "apiKey": key,
        "regions": "eu",
        "markets": "h2h,totals,spreads",  # 'spreads' is The Odds API's handicap key
        "oddsFormat": "decimal",
        "bookmakers": "pinnacle,betfair_ex_eu",
    }
    try:
        r = requests.get(ODDS_BASE, params=params, timeout=_HTTP_TIMEOUT)
        r.raise_for_status()
        try:
            _odds_quota = int(r.headers["x-requests-remaining"])
        except (KeyError, ValueError):
            pass
        events = r.json()
    except Exception as exc:  # noqa: BLE001 - any failure -> graceful fallback
        log.error("Odds fetch failed (%s) - serving sample fixtures.", exc)
        _last_odds_action = "fallback"
        return _sample_odds()
    if not events:
        log.warning("Odds API returned 0 WC events - serving sample fixtures.")
        _last_odds_action = "fallback"
        return _sample_odds()
    log.info("Fetched %d WC events from The Odds API.", len(events))
    normalised = [_normalize_event(ev) for ev in events]
    _ODDS_CACHE.write_text(json.dumps({"fetched_at": _now(), "quota": _odds_quota, "events": normalised}))
    _last_odds_action = "fetched"
    return normalised


_BOOKMAKER_PRIORITY = {"pinnacle": 0, "betfair_ex_eu": 1}


def _collect_market(event: dict, key: str):
    """Yield outcomes-lists for `key`, Pinnacle first then Betfair."""
    bookmakers = sorted(
        event.get("bookmakers", []),
        key=lambda bk: _BOOKMAKER_PRIORITY.get(bk.get("key", ""), 99),
    )
    for bk in bookmakers:
        for m in bk.get("markets", []):
            if m.get("key") == key:
                yield m["outcomes"]


def _normalize_event(event: dict) -> dict:
    home, away = event["home_team"], event["away_team"]
    h2h = totals = handicap = None

    for outs in _collect_market(event, "h2h"):
        price = {o["name"]: o["price"] for o in outs}
        if home in price and away in price and "Draw" in price:
            h2h = {"home": price[home], "draw": price["Draw"], "away": price[away]}
            break

    best_line = None  # use the lowest available O/U line (typically 2.5)
    for outs in _collect_market(event, "totals"):
        over = next((o for o in outs if o["name"] == "Over"), None)
        under = next((o for o in outs if o["name"] == "Under"), None)
        if over and under and over.get("point") is not None:
            line = over["point"]
            if best_line is None or line < best_line:
                best_line = line
                totals = {"line": line, "over": over["price"], "under": under["price"]}

    for outs in _collect_market(event, "spreads"):
        h = next((o for o in outs if o["name"] == home), None)
        a = next((o for o in outs if o["name"] == away), None)
        if h and a and h.get("point") is not None:
            handicap = {"line": h["point"], "home_price": h["price"], "away_price": a["price"]}
            break

    return {
        "home_team": home,
        "away_team": away,
        "commence_time": event.get("commence_time"),
        "h2h": h2h,
        "totals": totals,
        "asian_handicap": handicap,
    }


def _sample_odds() -> list[dict]:
    """Fallback fixtures (real Elevenify team names) so the UI renders pre-tournament."""

    def ev(h, a, hh, dd, aa, line, ov, un, hcap):
        return {
            "home_team": h,
            "away_team": a,
            "commence_time": "2026-06-11T18:00:00Z",
            "h2h": {"home": hh, "draw": dd, "away": aa},
            "totals": {"line": line, "over": ov, "under": un},
            "asian_handicap": {"line": hcap, "home_price": 1.9, "away_price": 1.9},
        }

    return [
        ev("Mexico", "South Africa", 1.55, 4.0, 6.5, 2.5, 1.95, 1.90, -1.0),
        ev("Korea Republic", "Czechia", 2.55, 3.3, 2.75, 2.5, 1.90, 1.95, 0.0),
        ev("Canada", "Bosnia and Herzegovina", 2.10, 3.4, 3.5, 2.5, 1.92, 1.92, -0.25),
    ]


# --------------------------------------------------------------------------- #
# Elevenify (Datawrapper CSVs)                                                #
# --------------------------------------------------------------------------- #
def _pct(s: str | None) -> float | None:
    s = (s or "").strip().replace("%", "")
    return float(s) / 100.0 if s else None


def _num(s: str | None) -> float | None:
    s = (s or "").strip()
    try:
        return float(s) if s else None
    except ValueError:
        return None


def _elevenify_dataset_url() -> str:
    """Resolve the dataset URL for the latest published version of the chart.

    The embed page references the current version as ``<id>/<n>/``; we take the
    highest n and fetch its version-pinned dataset. Falls back to the (possibly
    stale) static CSV if version discovery fails — never hard-fails.
    """
    fallback = f"https://static.dwcdn.net/data/{ELEVENIFY_CHART_ID}.csv"
    try:
        r = requests.get(ELEVENIFY_EMBED, timeout=_HTTP_TIMEOUT)
        r.raise_for_status()
        versions = [int(v) for v in re.findall(rf"{ELEVENIFY_CHART_ID}/(\d+)/", r.text)]
        if not versions:
            log.warning("Elevenify version not found in embed; using static CSV.")
            return fallback
        return f"https://datawrapper.dwcdn.net/{ELEVENIFY_CHART_ID}/{max(versions)}/dataset.csv"
    except Exception as exc:  # noqa: BLE001
        log.warning("Elevenify version lookup failed (%s); using static CSV.", exc)
        return fallback


def fetch_elevenify(force: bool = False) -> list[dict]:
    """Parse the Elevenify match CSV into per-fixture predictions.

    Results are cached for _ELEVENIFY_TTL seconds. Pass force=True to bypass.
    """
    if not force and _ELEVENIFY_CACHE.exists():
        try:
            cached = json.loads(_ELEVENIFY_CACHE.read_text())
            age = datetime.now(timezone.utc).timestamp() - cached["fetched_at"]
            if age < _ELEVENIFY_TTL:
                log.info("Elevenify from cache (%.0f min old).", age / 60)
                return cached["fixtures"]
        except Exception:
            pass

    try:
        r = requests.get(_elevenify_dataset_url(), timeout=_HTTP_TIMEOUT)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        log.error("Elevenify CSV fetch failed: %s", exc)
        return []

    r.encoding = "utf-8"  # CDN omits charset; requests else guesses Latin-1 -> mojibake
    rows = list(csv.reader(io.StringIO(r.text)))
    if not rows:
        return []
    header = [c.strip().lower() for c in rows[0]]
    log.info("Elevenify match CSV columns: %s", header)

    # Real columns: name, flag, group, goals, clean sheet, win.
    teams: list[dict] = []
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        name = row[0].strip()
        if name.lower() == "v":  # fixture separator
            continue
        teams.append(
            {
                "team": name,
                "group": row[2].strip() or None if len(row) > 2 else None,
                "goals": _num(row[3]) if len(row) > 3 else None,
                "cs": _pct(row[4]) if len(row) > 4 else None,
                "win": _pct(row[5]) if len(row) > 5 else None,
            }
        )

    fixtures: list[dict] = []
    for i in range(0, len(teams) - 1, 2):  # consecutive pairs = home, away
        h, a = teams[i], teams[i + 1]
        fixtures.append(
            {
                "home_team": h["team"],
                "away_team": a["team"],
                "group": h["group"],
                "home_goals": h["goals"],
                "away_goals": a["goals"],
                "home_cs": h["cs"],
                "away_cs": a["cs"],
                "home_win": h["win"],
                "away_win": a["win"],
            }
        )
    log.info("Parsed %d Elevenify fixtures.", len(fixtures))

    try:
        _ELEVENIFY_CACHE.write_text(json.dumps({
            "fetched_at": datetime.now(timezone.utc).timestamp(),
            "fixtures": fixtures,
        }))
    except Exception as exc:
        log.warning("Elevenify cache write failed: %s", exc)

    return fixtures


# The Odds API and Elevenify spell several countries differently. Map each side's
# variant to one canonical name so matching is an exact lookup; fuzzy is only a
# safety net for *unforeseen* renames (see match_elevenify). Keys are normalised
# (_norm) forms; values are the canonical normalised name.
_TEAM_ALIASES = {
    "bosnia & herzegovina": "bosnia and herzegovina",
    "cape verde": "cabo verde",
    "cabo verde": "cabo verde",
    "czech republic": "czechia",
    "czechia": "czechia",
    "dr congo": "congo dr",
    "congo dr": "congo dr",
    "iran": "ir iran",
    "ir iran": "ir iran",
    "ivory coast": "côte d'ivoire",
    "côte d'ivoire": "côte d'ivoire",
    "south korea": "korea republic",
    "korea republic": "korea republic",
    "turkey": "türkiye",
    "türkiye": "türkiye",
}

# Fuzzy fallback bar. Sits above every wrong match observed on the real WC slate
# (Czech/Korea Republic 0.71, South Korea/South Africa 0.70) and below real
# variants — so a future rename still auto-matches, but a confident-wrong guess
# can't silently corrupt a fixture. Below this we skip the blend and log instead.
_FUZZY_CUTOFF = 0.85


def _norm(name: str) -> str:
    return name.lower().strip()


def _canon(name: str) -> str:
    """Normalise then resolve through the alias map to a canonical team name."""
    n = _norm(name)
    return _TEAM_ALIASES.get(n, n)


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def match_elevenify(home: str, away: str, fixtures: list[dict], cutoff: float = _FUZZY_CUTOFF):
    """Match an odds fixture to an Elevenify fixture, oriented to (home, away).

    Canonicalises both sides through _TEAM_ALIASES, then takes the first fixture
    whose pair matches exactly (in either orientation). Falls back to whole-pair
    fuzzy scoring only for names the alias map doesn't cover, gated at `cutoff`
    high enough that the real-slate wrong matches (Czech/Korea Republic) can't slip
    through. Returns the Elevenify dict re-keyed so home_* aligns with the odds
    home team, or None (logging the pairing so an alias can be added).
    """
    ch, ca = _canon(home), _canon(away)
    for f in fixtures:
        cfh, cfa = _canon(f["home_team"]), _canon(f["away_team"])
        if {ch, ca} == {cfh, cfa}:
            swapped = (ch, ca) == (cfa, cfh) and ch != ca
            h_pref, a_pref = ("away", "home") if swapped else ("home", "away")
            return {
                "home_goals": f[f"{h_pref}_goals"],
                "away_goals": f[f"{a_pref}_goals"],
                "home_cs": f[f"{h_pref}_cs"],
                "away_cs": f[f"{a_pref}_cs"],
                "home_win": f[f"{h_pref}_win"],
                "away_win": f[f"{a_pref}_win"],
                "group": f.get("group"),
            }

    best: tuple[float, bool, dict] | None = None
    for f in fixtures:
        cfh, cfa = _canon(f["home_team"]), _canon(f["away_team"])
        direct = min(_ratio(ch, cfh), _ratio(ca, cfa))  # both teams must match
        swap = min(_ratio(ch, cfa), _ratio(ca, cfh))
        score, swapped = (direct, False) if direct >= swap else (swap, True)
        if best is None or score > best[0]:
            best = (score, swapped, f)

    if best is None or best[0] < cutoff:
        log.warning("No Elevenify match for %s vs %s (add an alias).", home, away)
        return None

    _, swapped, f = best
    h_pref, a_pref = ("away", "home") if swapped else ("home", "away")
    return {
        "home_goals": f[f"{h_pref}_goals"],
        "away_goals": f[f"{a_pref}_goals"],
        "home_cs": f[f"{h_pref}_cs"],
        "away_cs": f[f"{a_pref}_cs"],
        "home_win": f[f"{h_pref}_win"],
        "away_win": f[f"{a_pref}_win"],
        "group": f.get("group"),
    }


# --------------------------------------------------------------------------- #
# Scores (completed matches)                                                   #
# --------------------------------------------------------------------------- #
SCORES_BASE = os.environ.get(
    "ODDS_API_SCORES_BASE",
    "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/scores",
)
_SCORES_CACHE = Path(__file__).parent / ".scores_cache.json"
_SCORES_STORE = Path(__file__).parent / ".results_store.json"
_SCORES_TTL = 20 * 60


def _load_results_store() -> dict:
    """Load the persistent results store (keyed by canonical team pair)."""
    if not _SCORES_STORE.exists():
        return {}
    try:
        return json.loads(_SCORES_STORE.read_text())
    except Exception:
        return {}


def _save_results_store(store: dict) -> None:
    try:
        _SCORES_STORE.write_text(json.dumps(store))
    except Exception as exc:
        log.warning("Results store write failed: %s", exc)


def fetch_scores(force: bool = False) -> list[dict]:
    """Return all known completed WC26 scores.

    Fetches recent results from The Odds API (max daysFrom=3 per API limits),
    merges them into a persistent store so games beyond the 3-day window are
    retained for the rest of the tournament.
    """
    global _odds_quota
    store = _load_results_store()

    if not force and _SCORES_CACHE.exists():
        try:
            cached = json.loads(_SCORES_CACHE.read_text())
            age = datetime.now(timezone.utc).timestamp() - cached["fetched_at"]
            if age < _SCORES_TTL:
                log.info("Scores from cache (%.0f min old).", age / 60)
                return list(store.values())
        except Exception:
            pass

    key = os.environ.get("ODDS_API_KEY")
    if not key:
        log.warning("ODDS_API_KEY not set — no scores available.")
        return list(store.values())

    try:
        r = requests.get(
            SCORES_BASE,
            params={"apiKey": key, "daysFrom": 3},
            timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        try:  # scores call also reports remaining credits - keep the gauge current
            _odds_quota = int(r.headers["x-requests-remaining"])
        except (KeyError, ValueError):
            pass
        events = r.json()
    except Exception as exc:
        log.error("Scores fetch failed: %s", exc)
        return list(store.values())

    new_count = 0
    for ev in events:
        if not ev.get("completed"):
            continue
        score = _normalize_score(ev)
        k = f"{_canon(score['home_team'])}|{_canon(score['away_team'])}"
        if k not in store:
            new_count += 1
        # Merge so manual annotations (e.g. a "winner" override for a penalty
        # tie) survive a fresh API fetch while the score fields stay current.
        store[k] = {**store.get(k, {}), **score}

    _save_results_store(store)
    try:
        _SCORES_CACHE.write_text(json.dumps({"fetched_at": datetime.now(timezone.utc).timestamp()}))
    except Exception:
        pass
    log.info("Scores: %d total known, %d newly added.", len(store), new_count)
    return list(store.values())


def _normalize_score(ev: dict) -> dict:
    home, away = ev["home_team"], ev["away_team"]
    score_map = {s["name"]: s.get("score") for s in (ev.get("scores") or [])}
    h, a = score_map.get(home), score_map.get(away)
    return {
        "home_team": home,
        "away_team": away,
        "commence_time": ev.get("commence_time"),
        "home_score": int(h) if h is not None else None,
        "away_score": int(a) if a is not None else None,
    }


def match_score(home: str, away: str, scores: list[dict]) -> dict | None:
    """Find the score entry for this fixture, oriented to (home, away)."""
    ch, ca = _canon(home), _canon(away)
    for s in scores:
        sh, sa = _canon(s["home_team"]), _canon(s["away_team"])
        if {ch, ca} == {sh, sa}:
            if (ch, ca) == (sa, sh):
                return {**s, "home_score": s["away_score"], "away_score": s["home_score"]}
            return s
    return None
