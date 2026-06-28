"""FastAPI app: serves the dashboard and the prediction API.

Run with `python app.py` -> http://localhost:8000
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

import data
import model
import scoring

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("wc.app")

TEMPLATES = Path(__file__).parent / "templates"
_state: dict = {"matches": [], "completed": [], "updated_at": None, "total_points": 0}

_PREDICTIONS_STORE = Path(__file__).parent / ".predictions_store.json"


def _load_store() -> dict:
    if not _PREDICTIONS_STORE.exists():
        return {}
    try:
        return json.loads(_PREDICTIONS_STORE.read_text())
    except Exception:
        return {}


def _save_store(store: dict) -> None:
    try:
        _PREDICTIONS_STORE.write_text(json.dumps(store))
    except Exception as exc:
        log.warning("Predictions store write failed: %s", exc)


def _store_key(home: str, away: str) -> str:
    return f"{data._canon(home)}|{data._canon(away)}"


def _actual_outcome(home_score, away_score) -> str | None:
    if home_score is None or away_score is None:
        return None
    if home_score > away_score:
        return "home"
    if away_score > home_score:
        return "away"
    return "draw"


# Scoring is round-aware (see scoring.py): group win 1 / draw 2; knockout wins
# scale R32/R16 2, QF 3, SF 5, Final 10. Penalty ties await a manual winner.


def _compute(force_odds: bool = False, force_scores: bool = False, override: bool = False) -> None:
    """Fetch all sources, run the model per upcoming fixture, build completed list.

    force_odds and force_scores bypass their caches independently so a routine
    refresh only spends credits on what it needs (odds ~3, scores ~2). Elevenify
    is a free Datawrapper CDN fetch, so it refreshes alongside odds to keep the
    blend consistent with the prices it feeds.
    """
    odds = data.fetch_odds(force=force_odds, override=override)
    elevenify = data.fetch_elevenify(force=force_odds)
    scores = data.fetch_scores(force=force_scores)
    store = _load_store()

    # Keys for matches that have kicked off — predictions must not be overwritten after kickoff.
    now = datetime.now(timezone.utc)
    completed_keys = {
        _store_key(sc["home_team"], sc["away_team"])
        for sc in scores
        if sc.get("home_score") is not None and sc.get("away_score") is not None
    }
    kicked_off_keys = {
        _store_key(ev["home_team"], ev["away_team"])
        for ev in odds
        if datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")) <= now
    }

    upcoming: list[dict] = []
    for ev in odds:
        elev = (
            data.match_elevenify(ev["home_team"], ev["away_team"], elevenify)
            if elevenify
            else None
        )
        rnd = scoring.round_of(ev["commence_time"])
        match = {
            "home_team": ev["home_team"],
            "away_team": ev["away_team"],
            "commence_time": ev["commence_time"],
            "group": elev.get("group") if elev else None,
            "round": rnd,
            "odds_source": {
                "h2h": ev["h2h"],
                "totals": ev["totals"],
                "asian_handicap": ev["asian_handicap"],
            },
            "elevenify_source": (
                {
                    "home_win_pct": elev["home_win"],
                    "away_win_pct": elev["away_win"],
                    "home_cs_pct": elev["home_cs"],
                    "away_cs_pct": elev["away_cs"],
                    "home_goals": elev["home_goals"],
                    "away_goals": elev["away_goals"],
                    "group": elev.get("group"),
                }
                if elev
                else None
            ),
            "prediction": None,
        }

        if ev["h2h"] and ev["totals"]:
            try:
                pred = model.predict(
                    ml=ev["h2h"],
                    totals=ev["totals"],
                    handicap=ev["asian_handicap"]["line"] if ev["asian_handicap"] else None,
                    elevenify=elev,
                    knockout=model.KNOCKOUT_STAGE,
                    win_points=scoring.win_points(rnd),
                )
                match["prediction"] = {
                    "top_scorelines": [vars(s) for s in pred.top_scorelines],
                    "lambda_home": pred.lambda_home,
                    "lambda_away": pred.lambda_away,
                    "blend_applied": pred.blend_applied,
                    "knockout": pred.knockout,
                    "outcomes": vars(pred.outcomes),
                }
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Prediction failed for %s v %s: %s",
                    ev["home_team"],
                    ev["away_team"],
                    exc,
                )

        # Persist prediction so it survives after the match leaves the odds feed.
        # Lock once the match has kicked off — live/in-play odds corrupt the pre-match pick.
        key = _store_key(ev["home_team"], ev["away_team"])
        if match["prediction"] and key not in kicked_off_keys:
            store[key] = {
                "home_team": ev["home_team"],
                "away_team": ev["away_team"],
                "commence_time": ev["commence_time"],
                "group": match["group"],
                "round": rnd,
                "elevenify_source": match["elevenify_source"],
                "prediction": match["prediction"],
            }

        if key not in completed_keys:
            upcoming.append(match)

    _save_store(store)

    # Build completed list: scores feed + stored predictions.
    completed: list[dict] = []
    for sc in scores:
        key = _store_key(sc["home_team"], sc["away_team"])
        stored = store.get(key)
        prediction = stored["prediction"] if stored else None
        commence_time = sc.get("commence_time") or (stored.get("commence_time") if stored else None)
        rnd = scoring.round_of(commence_time)
        actual = _actual_outcome(sc["home_score"], sc["away_score"])
        pick = prediction["outcomes"]["optimal_pick"].lower() if prediction else None
        winner = sc.get("winner")  # manual override for penalty-decided ties
        points = scoring.points_earned(rnd, pick, actual, winner) if pick else None
        # A penalty-decided knockout shows the advancing side once recorded.
        display_actual = winner if (rnd != "group" and actual == "draw" and winner) else actual
        completed.append({
            "home_team": sc["home_team"],
            "away_team": sc["away_team"],
            "commence_time": commence_time,
            "group": stored.get("group") if stored else None,
            "round": rnd,
            "home_score": sc["home_score"],
            "away_score": sc["away_score"],
            "actual_outcome": display_actual,
            "points_earned": points,
            "elevenify_source": stored.get("elevenify_source") if stored else None,
            "prediction": prediction,
        })

    _state["matches"] = upcoming
    _state["completed"] = completed
    _state["total_points"] = sum(
        m["points_earned"] for m in completed if m["points_earned"] is not None
    )
    _state["updated_at"] = data.odds_fetched_at()
    log.info("Computed %d upcoming, %d completed matches.", len(upcoming), len(completed))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _compute()
    yield


app = FastAPI(title="WC26 Scoreline Predictor", lifespan=lifespan)


@app.get("/api/matches")
def api_matches():
    return {
        "updated_at": _state["updated_at"],
        "matches": _state["matches"],
        "completed": _state["completed"],
        "total_points": _state["total_points"],
        "odds_quota": data.odds_quota(),
    }


@app.get("/api/refresh")
def api_refresh(scope: str = "odds", override: bool = False):
    """Scoped, credit-aware refresh.

    scope=odds   -> prices + free Elevenify blend (~3 credits)   [default]
    scope=scores -> completed-match results only (~2 credits)
    scope=all    -> both (~5 credits)
    override=1   -> bypass the debounce / quota-floor guards on the odds fetch.
    """
    _compute(
        force_odds=scope in ("odds", "all"),
        force_scores=scope in ("scores", "all"),
        override=override,
    )
    return {
        "status": "ok",
        "scope": scope,
        "odds_action": data.last_odds_action(),
        "matches_count": len(_state["matches"]),
        "completed_count": len(_state["completed"]),
        "total_points": _state["total_points"],
        "timestamp": _state["updated_at"],
        "odds_quota": data.odds_quota(),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (TEMPLATES / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
