"""FastAPI app: serves the dashboard and the prediction API.

Run with `python app.py` -> http://localhost:8000
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

import data
import model

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


def _points_earned(pick: str, actual: str | None) -> int | None:
    if actual is None:
        return None
    if pick == actual:
        return 2 if pick == "draw" else 1
    return 0


def _compute(force: bool = False) -> None:
    """Fetch all sources, run the model per upcoming fixture, build completed list."""
    odds = data.fetch_odds(force=force)
    elevenify = data.fetch_elevenify(force=force)
    scores = data.fetch_scores(force=force)
    store = _load_store()

    upcoming: list[dict] = []
    for ev in odds:
        elev = (
            data.match_elevenify(ev["home_team"], ev["away_team"], elevenify)
            if elevenify
            else None
        )
        match = {
            "home_team": ev["home_team"],
            "away_team": ev["away_team"],
            "commence_time": ev["commence_time"],
            "group": elev.get("group") if elev else None,
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
                )
                match["prediction"] = {
                    "top_scorelines": [vars(s) for s in pred.top_scorelines],
                    "lambda_home": pred.lambda_home,
                    "lambda_away": pred.lambda_away,
                    "blend_applied": pred.blend_applied,
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
        if match["prediction"]:
            store[_store_key(ev["home_team"], ev["away_team"])] = {
                "home_team": ev["home_team"],
                "away_team": ev["away_team"],
                "commence_time": ev["commence_time"],
                "group": match["group"],
                "elevenify_source": match["elevenify_source"],
                "prediction": match["prediction"],
            }

        upcoming.append(match)

    _save_store(store)

    # Build completed list: scores feed + stored predictions.
    completed: list[dict] = []
    for sc in scores:
        key = _store_key(sc["home_team"], sc["away_team"])
        stored = store.get(key)
        prediction = stored["prediction"] if stored else None
        actual = _actual_outcome(sc["home_score"], sc["away_score"])
        pick = prediction["outcomes"]["optimal_pick"].lower() if prediction else None
        completed.append({
            "home_team": sc["home_team"],
            "away_team": sc["away_team"],
            "commence_time": sc.get("commence_time") or (stored.get("commence_time") if stored else None),
            "group": stored.get("group") if stored else None,
            "home_score": sc["home_score"],
            "away_score": sc["away_score"],
            "actual_outcome": actual,
            "points_earned": _points_earned(pick, actual) if pick else None,
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
def api_refresh():
    _compute(force=True)
    return {
        "status": "ok",
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
