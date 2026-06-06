"""FastAPI app: serves the dashboard and the prediction API.

Run with `python app.py` -> http://localhost:8000
"""

from __future__ import annotations

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
_state: dict = {"matches": [], "updated_at": None}

def _compute(force: bool = False) -> list[dict]:
    """Fetch both sources, run the model per fixture, cache the result."""
    odds = data.fetch_odds(force=force)
    elevenify = data.fetch_elevenify(force=force)
    matches: list[dict] = []

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
            except Exception as exc:  # noqa: BLE001 - one bad fixture shouldn't 500 the board
                log.error(
                    "Prediction failed for %s v %s: %s",
                    ev["home_team"],
                    ev["away_team"],
                    exc,
                )

        matches.append(match)

    _state["matches"] = matches
    _state["updated_at"] = data.odds_fetched_at()
    log.info("Computed %d matches.", len(matches))
    return matches


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
        "odds_quota": data.odds_quota(),
    }


@app.get("/api/refresh")
def api_refresh():
    _compute(force=True)
    return {
        "status": "ok",
        "matches_count": len(_state["matches"]),
        "timestamp": _state["updated_at"],
        "odds_quota": data.odds_quota(),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (TEMPLATES / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
