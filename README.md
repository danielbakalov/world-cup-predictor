# ⚽ WC26 Scoreline Predictor

A local dashboard that turns World Cup betting markets into scoreline
probabilities. It de-vigs the odds, recovers each team's expected goals, blends
in [Elevenify](https://elevenify.com)'s public predictions, and renders a Poisson
score grid per fixture.

> Probabilities only — pool/stake strategy is on you.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then paste your key into .env
python app.py                 # -> http://localhost:8000
```

Get a free API key at **[the-odds-api.com](https://the-odds-api.com)** (500
requests/month). Until the tournament's odds go live, the app falls back to
sample fixtures so the UI still renders — and the Elevenify blend works
immediately (those CSVs are already live for all 48 teams).

## How it works (`model.py`)

1. **De-vig** the 1X2 and totals markets into fair probabilities.
2. **Invert** `P(total > line)` under a Poisson to recover the total mean `λ_total`.
3. **Split** `λ_total` into `λ_home`/`λ_away` using the Asian handicap. *No
   handicap quoted?* It falls back to inverting 1X2 supremacy under a Skellam —
   so the model still works on the-odds-api.com's free tier, where Asian
   handicaps can be thin.
4. **Blend** with Elevenify: `0.6 · odds_goal_diff + 0.4 · elevenify_goal_diff`,
   then a clean-sheet Dixon-Coles tilt. *(The blend uses Elevenify's actual
   predicted goals per team — a stronger signal than a win%-derived proxy.)*
5. **Grid**: build `P(home=i, away=j)` for 0–8, return the most likely scorelines.

Knobs live at the top of `model.py` (`ELEVENIFY_WEIGHT`, `CS_TILT`, `MAX_GOALS`).

## API

| Route | Purpose |
|---|---|
| `GET /` | Dashboard |
| `GET /api/matches` | Matches + predictions (served from in-memory cache) |
| `GET /api/refresh` | Re-fetch both sources and recompute |

## Files

```
app.py            FastAPI app + per-fixture orchestration
model.py          De-vig, market inversion, blend, Poisson grid
data.py           Odds + Elevenify fetching, caching, fuzzy fixture matching
templates/
  index.html      Dark dashboard (vanilla JS, no build step)
```

## Notes & deviations from the original brief

- **Odds host:** uses `the-odds-api.com` (the brief's host string was a typo for
  it). The handicap market key there is `spreads`, not `asian_handicap`.
- **Elevenify parsing:** the real Datawrapper CSV is a paired-team-row layout,
  not the column-per-fixture shape the brief assumed — parsed with the stdlib
  `csv` module (pandas added nothing here, so it's not a dependency).
- **Fuzzy team matching** uses `difflib`; unmatched fixtures are logged
  (`No Elevenify match for X vs Y`) so you can add aliases.
- **`ODDS_API_KEY` is read from the environment only** — never hardcoded.
