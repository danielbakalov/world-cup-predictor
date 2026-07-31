# ⚽ WC26 Scoreline Predictor

A local dashboard that turned World Cup betting markets into match predictions
for a friends' prediction pool. It de-vigs the bookmakers' odds, recovers each
team's expected goals, blends in [Elevenify](https://elevenify.com)'s public
predictions, and builds a Poisson score grid per fixture.

The part that mattered: it doesn't stop at probabilities. It picks the outcome
that maximises **expected points under the pool's scoring rule** — which is not
the same thing as the most likely outcome, and that gap is the entire edge.

## Results

Ran for all 104 matches of WC26. Final record:

| Round | Matches | Correct | Points |
|---|---:|---:|---:|
| Group Stage | 72 | 43 (60%) | 53 |
| Round of 32 | 16 | 14 (88%) | 28 |
| Round of 16 | 8 | 6 (75%) | 12 |
| Quarter-Final | 4 | 4 (100%) | 12 |
| Semi-Final | 2 | 0 (0%) | 0 |
| Third Place | 1 | 0 (0%) | 0 |
| Final | 1 | 1 (100%) | 10 |
| **Total** | **104** | **68 (65%)** | **115** |

Good enough to finish **tied for second** in the pool. It went 0-for-3 across
both semi-finals and the third-place match, which is the honest caveat on a
65% hit rate — the sample is small and the tail is fat.

**Did the scoring-rule exploit actually pay?** Replaying the same stored
predictions with a naive "pick the most likely outcome" rule scores **108**.
The expected-points rule scored **115** — so the exploit was worth **+7 points**.
It picked Draw 28 times and hit 10 of them.

Both numbers are reproducible from this repo: `.predictions_store.json` and
`.results_store.json` are committed, and every prediction was **locked at
kickoff** (see `kicked_off_keys` in `app.py`) so live in-play odds could never
leak backwards into a pick that had already been made.

## The idea

The pool paid **1 point** for correctly calling a win and **2 points** for
correctly calling a draw. That asymmetry changes the optimal strategy: you
should pick Draw whenever

```
2 · P(draw)  >  P(favourite)
```

i.e. whenever `P(draw) > P(favourite) / 2` — even though a draw is *rarely* the
single most likely result. A 30% draw against a 45% favourite is a bad
prediction and a good bet. Most people in a pool like this pick the team they
think will win; the model picks Draw far more often than feels comfortable, and
over 72 group matches that is where the points came from.

In the knockouts draws can't stand (extra time and penalties resolve them), so
the draw probability is redistributed into each side's "to advance" probability
and the pick is simply the favourite. The round's point value then scales
`expected_points` but never changes *which* team gets picked.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then paste your key into .env
python app.py                 # -> http://localhost:8000
```

Get a free API key at **[the-odds-api.com](https://the-odds-api.com)** (500
requests/month). With no key set, the app falls back to a few sample fixtures so
the UI still renders.

Two things worth knowing before you run it:

- `app.py` binds **`0.0.0.0`**, so the dashboard is reachable from anywhere on
  your LAN. Change it to `127.0.0.1` at the bottom of `app.py` if you'd rather
  it stayed local.
- `WC_KNOCKOUT` **defaults to on** (`model.py`), because that's where the
  tournament finished. Set `WC_KNOCKOUT=0` to re-enable Draw as a legal pick for
  group-stage fixtures.

## How it works (`model.py`)

1. **De-vig** the 1X2 and totals markets into fair probabilities.
2. **Invert** `P(total > line)` under a Poisson to recover the total mean `λ_total`.
3. **Split** `λ_total` into `λ_home`/`λ_away` using the Asian handicap. *No
   handicap quoted?* It falls back to inverting 1X2 supremacy under a Skellam —
   so the model still works on the-odds-api.com's free tier, where Asian
   handicaps can be thin.
4. **Blend** with Elevenify at `0.8 · odds + 0.2 · elevenify`
   (`ELEVENIFY_WEIGHT = 0.2`), applied to *both* the total goals and the goal
   difference, which are then re-split into λ's.
5. **Grid**: build `P(home=i, away=j)` for 0–8 with a Dixon-Coles low-score
   correction, and sum it into P(home win) / P(draw) / P(away win).
6. **Pick**: group stage takes the argmax of `{1·P(home), 2·P(draw), 1·P(away)}`;
   knockouts redistribute the draw and take the favourite.

Knobs live at the top of `model.py`: `ELEVENIFY_WEIGHT`, `DC_RHO`, `MAX_GOALS`,
`LAMBDA_FLOOR`, `KNOCKOUT_STAGE`.

## Scoring (`scoring.py`)

`round_of(commence_time)` maps a kickoff to a tournament round from the WC26
schedule, and `points_earned(...)` prices the pick:

| Round | Correct win | Correct draw |
|---|---:|---:|
| Group Stage | 1 | 2 |
| Round of 32 / 16 | 2 | — |
| Quarter-Final | 3 | — |
| Semi-Final | 5 | — |
| Third Place | 1 | — |
| Final | 10 | — |

Two details that took a bug to find:

- **The 8-hour day shift.** Round boundaries are dates, but a 9pm US kickoff is
  already "tomorrow" in UTC. Kickoffs are shifted back 8h before the date is
  taken, so a late evening match lands in the round it was actually played in.
- **Penalty shootouts.** The scores feed reports the post-extra-time score, so a
  match settled on penalties reads as level and the advancing side is unknown.
  Those score `None` and show a grey **pending** chip until a manual
  `"winner": "home"|"away"` is added to that fixture in `.results_store.json`.
  The refresh merge preserves manual fields, so it survives re-fetching.

## Design notes

- **The blend deliberately ignores Elevenify's win/draw/loss percentages.** Only
  their expected *goals* feed in. Taking their outcome probabilities directly
  would short-circuit the Poisson model and, with it, the scoring-rule exploit —
  you'd inherit someone else's most-likely-outcome answer instead of computing
  your own expected-points one.
- **Elevenify freshness is decided by timestamp, not by source.** They update
  the same Datawrapper chart each round by *two* different mechanisms, and which
  one is freshest flips between them: group→R32 they published a new version
  (the versioned `dataset.csv` won, the static CSV froze), but R32→R16 they
  edited the data in place (the static CSV won, the versioned one froze at the
  previous round). Trusting either URL alone silently serves stale data for a
  whole round — which is exactly what happened. So `_fetch_freshest_elevenify_csv()`
  fetches both candidates and picks by `Last-Modified`.
- **Dixon-Coles tau** is applied only to the four correlated low-score cells
  (0-0, 1-0, 0-1, 1-1). `DC_RHO = 0.1` is deliberately conservative — real fits
  land higher, but this model is already leaning hard on draws and doesn't need
  help getting there.
- **Team names never hard-fail.** An alias map (`_TEAM_ALIASES` in `data.py`)
  runs first, then whole-pair fuzzy matching gated at 0.85 — high enough that
  Korea Republic and Czechia can't cross-match.

## Champion goals bonus (`champion_goals.py`)

Standalone script for the pool bonus question: *total goals scored by the
championship team, closest without going over.*

```bash
python champion_goals.py
```

Pulls outright winner odds, de-vigs them into win probabilities, then weights
each team's expected goal output against historical WC winner distributions
(1986–2022, all 7-game tournaments). Outputs a distribution and a recommended
bid at the **40th percentile** (~60% chance the actual exceeds your bid).

The main variable is `TEAM_STYLE` at the top of the script — verify style
classifications for any team before bidding.

## API

| Route | Purpose |
|---|---|
| `GET /` | Dashboard |
| `GET /api/matches` | Matches + predictions (served from cache) |
| `GET /api/refresh?scope=odds\|scores\|all&override=0\|1` | Re-fetch and recompute |

Odds calls cost ~3 credits and scores ~2, so forced refreshes are guarded by a
debounce and a quota floor (`ODDS_MIN_REFRESH_SECONDS`, `ODDS_QUOTA_FLOOR` —
both documented in `.env.example`). `override=1` bypasses both.

## Files

```
app.py                FastAPI app + per-fixture orchestration
model.py              De-vig, market inversion, blend, Poisson grid, pick
scoring.py            Round detection + per-round point values
data.py               Odds/Elevenify/scores fetching, caching, fuzzy matching
champion_goals.py     Standalone champion goals bid estimator
templates/
  index.html          Dark dashboard (vanilla JS, no build step)

.predictions_store.json   Every prediction made, locked at kickoff
.results_store.json       Every final score (the scores API only serves 3 days)
```

## Disclaimer

A personal project built for a friends' prediction pool, published because the
modelling was interesting. It is not betting advice, and a 104-match sample
proves considerably less than it appears to.

Elevenify data is fetched from their public Datawrapper CDN and remains their
copyright — it is not covered by this repo's license, and nothing here is
affiliated with or endorsed by Elevenify, FIFA, or The Odds API.

## License

[MIT](LICENSE)
