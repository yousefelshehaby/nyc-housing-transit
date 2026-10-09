# Living Next to the Traffic
### Affordable Housing Production × Motor Vehicle Collisions in New York City (2014 – 2025)

**Data Engineering and Visualization — Winter 2026 — Project Milestone 1 (GIU)**

> Are New York's affordable homes being built on its most dangerous streets — and did the Vision-Zero-era safety
> gains reach the neighbourhoods where the city builds the most?

| | |
|---|---|
| **Live website** | **https://nyc-housing-transit.vercel.app** (deployed on Vercel) |
| **Notebook** | [`notebook/NYC_Housing_x_Crashes.ipynb`](notebook/NYC_Housing_x_Crashes.ipynb) (executed, with all outputs) |
| **Housing dataset (H2)** | [Affordable Housing Production by Building](https://data.cityofnewyork.us/Housing-Development/Affordable-Housing-Production-by-Building/hg8x-zxpr) — `hg8x-zxpr` |
| **Transportation dataset (T4)** | [Motor Vehicle Collisions – Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) — `h9gi-nx95` |
| **Geographic helper** | [Modified ZIP Code Tabulation Areas (MODZCTA)](https://data.cityofnewyork.us/Health/Modified-Zip-Code-Tabulation-Areas-MODZCTA-/pri4-ifjk) — `pri4-ifjk` (ZIP boundaries + population, used only as spatial glue / denominator) |

---

## Key findings

Integrated data: **9,012 affordable buildings** (7,200 geocoded, ~306 K affordable units) × **1,928,476 crashes**, linked by
(a) a point-in-polygon join to 177 residential ZIP areas × 12 years and (b) every crash within **250 m** of every building.

| # | Question | Answer |
|---|---|---|
| RQ1 | Do ZIPs with the most affordable housing carry more injuries? | Weakly across all ZIPs (ρ = 0.14), but the top third of ZIPs (82 % of units) has **+19 % injuries per resident** |
| RQ2 | Are the poorest tiers placed in the most dangerous ZIPs? | **No** — every tier is above the city average, and middle-income units are the highest |
| RQ3 | Do crashes near a building change after a new building starts? | **No measurable effect** — new construction tracks preservation sites (null result) |
| RQ4 | Where are affordable buildings on the danger map? | The most dangerous ZIP quartile has 23 % of residents but **33 % of units** |
| RQ5 | Do the housing pipeline and crashes move together? | No — crashes −63 %, injuries only −19 %, while starts stayed at 25–35 K a year |
| RQ6 | When is the doorstep dangerous? | Evenings, nights and weekends are over-represented near housing; rush hours are under-represented |
| RQ7 | Are crash causes different near affordable housing? | More pedestrian confusion (≈ 2×), turning, backing and distraction; fewer rear-ends |
| RQ8 | Does doorstep exposure differ by borough and construction type? | New construction sits on busier streets in **every** borough (2.2× in Queens) |
| RQ9 | Did safety gains reach high-growth ZIPs? | **No** — injuries per resident are not lower in 2024–25 than in 2014–15, and the gap persists |
| RQ10 | Are family-sized units in dangerous ZIPs? | A lower share but **2× more per resident** in the most dangerous quartile |
| RQ11 | How do the metrics correlate? | The link is specific to **pedestrians (ρ 0.42) and cyclists (0.38)**, not deaths |
| RQ12 | Which vehicles are involved near affordable housing? | More e-bikes (1.65×), taxis (1.42×), buses (1.41×) and motorcycles (1.31×) |

**Limits:** correlation, not causation; police-reported data; static population; commuter-heavy ZIPs inflate per-resident rates.
**Impact:** DOT can schedule Vision Zero redesigns around the affordable pipeline before residents move in, and HPD can add a
street-safety review to site selection.

---

## Repository structure

```
nyc-housing-transit/
├── app.py                         # Dash website (filters, search mode, Generate Report, 9 charts, story)
├── assets/style.css               # Website styling (auto-loaded by Dash)
├── notebook/
│   ├── NYC_Housing_x_Crashes.ipynb   # Deliverable: EDA → cleaning → integration → post-cleaning → 12 RQs
│   ├── nyc_housing_crashes.py        # Same notebook in jupytext "percent" format (easy diffs / code review)
│   └── build_notebook.py             # Rebuilds + executes the .ipynb from the .py source
├── data/
│   ├── raw/                       # housing CSV + MODZCTA GeoJSON (crash CSV is downloaded by the notebook, git-ignored)
│   └── processed/                 # small pre-aggregated tables written by notebook §8 and read by the website
├── requirements.txt               # website runtime
├── requirements-notebook.txt      # + analysis libraries for the notebook
├── render.yaml / Procfile / .python-version   # deployment config (Render or Heroku)
└── README.md
```

## Setup (local)

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-notebook.txt
```

**Run the website** (uses the processed tables already committed in `data/processed/`):

```bash
python app.py                      # → http://localhost:8050
```

**Re-run the full pipeline** (downloads ~470 MB of crash data on first run, ~10 min):

```bash
jupyter nbconvert --to notebook --execute --inplace notebook/NYC_Housing_x_Crashes.ipynb --ExecutePreprocessor.timeout=3600
```

The notebook caches raw downloads in `data/raw/` and rewrites `data/processed/` (section 8), so the website always
reflects the latest notebook run.

## Deployment

The website is a standard Dash app (`app:server` is the WSGI entry point) and needs < 300 MB of RAM, so it runs on free tiers.

**Vercel (current deployment, free, no card):**
1. On [vercel.com/new](https://vercel.com/new) import the GitHub repository — Vercel detects the Python app automatically
   (`app.py` exposes `app`, the Flask server behind Dash; `vercel.json` excludes the notebook and raw data from the bundle).
2. Click **Deploy**. Every push to `main` redeploys automatically.

**Render (alternative, free plan but asks for card verification):**
1. Push this folder to a GitHub repository (the folder itself must be the repo root, or set *Root Directory* in Render).
2. On [render.com](https://render.com) → **New + → Blueprint** → select the repo. `render.yaml` configures everything:
   build `pip install -r requirements.txt`, start `gunicorn app:server`, Python 3.11.9.
   (Or **New + → Web Service** and enter those two commands manually.)
3. Wait for the build, open the `*.onrender.com` URL, click **Generate Report** and try a search such as
   `Bronx since 2019 pedestrians speeding near housing` to confirm everything works.
4. Paste the URL at the top of this README and in the submission e-mail.

**Heroku / Railway:** the `Procfile` (`web: gunicorn app:server`) and `.python-version` are picked up automatically.

> Free Render services sleep after 15 min of inactivity; the first request after that takes ~30 s.

## Website features (mapped to the requirements)

| Requirement | Implementation |
|---|---|
| Multiple dropdown filters | Borough (multi), From year, To year, **housing** construction type, **housing** income tier, **transport** victim type, **transport** contributing factor (multi), **transport** distance-to-housing |
| Search mode | Free-text box parsed into filters — boroughs/abbreviations, single years, ranges (`2016-2018`), `since 2019`, `before 2020`, `new buildings`/`preservation`, income tiers, `pedestrians`/`cyclists`/`motorists`, factors (`speeding`, `distracted`, `drunk`, `red light`…), `near housing`/`elsewhere`. Recognised terms are shown as chips and a report is generated immediately |
| Central **Generate Report** button | Every KPI and chart is recomputed from the current filters only when the button (or a search) is triggered |
| Variety of interactive charts | Choropleth **map** + building bubbles, bubble **scatter** with trend line, dual-axis **bar/line** time series with range slider, hour × weekday **heatmap**, diverging **bar**, grouped **box plot**, event-study **line** with confidence band, multi-**line** trend, horizontal **bar** — all with hover, zoom and filter updates |
| Cross-domain relationship | Map (ZIP danger × building locations), scatter (units per resident vs injuries per resident), income-tier exposure, event study, tercile trends |
| Story section | Question, data, discovery, limits, impact — at the bottom of the page |

## Team contributions

Team of 5 (tutorial group P003) → 10 required research questions (2 per member). RQ11 and RQ12 are **additional**
questions the team analysed together.

| Member | ID | Research questions & visualizations | Data / engineering / website work |
|---|---|---|---|
| Nada Adel Shawki Othman | 16007138 | **RQ1** ZIP scatter: units per 1k residents vs injuries · **RQ2** income-tier exposure bar | Data loading & caching (SODA API, per-year crash download), dataset overview, housing EDA |
| Saged Mohamed Atef Mohamed Hegazy Badr | 16008234 | **RQ3** event study around project start · **RQ4** choropleth map + buildings | Housing pre-integration cleaning (dates, structural zeros, confidential records, IQR flags) |
| Youssef Mohamed Ahmed Sobhy Abdelhamed Elhawary | 16006726 | **RQ5** housing pipeline vs crashes time series · **RQ6** hour × weekday heatmap | Crash pre-integration cleaning (timestamps, invalid coordinates, ZIPs, injury reconciliation, outlier rule) |
| Youssef Mohamed Yehia Amin Ibrahim Elshehabi | 13005381 | **RQ7** contributing factors near vs elsewhere · **RQ8** doorstep exposure box plot | Contributing-factor & vehicle-type standardisation; Dash website, search parser, deployment (Vercel) |
| Zeyad Mahmoud Ahmed Abdelhakeem Galal | 16005534 | **RQ9** Vision-Zero trend by housing-growth tercile · **RQ10** family units by danger quartile | Integration (point-in-polygon spatial join, ZIP × year panel, 250 m proximity join), post-integration cleaning, export for the website |
| Whole team (additional) | — | **RQ11** Spearman correlation heatmap · **RQ12** vehicle-type mix near vs elsewhere | Story section, README, final review |

## AI assistance

An AI assistant (Claude, Anthropic) was used to help write code and documentation. The prompts used are listed in
section 10 of the notebook, as required by the project description.
