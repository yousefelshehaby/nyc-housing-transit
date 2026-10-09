"""
Living Next to the Traffic — interactive website (Dash / Plotly)

Explores NYC Affordable Housing Production by Building (HPD) together with
Motor Vehicle Collisions – Crashes (NYPD), 2014-2025.

All data comes from the pre-aggregated tables written by the notebook
(`notebook/NYC_Housing_x_Crashes.ipynb`, section 8) into `data/processed/`.
Run locally with `python app.py`; in production `gunicorn app:server`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from plotly.subplots import make_subplots

DATA_DIR = Path(__file__).parent / "data" / "processed"
YEARS = list(range(2014, 2026))
BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
BOROUGH_COLORS = {"Bronx": "#66c2a5", "Brooklyn": "#fc8d62", "Manhattan": "#8da0cb", "Queens": "#e78ac3", "Staten Island": "#a6d854"}
TIERS = {
    "all_counted_units": "All affordable units",
    "extremely_low_income_units": "Extremely low income (≤30% AMI)",
    "very_low_income_units": "Very low income (31–50% AMI)",
    "low_income_units": "Low income (51–80% AMI)",
    "moderate_income_units": "Moderate income (81–120% AMI)",
    "middle_income_units": "Middle income (121–165% AMI)",
    "family_units": "Family-sized units (2+ bedrooms)",
}
VICTIMS = {"injured": "All people injured", "ped_injured": "Pedestrians injured", "cyc_injured": "Cyclists injured",
           "mot_injured": "Motorists injured", "crashes": "All crashes (any severity)"}
PROXIMITY = {"all": "All crashes", "near": "Within 250 m of affordable housing", "far": "More than 250 m away"}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
INK, MUTED, ACCENT, ACCENT_2 = "#1d2433", "#6b7280", "#e4572e", "#17807e"

# ---------------------------------------------------------------- data
buildings = pd.read_parquet(DATA_DIR / "buildings.parquet")
building_year = pd.read_parquet(DATA_DIR / "building_year.parquet")
crash_zip = pd.read_parquet(DATA_DIR / "crash_zip_month.parquet")
crash_hour = pd.read_parquet(DATA_DIR / "crash_hour_weekday.parquet")
population = pd.read_csv(DATA_DIR / "modzcta_population.csv", dtype={"modzcta": str})
population = population[population["pop_est"] >= 1000]          # non-residential MODZCTA (99999) excluded
with open(DATA_DIR / "modzcta_simplified.geojson", encoding="utf-8") as fh:
    ZIP_GEOJSON = json.load(fh)

buildings["modzcta"] = buildings["modzcta"].astype(str)
crash_zip["modzcta"] = crash_zip["modzcta"].astype(str)
FACTORS = sorted(f for f in crash_zip["factor_group"].unique() if f not in ("Unspecified", "Other"))
CITYWIDE_CRASHES_BY_YEAR = crash_zip.groupby("year")["crashes"].sum()

# ---------------------------------------------------------------- search parser
BOROUGH_WORDS = {"bronx": "Bronx", "bx": "Bronx", "brooklyn": "Brooklyn", "bk": "Brooklyn", "manhattan": "Manhattan",
                 "mn": "Manhattan", "queens": "Queens", "qn": "Queens", "staten island": "Staten Island", "staten": "Staten Island",
                 "si": "Staten Island"}
TIER_WORDS = [("extremely low", "extremely_low_income_units"), ("very low", "very_low_income_units"),
              ("low income", "low_income_units"), ("moderate", "moderate_income_units"), ("middle", "middle_income_units"),
              ("family", "family_units"), ("bedroom", "family_units")]
VICTIM_WORDS = [("pedestrian", "ped_injured"), ("walk", "ped_injured"), ("cyclist", "cyc_injured"), ("bike", "cyc_injured"),
                ("bicycl", "cyc_injured"), ("motorist", "mot_injured"), ("driver", "mot_injured"), ("injur", "injured"),
                ("crash", "crashes"), ("collision", "crashes")]
FACTOR_WORDS = [("speed", "Unsafe speed"), ("yield", "Failure to yield right-of-way"),
                ("distract", "Driver inattention / distraction"), ("inattention", "Driver inattention / distraction"),
                ("phone", "Driver inattention / distraction"), ("alcohol", "Alcohol / drugs"), ("drunk", "Alcohol / drugs"),
                ("drug", "Alcohol / drugs"), ("backing", "Backing unsafely"), ("aggressive", "Aggressive driving"),
                ("road rage", "Aggressive driving"), ("following", "Following too closely"), ("tailgat", "Following too closely"),
                ("red light", "Disregarded traffic control"), ("traffic control", "Disregarded traffic control"),
                ("lane", "Improper passing / lane use / turning"), ("turning", "Improper passing / lane use / turning"),
                ("fatigue", "Driver fatigue / illness"), ("asleep", "Driver fatigue / illness"), ("pavement", "Road / environment"),
                ("defect", "Vehicle defect"), ("brake", "Vehicle defect")]


def parse_query(text: str) -> tuple[dict, list[str]]:
    """Translate a free-text query such as 'Brooklyn 2022 new buildings pedestrians' into filter values."""
    q = (text or "").lower()
    found, notes = {}, []
    boroughs = sorted({b for word, b in BOROUGH_WORDS.items() if re.search(rf"\b{word}\b", q)})
    if boroughs:
        found["boroughs"] = boroughs
    years = sorted(int(y) for y in re.findall(r"\b(20[12]\d)\b", q) if 2014 <= int(y) <= 2025)
    if re.search(r"\b(since|after|from)\s+20[12]\d", q) and len(years) == 1:
        found["years"] = (years[0], 2025)
    elif re.search(r"\b(before|until|through)\s+20[12]\d", q) and len(years) == 1:
        found["years"] = (2014, years[0])
    elif years:
        found["years"] = (years[0], years[-1])
    if re.search(r"new (construction|build|buildings?|developments?|homes?|units?|projects?)|newly built", q):
        found["ctype"] = "New Construction"
    elif re.search(r"preserv|renovat|rehab", q):
        found["ctype"] = "Preservation"
    for word, tier in TIER_WORDS:
        if word in q:
            found["tier"] = tier
            break
    for word, victim in VICTIM_WORDS:
        if word in q:
            found["victim"] = victim
            break
    factors = sorted({f for word, f in FACTOR_WORDS if word in q})
    if factors:
        found["factors"] = factors
    if re.search(r"near|doorstep|close to|around (affordable )?housing", q):
        found["proximity"] = "near"
    elif re.search(r"elsewhere|far from|away from", q):
        found["proximity"] = "far"
    if not found and q.strip():
        notes.append("No filters recognised — try e.g. “Bronx 2019-2023 pedestrians speeding near housing”.")
    return found, notes


# ---------------------------------------------------------------- filtering helpers
def filter_buildings(boroughs, y0, y1, ctype) -> pd.DataFrame:
    df = buildings[buildings["start_year"].between(y0, y1)]
    if boroughs:
        df = df[df["borough"].isin(boroughs)]
    if ctype != "all":
        df = df[df["reporting_construction_type"] == ctype]
    return df


def filter_crashes(table: pd.DataFrame, boroughs, y0, y1, factors, proximity) -> pd.DataFrame:
    df = table[table["year"].between(y0, y1)]
    if boroughs:
        df = df[df["borough"].isin(boroughs)]
    if factors:
        df = df[df["factor_group"].isin(factors)]
    if proximity == "near":
        df = df[df["near_housing"]]
    elif proximity == "far":
        df = df[~df["near_housing"]]
    return df


def zip_table(bld: pd.DataFrame, crashes: pd.DataFrame, tier: str, victim: str, n_years: int) -> pd.DataFrame:
    """ZIP-level integration: affordable units (selected tier) + crash metric per 10k residents per year."""
    units = bld.groupby("modzcta")[tier].sum().rename("units")
    harm = crashes.groupby("modzcta")[victim].sum().rename("harm")
    out = population.set_index("modzcta").join([units, harm]).fillna({"units": 0, "harm": 0})
    out["units_per_1k"] = out["units"] / out["pop_est"] * 1000
    out["rate"] = out["harm"] / out["pop_est"] * 10_000 / n_years
    return out.reset_index()


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color=MUTED))
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, template="plotly_white", height=380)
    return fig


def style(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=30, b=10),
                      font=dict(family="Inter, system-ui, sans-serif", color=INK, size=12),
                      legend=dict(orientation="h", y=-0.18, x=0), hoverlabel=dict(font_size=12))
    return fig


# ---------------------------------------------------------------- figures
def fig_map(zips: pd.DataFrame, bld: pd.DataFrame, victim_label: str, tier_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Choroplethmap(
        geojson=ZIP_GEOJSON, locations=zips["modzcta"], z=zips["rate"], featureidkey="properties.modzcta",
        colorscale="OrRd", zmin=0, zmax=float(np.nanpercentile(zips["rate"], 97)) or 1, marker_line_width=0.3,
        marker_line_color="white", marker_opacity=0.75,
        colorbar=dict(title=dict(text="per 10k<br>per year", side="top"), thickness=12, len=0.7),
        customdata=np.stack([zips["units"], zips["pop_est"]], axis=1),
        hovertemplate="<b>ZIP %{location}</b><br>" + victim_label + ": %{z:.1f} per 10k residents / yr"
                      "<br>affordable units: %{customdata[0]:,.0f}<br>population: %{customdata[1]:,.0f}<extra></extra>",
        name="ZIP rate"))
    if len(bld):
        sizes = np.clip(np.sqrt(bld["units_selected"].clip(lower=1)) * 0.7, 3, 18)
        fig.add_trace(go.Scattermap(
            lat=bld["latitude"], lon=bld["longitude"], mode="markers",
            marker=dict(size=sizes, color="#1d3557", opacity=0.45),
            customdata=np.stack([bld["project_name"], bld["units_selected"], bld["start_year"],
                                 bld["reporting_construction_type"], bld["crashes_250m_per_year"].round(1)], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>" + tier_label + ": %{customdata[1]:,}<br>started %{customdata[2]} · "
                          "%{customdata[3]}<br>crashes within 250 m: %{customdata[4]} / yr<extra></extra>",
            name="affordable buildings"))
    # frame the map on the selected buildings (whole city by default)
    if 0 < len(bld) and (bld["latitude"].max() - bld["latitude"].min()) < 0.25:
        center, zoom = dict(lat=float(bld["latitude"].median()), lon=float(bld["longitude"].median())), 10.6
    else:
        center, zoom = dict(lat=40.705, lon=-73.95), 9.4
    fig.update_layout(map=dict(style="carto-positron", center=center, zoom=zoom),
                      margin=dict(l=0, r=0, t=0, b=0), height=560, showlegend=False)
    return fig


def fig_scatter(zips: pd.DataFrame, victim_label: str, tier_label: str) -> go.Figure:
    df = zips[(zips["units"] > 0)].copy()
    if len(df) < 3:
        return empty_figure("Not enough ZIPs with affordable units for this selection")
    df["borough"] = df["modzcta"].map(lambda z: next((b for b, p in [("Manhattan", ("100", "101", "102")), ("Staten Island", ("103",)),
                                                                     ("Bronx", ("104",)), ("Brooklyn", ("112",))] if z.startswith(p)), "Queens"))
    fig = go.Figure()
    for borough, grp in df.groupby("borough"):
        fig.add_trace(go.Scatter(
            x=grp["units_per_1k"], y=grp["rate"], mode="markers", name=borough,
            marker=dict(size=np.clip(np.sqrt(grp["units"]) / 2.2, 6, 34), color=BOROUGH_COLORS[borough], opacity=0.8,
                        line=dict(color="white", width=1)),
            customdata=np.stack([grp["modzcta"], grp["units"], grp["pop_est"]], axis=1),
            hovertemplate="<b>ZIP %{customdata[0]}</b><br>units/1k residents: %{x:.1f}<br>" + victim_label +
                          " per 10k/yr: %{y:.1f}<br>units: %{customdata[1]:,.0f}<extra>" + borough + "</extra>"))
    # log-linear least-squares trend + Spearman rho
    x, y = np.log10(df["units_per_1k"]), df["rate"]
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    fig.add_trace(go.Scatter(x=10 ** xs, y=intercept + slope * xs, mode="lines", name="trend",
                             line=dict(color=INK, dash="dash", width=1.5), hoverinfo="skip"))
    rho = df["units_per_1k"].rank().corr(df["rate"].rank())
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False, align="left",
                       text=f"Spearman ρ = {rho:.2f} · n = {len(df)} ZIPs", font=dict(color=MUTED))
    fig.update_xaxes(type="log", title=f"{tier_label} per 1,000 residents (log)",
                     tickvals=[0.01, 0.1, 1, 10, 100, 1000], ticktext=["0.01", "0.1", "1", "10", "100", "1,000"])
    fig.update_yaxes(title=f"{victim_label} per 10k residents / yr")
    return style(fig, 430)


def fig_timeline(crashes: pd.DataFrame, bld: pd.DataFrame, victim: str, victim_label: str, tier: str, tier_label: str) -> go.Figure:
    monthly = crashes.groupby(["year", "month"])[victim].sum().reset_index()
    monthly["date"] = pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=1))
    units = bld.groupby("start_year")[tier].sum()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=pd.to_datetime(units.index.astype(str) + "-07-01"), y=units.values, name=f"{tier_label} started (per year)",
                         marker_color=ACCENT_2, opacity=0.35, width=1000 * 3600 * 24 * 300,
                         hovertemplate="%{x|%Y}: %{y:,} units<extra></extra>"), secondary_y=True)
    fig.add_trace(go.Scatter(x=monthly["date"], y=monthly[victim], name=f"{victim_label} (per month)", mode="lines",
                             line=dict(color=ACCENT, width=2), hovertemplate="%{x|%b %Y}: %{y:,}<extra></extra>"), secondary_y=False)
    fig.add_vrect(x0="2020-03-01", x1="2020-06-30", fillcolor="grey", opacity=0.12, line_width=0,
                  annotation_text="COVID lockdown", annotation_position="top left")
    fig.update_yaxes(title_text=f"{victim_label} / month", secondary_y=False)
    fig.update_yaxes(title_text="affordable units started / year", secondary_y=True, showgrid=False)
    fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.06)
    return style(fig, 430)


def fig_heatmap(hours: pd.DataFrame, victim: str, victim_label: str) -> go.Figure:
    if hours.empty:
        return empty_figure("No crashes for this selection")
    grid = hours.pivot_table(index="weekday", columns="hour", values=victim, aggfunc="sum", fill_value=0)
    grid = grid.reindex(index=range(7), columns=range(24), fill_value=0)
    share = grid / max(grid.values.sum(), 1) * 100
    fig = go.Figure(go.Heatmap(z=share.values, x=[f"{h:02d}:00" for h in range(24)], y=DAYS, colorscale="Magma_r",
                               customdata=grid.values,
                               hovertemplate="%{y} %{x}<br>%{z:.2f}% of " + victim_label.lower() + "<br>(%{customdata:,})<extra></extra>",
                               colorbar=dict(title="% of total", thickness=12)))
    fig.update_yaxes(autorange="reversed")
    return style(fig, 360)


def fig_factors(hours: pd.DataFrame, victim: str) -> go.Figure:
    df = hours[~hours["factor_group"].isin(["Unspecified", "Unknown"])]
    if df.empty or df["near_housing"].nunique() < 2:
        return empty_figure("Select “All crashes” proximity to compare near vs. elsewhere")
    share = df.pivot_table(index="factor_group", columns="near_housing", values=victim, aggfunc="sum", fill_value=0)
    share = share / share.sum() * 100
    share["diff"] = share[True] - share[False]
    share = share.sort_values("diff")
    fig = go.Figure(go.Bar(x=share["diff"], y=share.index, orientation="h",
                           marker_color=np.where(share["diff"] > 0, ACCENT, "#3a6ea5"),
                           customdata=np.stack([share[True], share[False]], axis=1),
                           hovertemplate="<b>%{y}</b><br>near housing: %{customdata[0]:.1f}%<br>elsewhere: %{customdata[1]:.1f}%"
                                         "<br>difference: %{x:+.2f} pp<extra></extra>"))
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_xaxes(title="share near housing − share elsewhere (pp)")
    return style(fig, 420)


def fig_box(bld: pd.DataFrame, by_year: pd.DataFrame, victim: str, victim_label: str) -> go.Figure:
    col = victim if victim in ("crashes", "injured", "ped_injured", "cyc_injured") else "injured"
    exposure = by_year.groupby("record_id")[col].mean().rename("exposure")
    df = bld.join(exposure, on="record_id")
    if df.empty:
        return empty_figure("No buildings for this selection")
    fig = go.Figure()
    for ctype, color in [("New Construction", ACCENT), ("Preservation", ACCENT_2)]:
        sub = df[df["reporting_construction_type"] == ctype]
        fig.add_trace(go.Box(x=sub["borough"], y=sub["exposure"], name=ctype, marker_color=color, boxpoints="outliers",
                             marker_size=3, customdata=sub["project_name"],
                             hovertemplate="%{customdata}<br>%{y:.1f} per year<extra>" + ctype + "</extra>"))
    label = {"crashes": "crashes", "injured": "people injured", "ped_injured": "pedestrians injured",
             "cyc_injured": "cyclists injured"}[col]
    fig.update_layout(boxmode="group")
    fig.update_xaxes(categoryorder="array", categoryarray=BOROUGHS)
    fig.update_yaxes(title=f"{label} per year within 250 m", type="log" if col == "crashes" else "linear")
    return style(fig, 420)


def fig_tiers(bld: pd.DataFrame, zips: pd.DataFrame, victim_label: str, selected_tier: str) -> go.Figure:
    rate = zips.set_index("modzcta")["rate"]
    df = bld[bld["modzcta"].isin(rate.index)]
    rows = []
    for tier, label in TIERS.items():
        if tier == "all_counted_units" or df[tier].sum() == 0:
            continue
        rows.append((label, np.average(df["modzcta"].map(rate), weights=df[tier]), df[tier].sum(), tier))
    if not rows:
        return empty_figure("No units for this selection")
    out = pd.DataFrame(rows, columns=["tier", "rate", "units", "key"])
    city = np.average(zips["rate"], weights=zips["pop_est"])
    fig = go.Figure(go.Bar(x=out["rate"], y=out["tier"], orientation="h", customdata=out["units"],
                           marker_color=[ACCENT if k == selected_tier else "#9aa5b1" for k in out["key"]],
                           hovertemplate="<b>%{y}</b><br>" + victim_label + ": %{x:.2f} per 10k / yr<br>%{customdata:,} units<extra></extra>"))
    fig.add_vline(x=city, line_dash="dash", line_color=INK, annotation_text="average resident", annotation_position="top")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=f"rate in the unit's ZIP (per 10k residents / yr)")
    return style(fig, 380)


def fig_event(bld: pd.DataFrame, by_year: pd.DataFrame) -> go.Figure:
    eligible = bld[bld["start_year"].between(2017, 2022)][["record_id", "start_year", "reporting_construction_type"]]
    df = by_year.merge(eligible, on="record_id")
    if df.empty:
        return empty_figure("Pick a year range that includes 2017–2022 starts for the before/after view")
    df["rel"] = df["year"] - df["start_year"]
    df = df[df["rel"].between(-3, 3)].copy()
    df["share"] = df["crashes"] / df["year"].map(CITYWIDE_CRASHES_BY_YEAR) * 1e5
    fig = go.Figure()
    for ctype, color in [("New Construction", ACCENT), ("Preservation", ACCENT_2)]:
        sub = df[df["reporting_construction_type"] == ctype]
        if sub.empty:
            continue
        base = sub.loc[sub["rel"] < 0, "share"].mean()
        stats_ = sub.groupby("rel")["share"].agg(["mean", "sem"])
        idx, err = stats_["mean"] / base * 100, 1.96 * stats_["sem"] / base * 100
        fig.add_trace(go.Scatter(x=list(idx.index) + list(idx.index[::-1]), y=list(idx + err) + list((idx - err)[::-1]),
                                 fill="toself", fillcolor=color, opacity=0.15, line_width=0, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=idx.index, y=idx, mode="lines+markers", name=f"{ctype} (n = {sub['record_id'].nunique():,})",
                                 line=dict(color=color, width=2.5),
                                 hovertemplate="year %{x:+d}: index %{y:.1f}<extra>" + ctype + "</extra>"))
    fig.add_hline(y=100, line_color=MUTED, line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color=MUTED, annotation_text="project start")
    fig.update_xaxes(title="years relative to project start", dtick=1)
    fig.update_yaxes(title="trend-adjusted crashes within 250 m (pre = 100)")
    return style(fig, 400)


def fig_terciles(bld: pd.DataFrame, crashes: pd.DataFrame, tier: str, victim: str, victim_label: str) -> go.Figure:
    units = bld.groupby("modzcta")[tier].sum()
    zips = population.set_index("modzcta").join(units.rename("units")).fillna({"units": 0})
    zips = zips[zips["borough"].notna()]
    if zips["units"].sum() == 0:
        return empty_figure("No affordable units for this selection")
    zips["tercile"] = pd.qcut((zips["units"] / zips["pop_est"]).rank(method="first"), 3,
                              labels=["low housing growth", "medium", "high housing growth"])
    yearly = crashes.groupby(["modzcta", "year"])[victim].sum().reset_index()
    yearly = yearly.merge(zips[["tercile", "pop_est"]], left_on="modzcta", right_index=True)
    pops = zips.groupby("tercile", observed=True)["pop_est"].sum()
    trend = yearly.groupby(["tercile", "year"], observed=True)[victim].sum().reset_index()
    trend["rate"] = trend[victim] / trend["tercile"].map(pops).astype(float) * 10_000
    fig = go.Figure()
    for tercile, color in zip(["low housing growth", "medium", "high housing growth"], ["#3a6ea5", "#9aa5b1", ACCENT]):
        sub = trend[trend["tercile"] == tercile]
        fig.add_trace(go.Scatter(x=sub["year"], y=sub["rate"], mode="lines+markers", name=tercile, line=dict(color=color, width=2.5),
                                 hovertemplate="%{x}: %{y:.1f} per 10k<extra>" + tercile + "</extra>"))
    fig.update_xaxes(title="year", dtick=1)
    fig.update_yaxes(title=f"{victim_label} per 10k residents")
    return style(fig, 400)


# ---------------------------------------------------------------- layout
def card(title: str, graph_id: str, question: str, wide: bool = False) -> html.Div:
    # scroll-zoom only on the map, so scrolling the page over the other charts still scrolls the page
    config = {"displaylogo": False, "scrollZoom": graph_id == "g-map"}
    return html.Div(className="card wide" if wide else "card", children=[
        html.Div(className="card-head", children=[html.H3(title), html.P(question, className="question")]),
        dcc.Loading(dcc.Graph(id=graph_id, config=config), type="dot", color=ACCENT),
    ])


def dropdown(label: str, comp_id: str, options, value, multi: bool = False, clearable: bool = False,
             placeholder: str | None = None) -> html.Div:
    return html.Div(className="control", children=[
        html.Label(label, htmlFor=comp_id),
        dcc.Dropdown(id=comp_id, options=options, value=value, multi=multi, clearable=clearable, placeholder=placeholder),
    ])


STORY = [
    ("The question", "New York builds or preserves tens of thousands of affordable homes every year. A home is also a street: "
     "children walk to school from it, seniors cross the avenue in front of it. We asked whether the city's affordable homes are "
     "being placed on its most dangerous streets — and whether the Vision Zero-era safety gains reached the neighbourhoods where "
     "the city builds the most."),
    ("The data", "HPD's Affordable Housing Production by Building (2014–2025 starts, ~7,200 geocoded buildings) and NYPD's Motor "
     "Vehicle Collisions (≈1.9 M crashes 2014–2025). They share no key, so we built two bridges: every point was placed in its ZIP "
     "polygon (MODZCTA, which also gives population), and every building was linked to all crashes within 250 m (≈ 2–3 blocks). "
     "About a third of crashes lacked a borough or ZIP; we recovered most of them from coordinates instead of dropping them."),
    ("The discovery", "**Affordable homes are concentrated where streets are most dangerous.** The quarter of ZIPs with the "
     "highest pedestrian + cyclist injury rates houses 23 % of New Yorkers but received **33 % of all affordable units** started "
     "2014–2025. The third of ZIPs that absorbed the most units (82 % of all units) has **19 % more traffic injuries per resident** "
     "than the rest of the city. At the doorstep, new-construction buildings sit on busier streets than preserved ones in every "
     "borough (2.2× in Queens). And the gap did not close: injury rates per resident were no lower in 2024–25 than in 2014–15 in "
     "any group of ZIPs, and the high-growth ZIPs stayed the most dangerous. Surprise: the risk is not concentrated in the "
     "*poorest* tier — every income tier, from ≤30 % AMI to middle income, lives in ZIPs above the city average."),
    ("The limits", "Crashes are police-reported and miss many minor injuries; population is a single static estimate; a ZIP is a big "
     "area, and 250 m buffers overlap in dense neighbourhoods. Most importantly, this is correlation, not causation: affordable "
     "housing does not cause crashes — both are drawn to dense, transit-rich, wide-avenue neighbourhoods. Confidential "
     "homeownership units (≈ 2 % of units) have no location and are excluded from maps."),
    ("The impact", "The affordable-housing pipeline is a ready-made map of where tomorrow's pedestrians will be. **DOT** can "
     "use it to schedule Vision Zero street redesigns (daylighting, leading pedestrian intervals, protected bike lanes, slower "
     "signal timing) around new projects *before* residents move in. **HPD** could add a street-safety review to site selection "
     "and to the Housing Our Neighbors pipeline. **Advocates and community boards** can use this site to show which buildings "
     "sit inside the most dangerous blocks and to ask for safety funding to follow housing funding."),
]


def story_section() -> html.Div:
    return html.Div(className="story", id="story", children=[
        html.H2("The story"),
        html.Div(className="story-grid", children=[
            html.Div(className="story-item", children=[html.H4(title), dcc.Markdown(text)]) for title, text in STORY]),
    ])


dash_app = Dash(__name__, title="Living Next to the Traffic · NYC",
           external_stylesheets=["https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"])
server = dash_app.server
app = server   # WSGI entrypoint: Vercel loads `app` from app.py, gunicorn uses `app:server`

dash_app.layout = html.Div(className="page", children=[
    html.Header(className="hero", children=[
        html.Div(className="hero-inner", children=[
            html.P("NYC Open Data · Housing & Development × Transportation · 2014–2025", className="eyebrow"),
            html.H1("Living Next to the Traffic"),
            html.P("Are New York's affordable homes being built on its most dangerous streets? Explore 1.9 million crashes "
                   "alongside every affordable-housing building the city started since 2014.", className="lede"),
            html.Nav([html.A("Explore", href="#controls"), html.A("Story", href="#story"), html.A("Data & team", href="#about")]),
        ]),
    ]),
    html.Main(children=[
        html.Section(id="controls", className="controls", children=[
            html.Div(className="search-row", children=[
                dcc.Input(id="search", type="text", debounce=False, n_submit=0,
                          placeholder="Search, e.g. “Brooklyn 2022 new buildings pedestrians” or “Bronx since 2019 speeding near housing”"),
                html.Button("Apply search", id="search-btn", className="btn secondary"),
            ]),
            html.Div(id="search-feedback", className="search-feedback"),
            html.Div(className="filter-grid", children=[
                dropdown("Borough", "f-borough", [{"label": b, "value": b} for b in BOROUGHS], [], multi=True, clearable=True,
                         placeholder="All boroughs"),
                dropdown("From year", "f-y0", [{"label": y, "value": y} for y in YEARS], 2014),
                dropdown("To year", "f-y1", [{"label": y, "value": y} for y in YEARS], 2025),
                dropdown("Housing: construction type", "f-ctype",
                         [{"label": "All types", "value": "all"}, {"label": "New construction", "value": "New Construction"},
                          {"label": "Preservation", "value": "Preservation"}], "all"),
                dropdown("Housing: income tier", "f-tier", [{"label": v, "value": k} for k, v in TIERS.items()], "all_counted_units"),
                dropdown("Transport: who was hurt", "f-victim", [{"label": v, "value": k} for k, v in VICTIMS.items()], "injured"),
                dropdown("Transport: contributing factor", "f-factor", [{"label": f, "value": f} for f in FACTORS], [], multi=True,
                         clearable=True, placeholder="All factors"),
                dropdown("Transport: distance to housing", "f-prox", [{"label": v, "value": k} for k, v in PROXIMITY.items()], "all"),
            ]),
            html.Div(className="generate-row", children=[
                html.Button("Generate Report", id="generate", className="btn primary", n_clicks=0),
                html.Span(id="report-status", className="report-status"),
            ]),
        ]),
        html.Section(id="kpis", className="kpis"),
        html.Section(className="grid", children=[
            card("Map · danger by ZIP and where affordable homes are", "g-map",
                 "ZIP colour = selected injury rate per 10k residents per year; circles = affordable buildings (size = units). "
                 "Zoom, pan and hover.", wide=True),
            card("Housing vs. street danger, ZIP by ZIP", "g-scatter",
                 "RQ1 · Do ZIPs that absorbed more affordable housing carry a higher injury burden?"),
            card("Who lives with the risk? Income tiers", "g-tiers",
                 "RQ2 · The injury rate of the ZIP each affordable unit is built in, by affordability tier."),
            card("Housing pipeline vs. crashes over time", "g-timeline",
                 "RQ5 · Affordable units started per year (bars) against monthly harm (line). Drag the slider to zoom.", wide=True),
            card("When is the doorstep dangerous?", "g-heatmap", "RQ6 · Share of harm by weekday × hour for the selected crashes."),
            card("Why do crashes happen near affordable housing?", "g-factors",
                 "RQ7 · Contributing-factor share near affordable housing minus elsewhere (red = over-represented near housing)."),
            card("Doorstep exposure by borough", "g-box",
                 "RQ8 · Per-building average per year within 250 m, new construction vs. preservation."),
            card("Before & after a new building", "g-event",
                 "RQ3 · Crashes within 250 m around project start, adjusted for the citywide trend (pre-period = 100)."),
            card("Did safety gains reach high-growth ZIPs?", "g-terciles",
                 "RQ9 · Injury rate by year for ZIPs grouped by affordable units per resident.", wide=True),
        ]),
        story_section(),
        html.Section(id="about", className="about", children=[
            html.H2("Data & team"),
            dcc.Markdown("""
* **Housing (H2):** [Affordable Housing Production by Building](https://data.cityofnewyork.us/Housing-Development/Affordable-Housing-Production-by-Building/hg8x-zxpr) — HPD
* **Transportation (T4):** [Motor Vehicle Collisions – Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) — NYPD
* **Geography:** [Modified ZIP Code Tabulation Areas (MODZCTA)](https://data.cityofnewyork.us/Health/Modified-Zip-Code-Tabulation-Areas-MODZCTA-/pri4-ifjk) — boundaries & population
* Full cleaning / integration pipeline and member contributions: see the project notebook and README.
            """),
        ]),
    ]),
    html.Footer("Data Engineering & Visualization · GIU · Winter 2026 — correlation is not causation."),
    dcc.Store(id="search-trigger"),
])


# ---------------------------------------------------------------- callbacks
@dash_app.callback(
    Output("f-borough", "value"), Output("f-y0", "value"), Output("f-y1", "value"), Output("f-ctype", "value"),
    Output("f-tier", "value"), Output("f-victim", "value"), Output("f-factor", "value"), Output("f-prox", "value"),
    Output("search-feedback", "children"), Output("search-trigger", "data"),
    Input("search-btn", "n_clicks"), Input("search", "n_submit"),
    State("search", "value"), prevent_initial_call=True,
)
def apply_search(_clicks, _submit, text):
    """Search mode: parse the query, reset unspecified filters to defaults, and trigger a new report."""
    found, notes = parse_query(text)
    if not found:
        return (*[no_update] * 8, notes[0] if notes else "", no_update)
    y0, y1 = found.get("years", (2014, 2025))
    values = [found.get("boroughs", []), y0, y1, found.get("ctype", "all"), found.get("tier", "all_counted_units"),
              found.get("victim", "injured"), found.get("factors", []), found.get("proximity", "all")]
    chips = []
    labels = {"boroughs": "Borough", "years": "Years", "ctype": "Construction", "tier": "Tier", "victim": "Victims",
              "factors": "Factor", "proximity": "Distance"}
    for key, val in found.items():
        lookup = {"tier": TIERS, "victim": VICTIMS, "proximity": PROXIMITY}.get(key)
        shown = lookup[val] if lookup else val
        if isinstance(shown, (list, tuple)):
            shown = "–".join(map(str, shown)) if key == "years" else ", ".join(shown)
        chips.append(html.Span(f"{labels[key]}: {shown}", className="chip"))
    return (*values, [html.Span("Applied: ", className="muted"), *chips], {"query": text, "n": (_clicks or 0) + (_submit or 0)})


@dash_app.callback(
    Output("kpis", "children"), Output("g-map", "figure"), Output("g-scatter", "figure"), Output("g-tiers", "figure"),
    Output("g-timeline", "figure"), Output("g-heatmap", "figure"), Output("g-factors", "figure"), Output("g-box", "figure"),
    Output("g-event", "figure"), Output("g-terciles", "figure"), Output("report-status", "children"),
    Input("generate", "n_clicks"), Input("search-trigger", "data"),
    State("f-borough", "value"), State("f-y0", "value"), State("f-y1", "value"), State("f-ctype", "value"),
    State("f-tier", "value"), State("f-victim", "value"), State("f-factor", "value"), State("f-prox", "value"),
)
def generate_report(_n, _search, boroughs, y0, y1, ctype, tier, victim, factors, proximity):
    """The 'Generate Report' button: recompute every KPI and chart from the current filters."""
    y0, y1 = min(y0, y1), max(y0, y1)
    n_years = y1 - y0 + 1
    victim_label, tier_label = VICTIMS[victim], TIERS[tier]

    bld = filter_buildings(boroughs, y0, y1, ctype).copy()
    bld["units_selected"] = bld[tier]
    bld_units = bld[bld["units_selected"] > 0]
    crashes = filter_crashes(crash_zip, boroughs, y0, y1, factors, proximity)
    hours = filter_crashes(crash_hour, boroughs, y0, y1, factors, proximity)
    by_year = building_year[building_year["year"].between(y0, y1) & building_year["record_id"].isin(bld_units["record_id"])]

    zips = zip_table(bld_units, crashes, tier, victim, n_years)
    if boroughs:
        zips = zips[zips["borough"].isin(boroughs)]

    # KPIs
    high_danger = zips["rate"] >= zips["rate"].quantile(0.75) if len(zips) else pd.Series(dtype=bool)
    units_total = bld_units["units_selected"].sum()
    units_high = zips.loc[high_danger, "units"].sum() if len(zips) else 0
    rate_total = zips["harm"].sum() / max(zips["pop_est"].sum(), 1) * 10_000 / n_years
    kpis = [
        ("Affordable units", f"{units_total:,.0f}", tier_label),
        ("Buildings", f"{len(bld_units):,}", "geocoded, started in range"),
        ("Crashes", f"{crashes['crashes'].sum():,.0f}", "matching transport filters"),
        (victim_label if victim != "crashes" else "People injured",
         f"{crashes[victim if victim != 'crashes' else 'injured'].sum():,.0f}",
         f"{crashes['killed'].sum():,.0f} killed"),
        ("Rate per 10k residents / yr", f"{rate_total:,.1f}", victim_label.lower()),
        ("Units in the most dangerous 25 % of ZIPs", f"{(units_high / units_total * 100) if units_total else 0:.0f}%",
         "a fair share would be ≈ 25 %"),
    ]
    kpi_children = [html.Div(className="kpi", children=[html.Span(t, className="kpi-label"), html.Strong(v), html.Span(s, className="kpi-sub")])
                    for t, v, s in kpis]

    figures = [
        fig_map(zips, bld_units, victim_label, tier_label),
        fig_scatter(zips, victim_label, tier_label),
        fig_tiers(bld, zips, victim_label, tier),
        fig_timeline(crashes, bld, victim, victim_label, tier, tier_label),
        fig_heatmap(hours, victim, victim_label),
        fig_factors(hours, victim),
        fig_box(bld_units, by_year, victim, victim_label),
        fig_event(bld_units, building_year[building_year["record_id"].isin(bld_units["record_id"])]),
        fig_terciles(bld, crashes, tier, victim, victim_label),
    ]
    try:
        trigger = "search" if ctx.triggered_id == "search-trigger" else "filters"
    except Exception:   # called outside a Dash request (tests)
        trigger = "filters"
    status = f"Report generated from {trigger}: {', '.join(boroughs) if boroughs else 'all boroughs'} · {y0}–{y1} · " \
             f"{len(bld_units):,} buildings · {crashes['crashes'].sum():,.0f} crashes"
    return (kpi_children, *figures, status)


if __name__ == "__main__":
    dash_app.run(debug=False, host="0.0.0.0", port=8050)
