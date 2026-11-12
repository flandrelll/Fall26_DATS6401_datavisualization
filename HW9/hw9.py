"""HW9 — US Supercharger Geographic Explorer.

Streamlit application. Reads prepared data only; never calls Census API.
Run: python -m streamlit run hw9.py

Prepared files (same folder as this script):
    states_final.geojson
    chargers_processed.geojson
Optional for county income drill-down:
    counties_merged.geojson
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="US Supercharger Explorer", page_icon="🗺️", layout="wide")
BASE_DIR = Path(__file__).resolve().parent

# Five sequential, high-contrast class colors (low to high).
PALETTE = ["#e4f1f8", "#b9d9e8", "#77b6d5", "#378cbb", "#145687"]
NO_DATA_COLOR = "#b7b9bd"


# Geographic reference points only (not Supercharger locations).
# Curated major-city coordinates in WGS84; labels do not encode a data variable.
# National view is intentionally limited to avoid crowding.
CITY_REFERENCE = [
    ("Seattle", "Washington", 47.6062, -122.3321),
    ("Spokane", "Washington", 47.6588, -117.4260),
    ("Tacoma", "Washington", 47.2529, -122.4443),
    ("Portland", "Oregon", 45.5152, -122.6784),
    ("Eugene", "Oregon", 44.0521, -123.0868),
    ("San Francisco", "California", 37.7749, -122.4194),
    ("Los Angeles", "California", 34.0522, -118.2437),
    ("San Diego", "California", 32.7157, -117.1611),
    ("Sacramento", "California", 38.5816, -121.4944),
    ("San Jose", "California", 37.3382, -121.8863),
    ("Fresno", "California", 36.7378, -119.7871),
    ("Las Vegas", "Nevada", 36.1699, -115.1398),
    ("Reno", "Nevada", 39.5296, -119.8138),
    ("Phoenix", "Arizona", 33.4484, -112.0740),
    ("Tucson", "Arizona", 32.2226, -110.9747),
    ("Salt Lake City", "Utah", 40.7608, -111.8910),
    ("Provo", "Utah", 40.2338, -111.6585),
    ("Boise", "Idaho", 43.6150, -116.2023),
    ("Denver", "Colorado", 39.7392, -104.9903),
    ("Colorado Springs", "Colorado", 38.8339, -104.8214),
    ("Albuquerque", "New Mexico", 35.0844, -106.6504),
    ("Santa Fe", "New Mexico", 35.6870, -105.9378),
    ("Dallas", "Texas", 32.7767, -96.7970),
    ("Houston", "Texas", 29.7604, -95.3698),
    ("Austin", "Texas", 30.2672, -97.7431),
    ("San Antonio", "Texas", 29.4241, -98.4936),
    ("Fort Worth", "Texas", 32.7555, -97.3308),
    ("El Paso", "Texas", 31.7619, -106.4850),
    ("Oklahoma City", "Oklahoma", 35.4676, -97.5164),
    ("Tulsa", "Oklahoma", 36.1540, -95.9928),
    ("Wichita", "Kansas", 37.6872, -97.3301),
    ("Kansas City", "Missouri", 39.0997, -94.5786),
    ("St. Louis", "Missouri", 38.6270, -90.1994),
    ("Omaha", "Nebraska", 41.2565, -95.9345),
    ("Lincoln", "Nebraska", 40.8136, -96.7026),
    ("Des Moines", "Iowa", 41.5868, -93.6250),
    ("Minneapolis", "Minnesota", 44.9778, -93.2650),
    ("St. Paul", "Minnesota", 44.9537, -93.0900),
    ("Fargo", "North Dakota", 46.8772, -96.7898),
    ("Sioux Falls", "South Dakota", 43.5460, -96.7313),
    ("Billings", "Montana", 45.7833, -108.5007),
    ("Cheyenne", "Wyoming", 41.1400, -104.8202),
    ("Chicago", "Illinois", 41.8781, -87.6298),
    ("Springfield", "Illinois", 39.7817, -89.6501),
    ("Milwaukee", "Wisconsin", 43.0389, -87.9065),
    ("Madison", "Wisconsin", 43.0731, -89.4012),
    ("Detroit", "Michigan", 42.3314, -83.0458),
    ("Grand Rapids", "Michigan", 42.9634, -85.6681),
    ("Indianapolis", "Indiana", 39.7684, -86.1581),
    ("Columbus", "Ohio", 39.9612, -82.9988),
    ("Cleveland", "Ohio", 41.4993, -81.6944),
    ("Cincinnati", "Ohio", 39.1031, -84.5120),
    ("Louisville", "Kentucky", 38.2527, -85.7585),
    ("Lexington", "Kentucky", 38.0406, -84.5037),
    ("Nashville", "Tennessee", 36.1627, -86.7816),
    ("Memphis", "Tennessee", 35.1495, -90.0490),
    ("Knoxville", "Tennessee", 35.9606, -83.9207),
    ("Birmingham", "Alabama", 33.5186, -86.8104),
    ("Montgomery", "Alabama", 32.3792, -86.3077),
    ("Jackson", "Mississippi", 32.2988, -90.1848),
    ("Little Rock", "Arkansas", 34.7465, -92.2896),
    ("New Orleans", "Louisiana", 29.9511, -90.0715),
    ("Baton Rouge", "Louisiana", 30.4515, -91.1871),
    ("Atlanta", "Georgia", 33.7490, -84.3880),
    ("Savannah", "Georgia", 32.0809, -81.0912),
    ("Miami", "Florida", 25.7617, -80.1918),
    ("Orlando", "Florida", 28.5383, -81.3792),
    ("Tampa", "Florida", 27.9506, -82.4572),
    ("Jacksonville", "Florida", 30.3322, -81.6557),
    ("Tallahassee", "Florida", 30.4383, -84.2807),
    ("Charlotte", "North Carolina", 35.2271, -80.8431),
    ("Raleigh", "North Carolina", 35.7796, -78.6382),
    ("Greensboro", "North Carolina", 36.0726, -79.7920),
    ("Charleston", "South Carolina", 32.7765, -79.9311),
    ("Columbia", "South Carolina", 34.0007, -81.0348),
    ("Richmond", "Virginia", 37.5407, -77.4360),
    ("Virginia Beach", "Virginia", 36.8529, -75.9780),
    ("Washington, DC", "District of Columbia", 38.9072, -77.0369),
    ("Baltimore", "Maryland", 39.2904, -76.6122),
    ("Philadelphia", "Pennsylvania", 39.9526, -75.1652),
    ("Pittsburgh", "Pennsylvania", 40.4406, -79.9959),
    ("New York City", "New York", 40.7128, -74.0060),
    ("Buffalo", "New York", 42.8864, -78.8784),
    ("Albany", "New York", 42.6526, -73.7562),
    ("Newark", "New Jersey", 40.7357, -74.1724),
    ("Trenton", "New Jersey", 40.2206, -74.7597),
    ("Wilmington", "Delaware", 39.7447, -75.5484),
    ("Boston", "Massachusetts", 42.3601, -71.0589),
    ("Worcester", "Massachusetts", 42.2626, -71.8023),
    ("Providence", "Rhode Island", 41.8240, -71.4128),
    ("Hartford", "Connecticut", 41.7658, -72.6734),
    ("New Haven", "Connecticut", 41.3083, -72.9279),
    ("Portland, ME", "Maine", 43.6591, -70.2568),
    ("Augusta", "Maine", 44.3106, -69.7795),
    ("Manchester", "New Hampshire", 42.9956, -71.4548),
    ("Concord", "New Hampshire", 43.2081, -71.5376),
    ("Burlington", "Vermont", 44.4759, -73.2121),
    ("Montpelier", "Vermont", 44.2601, -72.5754),
    ("Charleston, WV", "West Virginia", 38.3498, -81.6326),
    ("Anchorage", "Alaska", 61.2181, -149.9003),
    ("Fairbanks", "Alaska", 64.8378, -147.7164),
    ("Juneau", "Alaska", 58.3019, -134.4197),
    ("Honolulu", "Hawaii", 21.3099, -157.8581),
    ("Hilo", "Hawaii", 19.7297, -155.0900),
]
CITY_DF = pd.DataFrame(CITY_REFERENCE, columns=["city", "state", "lat", "lon"])
NATIONAL_CITY_LABELS = [
    "Seattle", "San Francisco", "Los Angeles", "Phoenix", "Denver",
    "Dallas", "Houston", "Minneapolis", "Chicago", "Detroit",
    "Atlanta", "Miami", "Washington, DC", "New York City", "Boston",
]


def cities_for_map(selected_state):
    if selected_state == "All states":
        return CITY_DF[CITY_DF["city"].isin(NATIONAL_CITY_LABELS)]
    return CITY_DF[CITY_DF["state"] == selected_state].head(10)


@st.cache_data(show_spinner="Loading prepared geographic data…")
def load_data():
    required = ["states_final.geojson", "chargers_processed.geojson"]
    missing = [name for name in required if not (BASE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared data: " + ", ".join(missing)
            + ". Run prepare_data.py locally and commit its output to HW9/."
        )
    states = gpd.read_file(BASE_DIR / required[0]).to_crs("EPSG:4326")
    chargers = gpd.read_file(BASE_DIR / required[1]).to_crs("EPSG:4326")

    states["GEOID"] = states["GEOID"].astype(str).str.zfill(2)
    chargers["GEOID"] = chargers["GEOID"].astype("string").str.zfill(2)
    for column in ["pop_total", "median_income", "land_sqmi"]:
        if column in states:
            states[column] = pd.to_numeric(states[column], errors="coerce")
    for column in ["lat", "lon", "kw", "stalls"]:
        chargers[column] = pd.to_numeric(chargers[column], errors="coerce")
    chargers["opened_year"] = pd.to_datetime(chargers["opened"], errors="coerce").dt.year
    return states, chargers


@st.cache_data(show_spinner=False)
def load_counties():
    path = BASE_DIR / "counties_merged.geojson"
    if not path.exists():
        return None
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    gdf["median_income"] = pd.to_numeric(gdf["median_income"], errors="coerce")
    return gdf


def make_quantile_classes(values, desired_classes=5):
    """Assign class integers 0..k-1 and return human-readable bin ranges.

    Quantile breaks can coincide when data contain ties; report effective k.
    """
    series = pd.to_numeric(values, errors="coerce")
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index), []
    edges = np.unique(np.quantile(valid, np.linspace(0, 1, desired_classes + 1)))
    if len(edges) == 1:
        result = pd.Series(np.nan, index=series.index)
        result.loc[valid.index] = 0
        return result, [(float(edges[0]), float(edges[0]))]
    classes = pd.Series(np.nan, index=series.index)
    classes.loc[valid.index] = np.searchsorted(edges[1:-1], valid.to_numpy(), side="left")
    bounds = [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:])]
    return classes, bounds


def stepped_scale(colors):
    n = len(colors)
    result = []
    for i, color in enumerate(colors):
        result += [[i / n, color], [(i + 1) / n, color]]
    return result


def map_view(states, points, metric, display_points=True, selected_state="All states", show_city_labels=True):
    current = states.copy()
    class_ids, bounds = make_quantile_classes(current[metric])
    current["class_id"] = class_ids
    k = len(bounds)
    colors = PALETTE if k == 5 else [PALETTE[round(i * 4 / max(k - 1, 1))] for i in range(k)]
    fig = go.Figure()

    # Separate neutral fill for missing polygons: not confused with zero counts.
    missing = current[current["class_id"].isna()]
    if len(missing):
        fig.add_trace(go.Choropleth(
            geojson=current.__geo_interface__, featureidkey="properties.GEOID",
            locations=missing["GEOID"], z=[0] * len(missing),
            colorscale=[[0, NO_DATA_COLOR], [1, NO_DATA_COLOR]], showscale=False,
            text=missing["NAME"], marker_line_width=.65, marker_line_color="white",
            name="No data", hovertemplate="%{text}<br>No comparable data<extra></extra>",
        ))

    valid = current.dropna(subset=["class_id"])
    if len(valid):
        labels = [f"{lo:,.2f} – {hi:,.2f}" for lo, hi in bounds]
        custom = np.column_stack([
            valid["NAME"].astype(str).to_numpy(),
            valid["charger_count_view"].astype(int).to_numpy(),
            valid["pop_total"].to_numpy(),
            valid[metric].to_numpy(),
        ])
        fig.add_trace(go.Choropleth(
            geojson=current.__geo_interface__, featureidkey="properties.GEOID",
            locations=valid["GEOID"], z=valid["class_id"],
            zmin=-.5, zmax=k - .5,
            colorscale=stepped_scale(colors),
            marker_line_color="white", marker_line_width=.65,
            customdata=custom,
            colorbar=dict(title="Class", tickvals=list(range(k)),
                          ticktext=[str(i + 1) for i in range(k)], len=.72, thickness=15),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Filtered locations: %{customdata[1]}<br>"
                "Population: %{customdata[2]:,.0f}<br>"
                "Mapped measure: %{customdata[3]:,.2f}<extra></extra>"
            ),
            name="State-level measure",
        ))

    if display_points and len(points):
        # Graduated circles: logarithmically compressed for readability.
        stalls = points["stalls"].clip(lower=1).fillna(1)
        sizes = 4 + np.sqrt(stalls) * 1.8
        fig.add_trace(go.Scattergeo(
            lon=points["lon"], lat=points["lat"], mode="markers",
            marker=dict(size=sizes, color="#e76a3b", opacity=.67,
                        line=dict(color="#ffffff", width=.35)),
            customdata=np.column_stack([
                points["name"].fillna("Unknown").astype(str),
                points["city"].fillna("Unknown").astype(str),
                points["stalls"].fillna(0),
                points["kw"].fillna(0),
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                "Stalls: %{customdata[2]:.0f}<br>"
                "Max power: %{customdata[3]:.0f} kW<extra></extra>"
            ),
            name="Supercharger locations",
        ))

    # Geographic reference: major cities in a separate, non-thematic label layer.
    # Add last so labels remain visible above station markers.
    if show_city_labels:
        cities = cities_for_map(selected_state)
        if not cities.empty:
            fig.add_trace(go.Scattergeo(
                lon=cities["lon"], lat=cities["lat"],
                text=cities["city"], mode="markers+text",
                marker=dict(size=3.5, color="#202a35", opacity=0.95),
                textposition="top center",
                textfont=dict(size=10, color="#202a35", family="Arial"),
                hoverinfo="skip", showlegend=False,
                name="City reference labels",
            ))

    geo = dict(scope="usa", projection_type="albers usa", visible=False,
               showland=False, showlakes=False, showcountries=False)
    if selected_state != "All states":
        selected = current.loc[current["NAME"] == selected_state]
        if not selected.empty:
            # Fit only in state mode, without clipping the remaining layers by lat/lon.
            bounds_xy = selected.total_bounds
            padding_lon = max((bounds_xy[2] - bounds_xy[0]) * .12, .4)
            padding_lat = max((bounds_xy[3] - bounds_xy[1]) * .12, .3)
            # US Albers inset handles Alaska/Hawaii, so state view uses fitbounds instead.
            geo = dict(projection_type="mercator", visible=False,
                       lonaxis_range=[bounds_xy[0]-padding_lon, bounds_xy[2]+padding_lon],
                       lataxis_range=[bounds_xy[1]-padding_lat, bounds_xy[3]+padding_lat])

    fig.update_geos(**geo)
    fig.update_layout(
        height=590, margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=-.08, x=0),
        coloraxis_showscale=True, dragmode="pan",
    )
    return fig, bounds


def selection_table(data, metric):
    columns = ["NAME", "charger_count_view", "pop_total", "chargers_per_100k_view", "median_income"]
    clean = data[columns].rename(columns={
        "NAME": "State", "charger_count_view": "Locations",
        "pop_total": "Population", "chargers_per_100k_view": "Locations per 100k",
        "median_income": "Median household income ($)",
    })
    return clean.sort_values("Locations per 100k", ascending=False)


# -------------------- Application --------------------
st.title("US Supercharger Geographic Explorer")
st.caption("2024 ACS household income, state-level population, and US Supercharger locations")

try:
    states_raw, chargers_raw = load_data()
except Exception as exc:
    st.error("The prepared data files could not be loaded.")
    st.exception(exc)
    st.stop()

st.sidebar.header("Explore")
state_options = ["All states"] + sorted(states_raw["NAME"].astype(str).unique().tolist())
selected_state = st.sidebar.selectbox("Geographic scope", state_options)
metric_options = {
    "Locations per 100,000 residents": "chargers_per_100k_view",
    "Supercharger location count": "charger_count_view",
    "Median household income (USD)": "median_income",
}
metric_label = st.sidebar.selectbox("Polygon color", list(metric_options))
metric = metric_options[metric_label]

power_values = sorted(chargers_raw["kw"].dropna().unique().tolist())
selected_power = st.sidebar.multiselect(
    "Maximum charging power (kW)", power_values,
    default=power_values, help="A shared filter: both map and regional analysis update."
)
show_points = st.sidebar.checkbox("Show Supercharger point layer", value=True)
show_city_labels = st.sidebar.checkbox("Show major city labels", value=True)

# When all power bands are selected, retain stations with an unknown kW value.
# Selecting a subset explicitly excludes unknown-power stations.
if len(selected_power) == len(power_values):
    points = chargers_raw.copy()
else:
    points = chargers_raw[chargers_raw["kw"].isin(selected_power)].copy()
if selected_state != "All states":
    state_id = states_raw.loc[states_raw["NAME"] == selected_state, "GEOID"].iloc[0]
    points = points[points["GEOID"] == state_id].copy()

counts = points.dropna(subset=["GEOID"]).groupby("GEOID").size()
states_view = states_raw.copy()
states_view["charger_count_view"] = states_view["GEOID"].map(counts).fillna(0).astype(int)
states_view["chargers_per_100k_view"] = (
    states_view["charger_count_view"]
    / states_view["pop_total"].replace(0, np.nan)
    * 100_000
)

# Show selected state's polygon on the map, but keep all state rows for comparison.
map_states = (states_view if selected_state == "All states"
              else states_view[states_view["NAME"] == selected_state].copy())

num_unmatched = int(chargers_raw["GEOID"].isna().sum())
num_matched = int(chargers_raw["GEOID"].notna().sum())
num_missing_pop = int(states_raw["pop_total"].isna().sum())

one, two, three, four = st.columns(4)
one.metric("Visible locations", f"{len(points):,}")
two.metric("Mapped locations", f"{int(states_view['charger_count_view'].sum()):,}")
three.metric("Original point matches", f"{num_matched:,} / {len(chargers_raw):,}")
four.metric("Unmatched points", f"{num_unmatched:,}")

tab_map, tab_analysis = st.tabs(["Geographic map", "Regional analysis & validation"])

with tab_map:
    st.subheader("Layered thematic map")
    fig, bounds = map_view(map_states, points, metric, show_points, selected_state, show_city_labels)
    st.plotly_chart(fig, width="stretch", theme=None,
                    config={"displaylogo": False, "scrollZoom": True})

    class_count = len(bounds)
    with st.expander("Map caption, encoding and method", expanded=True):
        st.markdown(
            f"**Polygon layer:** {metric_label}. "
            f"**Classification:** quantile (equal-frequency), requested 5 classes; "
            f"{class_count} effective classes after merging duplicate breakpoints. "
            "Numbers 1–5 in the colorbar represent low to high classes. "
            "**Projection:** Albers USA for national view; Mercator for a single-state zoom. "
            f"**Point layer:** {len(points):,} Supercharger locations with valid "
            "coordinates, sized by sqrt(stalls), orange markers; "
            "no heatmap or clustering applied. "
            f"**City labels:** {'major-city reference points (not a thematic layer)' if show_city_labels else 'hidden'}; "
            "15 cities nationally, up to 10 per selected state; no automatic extra labels when zooming. "
            "**Basemap:** none. **Excluded from region counts:** points with no spatial match."
        )
        if bounds:
            st.write("Class boundaries for current view:")
            for i, (lo, hi) in enumerate(bounds, 1):
                st.caption(f"Class {i}: {lo:,.2f} to {hi:,.2f}")
        st.caption("Population-normalized rates are locations / population × 100,000. "
                   "Population is not filtered when charging power is filtered. "
                   "When all power options are selected, unknown-power stations are retained; "
                   "when only a subset is selected, unknown-power stations are excluded. "
                   "Charger point dataset reflects the downloaded snapshot; no date coverage is assumed.")

    if selected_state != "All states":
        st.info("Geographic scope is controlled from the sidebar. Select 'All states' to return to the national map.")

with tab_analysis:
    st.subheader("State-level comparison")
    st.caption("All charts use the same charging-power and geographic-scope filters as the map.")
    comparison = map_states.sort_values(metric, ascending=False).copy()
    if selected_state == "All states":
        comparison = comparison.head(20)

    plot = px.bar(
        comparison.sort_values(metric), y="NAME", x=metric,
        orientation="h", color=metric, color_continuous_scale=PALETTE,
        labels={"NAME": "State", metric: metric_label},
        custom_data=["GEOID", "charger_count_view", "pop_total"],
        title=("Top 20 states" if selected_state == "All states" else selected_state)
    )
    plot.update_layout(height=max(330, len(comparison) * 29 + 130),
                       margin=dict(l=10, r=10, t=45, b=20), coloraxis_showscale=False)
    # Bonus: selecting a bar updates the details below this chart.
    event = st.plotly_chart(plot, width="stretch", key="region_bar_selection",
                            on_select="rerun", selection_mode="points")
    selected_geoids = []
    for selected in event.selection.points:
        custom = selected.get("customdata", [])
        if len(custom):
            selected_geoids.append(str(custom[0]).zfill(2))
    if selected_geoids:
        detail = states_view[states_view["GEOID"].isin(selected_geoids)]
        st.markdown("**Selected regions — coordinated detail view**")
        st.dataframe(selection_table(detail, metric), width="stretch", hide_index=True)
    else:
        st.caption("Select a bar to show that state's details in the linked table below.")

    st.subheader("Distribution of state-level rates")
    histogram = px.histogram(
        map_states.dropna(subset=["chargers_per_100k_view"]),
        x="chargers_per_100k_view", nbins=14,
        labels={"chargers_per_100k_view": "Locations per 100,000 residents"}
    )
    histogram.update_layout(height=310, margin=dict(l=10, r=10, t=25, b=20))
    st.plotly_chart(histogram, width="stretch")
    st.caption("Histogram: count of states in each rate bin (not individual stations).")

    st.subheader("Join quality & missing values")
    j1, j2, j3 = st.columns(3)
    j1.metric("Matched points (original)", f"{num_matched:,}")
    j2.metric("Unmatched points (original)", f"{num_unmatched:,}")
    j3.metric("States lacking population", f"{num_missing_pop:,}")
    st.write(
        "**Spatial join:** station coordinates (EPSG:4326) → 2010 state polygons "
        "using `predicate='within'`. Original 2,264 station points produced 2,262 "
        "matched records and 2 unmatched coastal points (Alameda, CA; Honolulu, HI). "
        "Both were near the respective state polygons; simplified shoreline geometry "
        "is a plausible explanation, not a proven cause. Unmatched points remain in "
        "the source points and are excluded from state counts."
    )
    st.write(
        "**Attribute join:** state boundaries → external `state_stats.csv` "
        "population using two-digit, zero-padded state FIPS / GEOID. "
        f"Population missing in {num_missing_pop} of {len(states_raw)} states. "
        "Rate = spatially matched stations ÷ state population × 100,000. "
        "The ACS median household income is a separate joined attribute, not a denominator."
    )
    st.caption("Unmatched-point counts refer to the original unfiltered dataset. "
               "Current rate and bar chart respond to the sidebar power filter.")
    with st.expander("Explore complete state data"):
        st.dataframe(selection_table(map_states, metric), width="stretch", hide_index=True)

    if selected_state != "All states":
        with st.expander("Optional county-level income detail"):
            county_data = load_counties()
            if county_data is None:
                st.info("Add counties_merged.geojson to HW9/ to enable this optional view.")
            else:
                state_id = states_raw.loc[states_raw["NAME"] == selected_state, "GEOID"].iloc[0]
                county_subset = county_data[county_data["STATE"].astype(str).str.zfill(2) == state_id]
                st.dataframe(county_subset[["NAME", "GEOID", "median_income"]],
                             hide_index=True, width="stretch")
                st.caption("County income is a separate ACS attribute view, not a county-level station rate.")

st.divider()
st.caption(
    "Sources: course Supercharger point dataset and state_stats.csv; "
    "2010 Census boundaries; 2024 ACS 5-year estimates for income. "
    "See prepare_data.py for join steps and validation."
)
