
#%%
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="US Geographic Data Explorer",
    page_icon="🗺️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

st.title("US Geographic Data Explorer")
st.caption(
    "Exploring median household income across US States "
    "and Counties | 2024 ACS 5-Year Estimates"
)


# --------------------------------------------------
# 2. Load and merge data
# --------------------------------------------------
@st.cache_data
def load_data():
    states = gpd.read_file(BASE_DIR / "us_states.json")
    counties = gpd.read_file(
        BASE_DIR / "us_counties_utf8.geojson"
    )

    state_income = pd.read_csv(
        BASE_DIR / "state_income.csv",
        dtype={"state": str}
    )

    county_income = pd.read_csv(
        BASE_DIR / "county_income.csv",
        dtype={"state": str, "county": str}
    )

    # Create matching geographic IDs
    states["GEOID"] = states["STATE"].astype(str).str.zfill(2)

    counties["GEOID"] = (
        counties["STATE"].astype(str).str.zfill(2)
        + counties["COUNTY"].astype(str).str.zfill(3)
    )

    state_income["GEOID"] = (
        state_income["state"].str.zfill(2)
    )

    county_income["GEOID"] = (
        county_income["state"].str.zfill(2)
        + county_income["county"].str.zfill(3)
    )

    # Convert income to numeric
    for df in [state_income, county_income]:
        df["median_income"] = pd.to_numeric(
            df["B19013_001E"],
            errors="coerce"
        )

        df.loc[
            df["median_income"] < 0,
            "median_income"
        ] = np.nan

    # Merge geography and income
    states = states.merge(
        state_income[["GEOID", "median_income"]],
        on="GEOID",
        how="left",
        validate="one_to_one"
    )

    counties = counties.merge(
        county_income[["GEOID", "median_income"]],
        on="GEOID",
        how="left",
        validate="one_to_one"
    )

    return states, counties


states, counties = load_data()


# --------------------------------------------------
# 3. Sidebar controls
# --------------------------------------------------
st.sidebar.header("Explore Geography")

state_names = sorted(states["NAME"].unique())

selected_state = st.sidebar.selectbox(
    "Select a State",
    ["United States"] + state_names
)

st.sidebar.divider()

st.sidebar.markdown("**Indicator**")
st.sidebar.info("Median Household Income (USD)")

st.sidebar.caption(
    "Data: 2024 ACS 5-Year Estimates\n\n"
    "Boundaries: 2010 Census GeoJSON"
)


# --------------------------------------------------
# 4. Choropleth helper
# --------------------------------------------------
def create_map(gdf, title, national=False):
    valid = gdf[gdf["median_income"].notna()].copy()
    missing = gdf[gdf["median_income"].isna()].copy()

    fig = go.Figure()

    # Income layer
    if not valid.empty:
        fig.add_trace(
            go.Choropleth(
                geojson=gdf.__geo_interface__,
                featureidkey="properties.GEOID",
                locations=valid["GEOID"],
                z=valid["median_income"],
                text=valid["NAME"],
                colorscale="YlGnBu",
                marker_line_color="white",
                marker_line_width=0.5,
                colorbar=dict(
                    title="Income ($)",
                    thickness=15
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Median Household Income: "
                    "$%{z:,.0f}<extra></extra>"
                )
            )
        )

    # Missing-data layer
    if not missing.empty:
        fig.add_trace(
            go.Choropleth(
                geojson=gdf.__geo_interface__,
                featureidkey="properties.GEOID",
                locations=missing["GEOID"],
                z=[0] * len(missing),
                text=missing["NAME"],
                colorscale=[
                    [0, "#d3d3d3"],
                    [1, "#d3d3d3"]
                ],
                showscale=False,
                marker_line_color="white",
                marker_line_width=0.5,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "No comparable 2024 data"
                    "<extra></extra>"
                )
            )
        )

    # Geographic projection
    if national:
        fig.update_geos(
            scope="usa",
            projection_type="albers usa",
            visible=False
        )
    else:
        fig.update_geos(
            projection_type="mercator",
            fitbounds="locations",
            visible=False
        )

    fig.update_layout(
        title=title,
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


# --------------------------------------------------
# 5. Geographic navigation
# --------------------------------------------------
if selected_state == "United States":

    current_data = states.copy()
    current_level = "State"

    st.subheader("United States Overview")

    st.plotly_chart(
        create_map(
            current_data,
            "Median Household Income by State",
            national=True
        ),
        width="stretch"
    )

else:

    selected_row = states[
        states["NAME"] == selected_state
    ].iloc[0]

    state_code = selected_row["STATE"]

    current_data = counties[
        counties["STATE"] == state_code
    ].copy()

    current_level = "County"

    st.subheader(f"{selected_state} — County Overview")

    st.plotly_chart(
        create_map(
            current_data,
            f"Median Household Income in {selected_state}"
        ),
        width="stretch"
    )


# --------------------------------------------------
# 6. Summary statistics
# --------------------------------------------------
valid_data = current_data.dropna(
    subset=["median_income"]
)

st.divider()
st.subheader(f"{current_level}-Level Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    f"Number of {current_level}s",
    len(current_data)
)

col2.metric(
    "Highest Median Income",
    (
        f"${valid_data['median_income'].max():,.0f}"
        if not valid_data.empty else "N/A"
    )
)

col3.metric(
    "Lowest Median Income",
    (
        f"${valid_data['median_income'].min():,.0f}"
        if not valid_data.empty else "N/A"
    )
)


# --------------------------------------------------
# 7. Regional comparison
# --------------------------------------------------
st.divider()
st.subheader("Regional Income Comparison")

if not valid_data.empty:

    top_regions = valid_data.nlargest(
        10, "median_income"
    ).sort_values("median_income")

    bar_fig = px.bar(
        top_regions,
        x="median_income",
        y="NAME",
        orientation="h",
        color="median_income",
        color_continuous_scale="YlGnBu",
        labels={
            "NAME": "Region",
            "median_income": "Median Household Income ($)"
        }
    )

    bar_fig.update_layout(
        height=420,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=20, b=20)
    )

    st.plotly_chart(bar_fig, width="stretch")

else:
    st.warning("No valid income data available.")


# --------------------------------------------------
# 8. Data transparency
# --------------------------------------------------
with st.expander("View underlying data"):
    st.dataframe(
        current_data[
            ["NAME", "GEOID", "median_income"]
        ].sort_values("NAME"),
        width="stretch"
    )

st.caption(
    "Note: Geographic boundaries are from 2010, "
    "while income estimates are from the 2024 ACS. "
    "Some county boundaries and identifiers have changed."
)
