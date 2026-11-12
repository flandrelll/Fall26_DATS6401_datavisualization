"""
HW9 - US Geographic Data Explorer
Data preparation pipeline.

Sources
-------
2010 State and County boundaries: https://eric.clst.org/tech/usgeojson/
2024 ACS five-year income: https://api.census.gov/data/2024/acs/acs5
Course point dataset: superchargers_us.csv
Course state attributes: state_stats.csv

Run (from any working directory)
--------------------------------
python prepare_data.py                 # reuse saved ACS CSV files
python prepare_data.py --refresh-acs   # download ACS CSV files again

Census key (optional if API accepts requests without it): set CENSUS_API_KEY
in the environment; never hard-code credentials in the repository.
"""

from pathlib import Path
from io import StringIO
import argparse
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import requests


# ==================================================
# 1. Configuration
# ==================================================
BASE_DIR = Path(__file__).resolve().parent
STATES_PATH = BASE_DIR / "us_states.json"
COUNTIES_PATH = BASE_DIR / "us_countries.json"  # Original filename
COUNTIES_UTF8_PATH = BASE_DIR / "us_counties_utf8.geojson"
STATE_INCOME_PATH = BASE_DIR / "state_income.csv"
COUNTY_INCOME_PATH = BASE_DIR / "county_income.csv"
STATE_STATS_PATH = BASE_DIR / "state_stats.csv"
CHARGERS_PATH = BASE_DIR / "superchargers_us.csv"

STATES_MERGED_PATH = BASE_DIR / "states_merged.geojson"
COUNTIES_MERGED_PATH = BASE_DIR / "counties_merged.geojson"
STATES_FINAL_PATH = BASE_DIR / "states_final.geojson"
CHARGERS_PROCESSED_PATH = BASE_DIR / "chargers_processed.geojson"

BASE_URL = "https://api.census.gov/data/2024/acs/acs5"
API_KEY = os.getenv("CENSUS_API_KEY")


# ==================================================
# 2. Geographic boundaries and ACS income
# ==================================================
def load_geographic_data():
    """Read boundaries, converting the original County JSON to UTF-8."""
    print("Loading geographic data...")
    content = COUNTIES_PATH.read_bytes().decode("cp1252")
    COUNTIES_UTF8_PATH.write_text(content, encoding="utf-8")

    states = gpd.read_file(STATES_PATH)
    counties = gpd.read_file(COUNTIES_UTF8_PATH)
    print("States:", states.shape, "Counties:", counties.shape)
    return states, counties


def fetch_acs(params):
    """Download a Census ACS CSV response and parse it into a DataFrame."""
    request_params = {**params, "outputFormat": "csv"}
    if API_KEY:
        request_params["key"] = API_KEY

    response = requests.get(BASE_URL, params=request_params, timeout=60)
    response.raise_for_status()
    if not response.text.lstrip().startswith("NAME,"):
        raise ValueError("Census API did not return expected CSV: " + response.text[:300])
    return pd.read_csv(StringIO(response.text), dtype=str)


def load_income_data(refresh_acs=False):
    """Reuse previously saved ACS files unless explicitly refreshing."""
    if refresh_acs:
        print("Downloading 2024 ACS income data...")
        state_income = fetch_acs({"get": "NAME,B19013_001E", "for": "state:*"})
        county_income = fetch_acs({
            "get": "NAME,B19013_001E", "for": "county:*", "in": "state:*"
        })
        state_income.to_csv(STATE_INCOME_PATH, index=False)
        county_income.to_csv(COUNTY_INCOME_PATH, index=False)
    else:
        print("Loading existing ACS income CSV files...")
        state_income = pd.read_csv(STATE_INCOME_PATH, dtype=str)
        county_income = pd.read_csv(COUNTY_INCOME_PATH, dtype=str)

    print("State income:", state_income.shape, "County income:", county_income.shape)
    return state_income, county_income


def prepare_geoid(states, counties, state_income, county_income):
    """Make matching zero-padded FIPS keys for State and County joins."""
    states["GEOID"] = states["STATE"].astype(str).str.zfill(2)
    counties["GEOID"] = (
        counties["STATE"].astype(str).str.zfill(2)
        + counties["COUNTY"].astype(str).str.zfill(3)
    )
    state_income["GEOID"] = state_income["state"].astype(str).str.zfill(2)
    county_income["GEOID"] = (
        county_income["state"].astype(str).str.zfill(2)
        + county_income["county"].astype(str).str.zfill(3)
    )
    return states, counties, state_income, county_income


def clean_income_data(state_income, county_income):
    """Treat negative ACS special codes as missing rather than real income."""
    for df in (state_income, county_income):
        df["median_income"] = pd.to_numeric(df["B19013_001E"], errors="coerce")
        df.loc[df["median_income"] < 0, "median_income"] = np.nan
    return state_income, county_income


def merge_income(states, counties, state_income, county_income):
    """Attribute joins on standardized geographic identifiers."""
    states_merged = states.merge(
        state_income[["GEOID", "median_income"]],
        on="GEOID", how="left", validate="one_to_one"
    )
    counties_merged = counties.merge(
        county_income[["GEOID", "median_income"]],
        on="GEOID", how="left", validate="one_to_one"
    )
    print("States merged:", states_merged.shape)
    print("Counties merged:", counties_merged.shape)
    print("Missing State income:", states_merged["median_income"].isna().sum())
    print("Missing County income:", counties_merged["median_income"].isna().sum())
    print("Counties without comparable income:")
    print(counties_merged.loc[
        counties_merged["median_income"].isna(), ["GEOID", "NAME", "STATE"]
    ].to_string(index=False))
    return states_merged, counties_merged


# ==================================================
# 3. Population attribute join and Supercharger points
# ==================================================
def join_state_population(states_merged):
    """Add the denominator and land-area fields using two-digit State FIPS."""
    stats = pd.read_csv(STATE_STATS_PATH, dtype={"fips": str})
    stats["GEOID"] = stats["fips"].str.zfill(2)
    for col in ("pop_total", "land_sqmi", "pop_per_sqmi"):
        stats[col] = pd.to_numeric(stats[col], errors="coerce")

    states_with_population = states_merged.merge(
        stats[["GEOID", "pop_total", "land_sqmi", "pop_per_sqmi"]],
        on="GEOID", how="left", validate="one_to_one"
    )
    print("Population attribute join:", states_with_population.shape)
    print("Missing State population:", states_with_population["pop_total"].isna().sum())
    return states_with_population


def join_chargers_to_states(states_with_population):
    """Spatially match charger points to States and calculate per-capita rates."""
    chargers = pd.read_csv(CHARGERS_PATH)
    for col in ("lat", "lon"):
        chargers[col] = pd.to_numeric(chargers[col], errors="coerce")

    valid_coords = (
        chargers["lat"].between(-90, 90)
        & chargers["lon"].between(-180, 180)
    )
    invalid_count = int((~valid_coords).sum())
    print("Invalid/missing coordinates:", invalid_count)

    chargers_valid = chargers.loc[valid_coords].copy()
    points = gpd.GeoDataFrame(
        chargers_valid,
        geometry=gpd.points_from_xy(chargers_valid["lon"], chargers_valid["lat"]),
        crs="EPSG:4326"
    )
    states_geo = states_with_population.to_crs(points.crs)
    joined = gpd.sjoin(
        points, states_geo[["GEOID", "NAME", "geometry"]],
        how="left", predicate="within"
    )

    # Guard against unexpected multi-polygon overlaps creating duplicate matches.
    if len(joined) != len(points):
        raise ValueError("Spatial join produced duplicate point matches; inspect boundaries.")

    unmatched = joined.loc[joined["GEOID"].isna()].copy()
    print("Original points:", len(chargers))
    print("Spatial join rows:", len(joined))
    print("Unmatched valid-coordinate points:", len(unmatched))
    print("Unmatched locations:")
    print(unmatched[["name", "city", "state", "lat", "lon"]].to_string(index=False))

    counts = (
        joined.dropna(subset=["GEOID"])
        .groupby("GEOID")
        .size()
        .reset_index(name="charger_count")
    )
    states_final = states_with_population.merge(
        counts, on="GEOID", how="left", validate="one_to_one"
    )
    states_final["charger_count"] = states_final["charger_count"].fillna(0).astype(int)
    states_final["chargers_per_100k"] = (
        states_final["charger_count"]
        / states_final["pop_total"].replace(0, np.nan)
        * 100_000
    )
    print("Matched chargers:", int(states_final["charger_count"].sum()))
    print("Missing population denominators:", states_final["pop_total"].isna().sum())
    return states_final, joined, unmatched, states_geo


def diagnose_unmatched(unmatched, states_geo):
    """Find nearest state and approximate distance for unmatched coastal points.

    EPSG:5070 is useful for preliminary diagnostics but distances in Hawaii
    should not be treated as accurate ground distances.
    """
    if unmatched.empty:
        print("No unmatched valid-coordinate points to diagnose.")
        return

    points_proj = unmatched.drop(
        columns=["index_right", "NAME"], errors="ignore"
    ).to_crs("EPSG:5070")
    states_proj = states_geo[["NAME", "geometry"]].to_crs("EPSG:5070")
    nearest = gpd.sjoin_nearest(
        points_proj, states_proj,
        how="left", distance_col="distance_m"
    )
    print("Nearest state to unmatched points (approximate projected distance):")
    print(nearest[["name", "state", "NAME", "distance_m"]].to_string(index=False))


# ==================================================
# 4. Export and entry point
# ==================================================
def export_data(states_merged, counties_merged, states_final, chargers_joined):
    """Write all four processed layers used by the Streamlit application."""
    for data, path in (
        (states_merged, STATES_MERGED_PATH),
        (counties_merged, COUNTIES_MERGED_PATH),
        (states_final, STATES_FINAL_PATH),
        (chargers_joined, CHARGERS_PROCESSED_PATH),
    ):
        data.to_file(path, driver="GeoJSON")
        print("Saved:", path.name)


def main(refresh_acs=False):
    states, counties = load_geographic_data()
    state_income, county_income = load_income_data(refresh_acs=refresh_acs)
    states, counties, state_income, county_income = prepare_geoid(
        states, counties, state_income, county_income
    )
    state_income, county_income = clean_income_data(state_income, county_income)
    states_merged, counties_merged = merge_income(
        states, counties, state_income, county_income
    )
    states_with_population = join_state_population(states_merged)
    states_final, chargers_joined, unmatched, states_geo = join_chargers_to_states(
        states_with_population
    )
    diagnose_unmatched(unmatched, states_geo)
    export_data(states_merged, counties_merged, states_final, chargers_joined)
    print("\nData preparation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare HW9 geographic datasets")
    parser.add_argument(
        "--refresh-acs", action="store_true",
        help="Download fresh ACS CSV files instead of using local saved copies"
    )
    args = parser.parse_args()
    main(refresh_acs=args.refresh_acs)
