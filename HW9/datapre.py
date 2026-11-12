
"""
HW9 - US Geographic Data Explorer
Data Preparation Pipeline

Data Sources
------------
1. US State and County GeoJSON (2010 Census boundaries)
   https://eric.clst.org/tech/usgeojson/

2. American Community Survey (ACS) 2024 5-Year Estimates
   https://api.census.gov/data/2024/acs/acs5

Indicator
---------
B19013_001E: Median Household Income
(2024 inflation-adjusted USD)

Workflow
--------
1. Load geographic boundary files
2. Convert County GeoJSON encoding to UTF-8
3. Fetch ACS income data through Census API
4. Save raw income CSV files
5. Construct geographic identifiers (GEOID)
6. Clean income values
7. Merge geography and income data
8. Validate merge results
9. Export processed GeoJSON files

Note
----
This script is intended to run only when preparing
or updating the data. The Streamlit application
loads the processed GeoJSON files directly.
"""

#%%
from pathlib import Path
from io import StringIO

import requests
import pandas as pd
import numpy as np
import geopandas as gpd


# ==================================================
# 1. Configuration
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

# Geographic data
STATES_PATH = BASE_DIR / "us_states.json"
COUNTIES_PATH = BASE_DIR / "us_countries.json"
COUNTIES_UTF8_PATH = BASE_DIR / "us_counties_utf8.geojson"

# ACS income data
STATE_INCOME_PATH = BASE_DIR / "state_income.csv"
COUNTY_INCOME_PATH = BASE_DIR / "county_income.csv"

# Processed output
STATES_OUTPUT = BASE_DIR / "states_merged.geojson"
COUNTIES_OUTPUT = BASE_DIR / "counties_merged.geojson"

# Census API
BASE_URL = "https://api.census.gov/data/2024/acs/acs5"

# Set your Census API key before executing the script.
# Do not commit your API key to GitHub.
import os
API_KEY = os.getenv("CENSUS_API_KEY")


# ==================================================
# 2. Load Geographic Data
# ==================================================

def load_geographic_data():
    """
    Load State and County GeoJSON files.

    The original County GeoJSON uses Windows-1252
    encoding, so it is converted to UTF-8 first.
    """

    print("Loading geographic data...")

    # Convert County file encoding
    content = COUNTIES_PATH.read_bytes().decode("cp1252")

    COUNTIES_UTF8_PATH.write_text(
        content,
        encoding="utf-8"
    )

    # Load GeoJSON
    states = gpd.read_file(STATES_PATH)
    counties = gpd.read_file(COUNTIES_UTF8_PATH)

    print("States:", states.shape)
    print("Counties:", counties.shape)

    return states, counties


# ==================================================
# 3. Fetch ACS Data
# ==================================================

def fetch_acs(params):
    """
    Fetch ACS data from the Census API in CSV format.

    Parameters
    ----------
    params : dict
        API query parameters.

    Returns
    -------
    pandas.DataFrame
        ACS response converted into a DataFrame.
    """

    request_params = {
        **params,
        "outputFormat": "csv"
    }

    if API_KEY:
        request_params["key"] = API_KEY

    response = requests.get(
        BASE_URL,
        params=request_params,
        timeout=60
    )

    response.raise_for_status()

    # Verify returned data format
    if not response.text.lstrip().startswith("NAME,"):
        raise ValueError(
            "Census API did not return expected CSV:\n"
            + response.text[:300]
        )

    return pd.read_csv(
        StringIO(response.text),
        dtype=str
    )


def download_income_data():
    """
    Download 2024 ACS median household income data
    for US States and Counties.
    """

    print("\nDownloading ACS income data...")

    # State-level income
    state_income = fetch_acs({
        "get": "NAME,B19013_001E",
        "for": "state:*"
    })

    # County-level income
    county_income = fetch_acs({
        "get": "NAME,B19013_001E",
        "for": "county:*",
        "in": "state:*"
    })

    # Save raw ACS data
    state_income.to_csv(
        STATE_INCOME_PATH,
        index=False
    )

    county_income.to_csv(
        COUNTY_INCOME_PATH,
        index=False
    )

    print("State income:", state_income.shape)
    print("County income:", county_income.shape)

    return state_income, county_income


# ==================================================
# 4. Prepare Geographic Identifiers
# ==================================================

def prepare_geoid(
    states,
    counties,
    state_income,
    county_income
):
    """
    Construct matching GEOIDs for geographic
    boundaries and ACS income data.
    """

    # State GEOID (2 digits)
    states["GEOID"] = (
        states["STATE"].astype(str).str.zfill(2)
    )

    state_income["GEOID"] = (
        state_income["state"].astype(str).str.zfill(2)
    )

    # County GEOID (5 digits)
    counties["GEOID"] = (
        counties["STATE"].astype(str).str.zfill(2)
        + counties["COUNTY"].astype(str).str.zfill(3)
    )

    county_income["GEOID"] = (
        county_income["state"].astype(str).str.zfill(2)
        + county_income["county"].astype(str).str.zfill(3)
    )

    return states, counties, state_income, county_income


# ==================================================
# 5. Clean Income Data
# ==================================================

def clean_income_data(state_income, county_income):
    """
    Convert income estimates to numeric values.

    Negative ACS special values are treated
    as missing data rather than actual income.
    """

    for df in [state_income, county_income]:

        df["median_income"] = pd.to_numeric(
            df["B19013_001E"],
            errors="coerce"
        )

        df.loc[
            df["median_income"] < 0,
            "median_income"
        ] = np.nan

    return state_income, county_income


# ==================================================
# 6. Merge Geographic and Income Data
# ==================================================

def merge_data(
    states,
    counties,
    state_income,
    county_income
):
    """
    Merge ACS median income estimates with
    geographic boundaries using GEOID.
    """

    print("\nMerging geographic and income data...")

    states_merged = states.merge(
        state_income[["GEOID", "median_income"]],
        on="GEOID",
        how="left",
        validate="one_to_one"
    )

    counties_merged = counties.merge(
        county_income[["GEOID", "median_income"]],
        on="GEOID",
        how="left",
        validate="one_to_one"
    )

    return states_merged, counties_merged


# ==================================================
# 7. Data Validation
# ==================================================

def validate_data(states_merged, counties_merged):
    """
    Check the dimensions and missing income
    values after merging.
    """

    print("\n========== DATA VALIDATION ==========")

    print("States merged:", states_merged.shape)
    print("Counties merged:", counties_merged.shape)

    print(
        "Missing State income:",
        states_merged["median_income"].isna().sum()
    )

    print(
        "Missing County income:",
        counties_merged["median_income"].isna().sum()
    )

    unmatched = counties_merged[
        counties_merged["median_income"].isna()
    ]

    print("\nCounties without comparable income:")
    print(unmatched[["GEOID", "NAME", "STATE"]])


# ==================================================
# 8. Export Processed GeoJSON
# ==================================================

def export_data(states_merged, counties_merged):
    """
    Export the processed geographic datasets
    for the Streamlit application.
    """

    print("\nExporting processed GeoJSON...")

    states_merged.to_file(
        STATES_OUTPUT,
        driver="GeoJSON"
    )

    counties_merged.to_file(
        COUNTIES_OUTPUT,
        driver="GeoJSON"
    )

    print("States saved:", STATES_OUTPUT)
    print("Counties saved:", COUNTIES_OUTPUT)


# ==================================================
# 9. Main Pipeline
# ==================================================

def main():

    # Load geographic boundaries
    states, counties = load_geographic_data()

    # Download ACS data
    state_income, county_income = download_income_data()

    # Prepare geographic identifiers
    states, counties, state_income, county_income = (
        prepare_geoid(
            states,
            counties,
            state_income,
            county_income
        )
    )

    # Clean income estimates
    state_income, county_income = clean_income_data(
        state_income,
        county_income
    )

    # Merge geographic and income datasets
    states_merged, counties_merged = merge_data(
        states,
        counties,
        state_income,
        county_income
    )

    # Validate merged results
    validate_data(states_merged, counties_merged)

    # Export final datasets
    export_data(states_merged, counties_merged)

    print("\nData preparation completed!")


if __name__ == "__main__":
    main()
