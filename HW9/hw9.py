
#%%
from pathlib import Path
import geopandas as gpd

# Directory of hw9.py
BASE_DIR = Path(__file__).resolve().parent

# File paths
states_path = BASE_DIR / "us_states.json"
counties_path = BASE_DIR / "us_countries.json"
counties_utf8_path = BASE_DIR / "us_counties_utf8.geojson"

# Step 1: Convert County file to UTF-8
print("Converting County encoding...", flush=True)

content = counties_path.read_bytes().decode("cp1252")
counties_utf8_path.write_text(content, encoding="utf-8")

print("Conversion successful!", flush=True)

# Step 2: Read State GeoJSON
print("Loading States...", flush=True)

states = gpd.read_file(states_path)

print("States:", states.shape, flush=True)

# Step 3: Read converted County GeoJSON
print("Loading Counties...", flush=True)

counties = gpd.read_file(counties_utf8_path)

print("Counties:", counties.shape, flush=True)

# Step 4: Preview
print("\nStates preview:")
print(states[["GEO_ID", "STATE", "NAME"]].head())

print("\nCounties preview:")
print(counties[["GEO_ID", "STATE", "COUNTY", "NAME"]].head())
