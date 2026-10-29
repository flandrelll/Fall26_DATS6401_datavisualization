
#%%
from pathlib import Path
from collections import Counter

import duckdb
import pandas as pd
from datasets import load_dataset

#%%
# Project settings
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "artscope.duckdb"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Download settings
IMAGES_PER_STYLE = 10
MAX_SCAN = 3000

#%%
# Load WikiArt dataset
print("Connecting to WikiArt...")

ds = load_dataset(
    "huggan/wikiart",
    split="train",
    streaming=True
)

print("\nDataset features:")
print(ds.features)

#%%
# Convert label IDs to names
def decode_label(column, value):
    feature = ds.features[column]

    if value is None:
        return "Unknown"

    if hasattr(feature, "int2str") and isinstance(value, int):
        try:
            return feature.int2str(value)
        except ValueError:
            pass

    return str(value)

#%%
# Download images
records = []
counts = Counter()

print("\nDownloading images...")

for idx, item in enumerate(ds):

    if idx >= MAX_SCAN:
        break

    style = decode_label("style", item["style"])

    if counts[style] >= IMAGES_PER_STYLE:
        continue

    artist = decode_label("artist", item["artist"])
    genre = decode_label("genre", item["genre"])

    try:
        image = item["image"].convert("RGB")

        filename = f"artwork_{idx:06d}.jpg"
        image_path = IMAGE_DIR / filename

        image.save(image_path, quality=90)

        # Store relative path in DuckDB
        relative_path = image_path.relative_to(ROOT).as_posix()

        records.append({
            "artwork_id": idx,
            "image_path": relative_path,
            "artist": artist,
            "genre": genre,
            "style": style,
            "width": image.width,
            "height": image.height
        })

        counts[style] += 1

    except Exception as e:
        print(f"Skipping image {idx}: {e}")
        continue

    if (idx + 1) % 100 == 0:
        print(f"Scanned {idx + 1} records")

#%%
# Convert to DataFrame
df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("No images were downloaded.")

print("\nDownloaded:", len(df), "images")

#%%
# Save metadata into DuckDB
con = duckdb.connect(str(DB_PATH))

con.register("artwork_df", df)

con.execute("""
    CREATE OR REPLACE TABLE artworks AS
    SELECT
        artwork_id,
        image_path,
        artist,
        genre,
        style,
        width,
        height
    FROM artwork_df
""")

#%%
# Check database
print("\nArtworks by Style:")

result = con.execute("""
    SELECT
        style,
        COUNT(*) AS image_count
    FROM artworks
    GROUP BY style
    ORDER BY image_count DESC, style
""").df()

print(result.to_string(index=False))

print("\nDatabase preview:")
print(con.execute("""
    SELECT *
    FROM artworks
    LIMIT 5
""").df())

con.close()

print("\nDatabase saved to:", DB_PATH)
print("Images saved to:", IMAGE_DIR)
print("Download completed!")
