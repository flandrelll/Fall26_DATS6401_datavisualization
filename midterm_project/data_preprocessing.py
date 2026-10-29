#%%
import duckdb
print(duckdb.__version__)
#%%
url = (
    "https://huggingface.co/datasets/open-index/arctic/"
    "resolve/main/data/comments/2020/12/000.parquet"
)

con = duckdb.connect()

# 先检查字段
print(con.execute(
    "DESCRIBE SELECT * FROM read_parquet(?)",
    [url]
).df())

# test
df = con.execute("""
    SELECT *
    FROM read_parquet(?)
    WHERE lower(subreddit) = 'stardewvalley'
    LIMIT 10
""", [url]).df()

print(df.head())
# %%
con = duckdb.connect()

url = (
    "https://huggingface.co/datasets/open-index/arctic/"
    "resolve/main/data/comments/2020/12/000.parquet"
)

result = con.execute("""
    SELECT
        MIN(created_at) AS first_comment,
        MAX(created_at) AS last_comment,
        COUNT(*) AS total_comments,
        COUNT(*) FILTER (
            WHERE lower(subreddit) = 'stardewvalley'
        ) AS stardew_comments
    FROM read_parquet(?)
""", [url]).df()

print(result)
# %%
#%%
con = duckdb.connect()

base_url = (
    "https://huggingface.co/datasets/open-index/arctic/"
    "resolve/main/data/comments/2020/12/"
)

# Test several shards
for i in [100, 200, 250, 300]:
    url = f"{base_url}{i:03d}.parquet"

    result = con.execute("""
        SELECT
            MIN(created_at) AS first_comment,
            MAX(created_at) AS last_comment
        FROM read_parquet(?)
    """, [url]).df()

    print(f"Shard {i:03d}")
    print(result)
#%%
for i in [255, 260, 265, 270]:
    url = f"{base_url}{i:03d}.parquet"

    result = con.execute("""
        SELECT
            MIN(created_at) AS first_comment,
            MAX(created_at) AS last_comment
        FROM read_parquet(?)
    """, [url]).df()

    print(f"Shard {i:03d}")
    print(result)

#%% 
# Step 1: Check the last comment shard

i = 382

url = f"{base_url}{i:03d}.parquet"

result = con.execute("""
    SELECT
        MIN(created_at) AS first_comment,
        MAX(created_at) AS last_comment,
        COUNT(*) AS total_comments
    FROM read_parquet(?)
""", [url]).df()

print(result)


#%% Step 2: Extract Stardew Valley comments

from pathlib import Path

output_dir = Path("stardew_data")
output_dir.mkdir(exist_ok=True)

batch_size = 10

for start in range(0, 383, batch_size):

    end = min(start + batch_size, 383)

    output_file = output_dir / f"comments_{start:03d}_{end-1:03d}.csv"

    # Skip batches that were already downloaded
    if output_file.exists():
        print(f"Skipping {start:03d}-{end-1:03d}")
        continue

    urls = [
        f"{base_url}{i:03d}.parquet"
        for i in range(start, end)
    ]

    print(f"Processing shards {start:03d}-{end-1:03d}...")

    try:
        df_batch = con.execute("""
            SELECT
                id,
                author,
                subreddit,
                body,
                score,
                created_utc,
                created_at,
                link_id,
                parent_id
            FROM read_parquet(?)
            WHERE lower(subreddit) = 'stardewvalley'
        """, [urls]).df()

        df_batch.to_csv(output_file, index=False)

        print(f"Saved {len(df_batch)} comments")

    except Exception as e:
        print(f"Error in batch {start:03d}-{end-1:03d}: {e}")
        break

#%%
import time

batch_size = 5

for start in range(90, 383, batch_size):

    end = min(start + batch_size, 383)
    output_file = output_dir / f"comments_{start:03d}_{end-1:03d}.csv"

    if output_file.exists():
        print(f"Skipping {start:03d}-{end-1:03d}")
        continue

    urls = [
        f"{base_url}{i:03d}.parquet"
        for i in range(start, end)
    ]

    for attempt in range(5):
        try:
            df_batch = con.execute("""
                SELECT
                    id, author, subreddit, body, score,
                    created_utc, created_at, link_id, parent_id
                FROM read_parquet(?)
                WHERE lower(subreddit) = 'stardewvalley'
            """, [urls]).df()

            df_batch.to_csv(output_file, index=False)

            print(f"Saved {start:03d}-{end-1:03d}: {len(df_batch)} comments")

            time.sleep(15)
            break

        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait = 120 * (attempt + 1)
                print(f"Rate limited. Waiting {wait} seconds...")
                time.sleep(wait)
            else:
                print(f"Error: {e}")
                raise
    else:
        print(f"Stopped at batch {start:03d}-{end-1:03d}")
        break

#%% Step 2: Check extracted batches

files = sorted(output_dir.glob("comments_*.csv"))

print("Saved batches:", len(files))
print("Expected batches:", 39)
#%% Step 3.1: Validate shard coverage

files = sorted(output_dir.glob("comments_[0-9]*_[0-9]*.csv"))

covered = []

for file in files:
    start, end = map(int, file.stem.split("_")[1:3])
    covered.extend(range(start, end + 1))

missing = sorted(set(range(383)) - set(covered))
overlapping = len(covered) - len(set(covered))

print("Batch files:", len(files))
print("Covered shards:", len(set(covered)))
print("Missing shards:", missing)
print("Overlapping shards:", overlapping)
#%% Step 3: Merge and validate comments

import pandas as pd

files = sorted(output_dir.glob("comments_*.csv"))

#%% Step 3.2: Merge all comments

comments = pd.concat(
    [pd.read_csv(file, low_memory=False) for file in files],
    ignore_index=True
)

print("Before deduplication:", len(comments))

# Remove duplicate comment IDs
comments = comments.drop_duplicates(subset="id")

print("After deduplication:", len(comments))

# Convert timestamps
comments["created_at"] = pd.to_datetime(
    comments["created_at"], utc=True
)

# Save merged dataset
comments.to_csv(
    output_dir / "stardew_comments_2020_12.csv",
    index=False
)

print("Date range:")
print(comments["created_at"].min())
print(comments["created_at"].max())

print("Final shape:", comments.shape)
#%% Step 4: Check submission data

posts_base_url = base_url.replace(
    "/comments/", "/submissions/"
)

post_url = f"{posts_base_url}000.parquet"

# Check columns
post_columns = con.execute("""
    DESCRIBE SELECT *
    FROM read_parquet(?)
""", [post_url]).df()

print(post_columns[["column_name", "column_type"]])

# Check first shard date range
post_info = con.execute("""
    SELECT
        MIN(created_at) AS first_post,
        MAX(created_at) AS last_post,
        COUNT(*) AS total_posts,
        COUNT(*) FILTER (
            WHERE lower(subreddit) = 'stardewvalley'
        ) AS stardew_posts
    FROM read_parquet(?)
""", [post_url]).df()

print(post_info)
#%% Step 4.1: Find all submission shards

import requests

api_url = (
    "https://huggingface.co/api/datasets/open-index/arctic/"
    "tree/main/data/submissions/2020/12"
)

response = requests.get(api_url, params={"recursive": "false", "limit": 1000})
response.raise_for_status()

items = response.json()

post_shards = sorted(
    item["path"].split("/")[-1]
    for item in items
    if item["path"].endswith(".parquet")
)

print("Total submission shards:", len(post_shards))
print("First shard:", post_shards[0])
print("Last shard:", post_shards[-1])

#%% Step 4.2: Extract Stardew Valley submissions

batch_size = 5

for start in range(0, 60, batch_size):

    end = min(start + batch_size, 60)

    output_file = output_dir / f"posts_{start:03d}_{end-1:03d}.csv"

    # Skip batches already saved
    if output_file.exists():
        print(f"Skipping {start:03d}-{end-1:03d}")
        continue

    urls = [
        f"{posts_base_url}{i:03d}.parquet"
        for i in range(start, end)
    ]

    print(f"Processing submissions {start:03d}-{end-1:03d}...")

    for attempt in range(5):
        try:
            df_batch = con.execute("""
                SELECT
                    id,
                    author,
                    subreddit,
                    title,
                    selftext,
                    score,
                    created_utc,
                    created_at,
                    num_comments,
                    url
                FROM read_parquet(?)
                WHERE lower(subreddit) = 'stardewvalley'
            """, [urls]).df()

            df_batch.to_csv(output_file, index=False)

            print(f"Saved {len(df_batch)} posts")

            time.sleep(15)
            break

        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait = 120 * (attempt + 1)
                print(f"Rate limited. Waiting {wait} seconds...")
                time.sleep(wait)
            else:
                print(f"Error: {e}")
                raise
    else:
        print(f"Stopped at {start:03d}-{end-1:03d}")
        break

#%% Step 4.3: Check submission batches

post_files = sorted(output_dir.glob("posts_[0-9]*_[0-9]*.csv"))

print("Saved submission batches:", len(post_files))
print("Expected:", 12)
# %%
#%% Step 5: Merge and validate submissions

post_files = sorted(output_dir.glob("posts_[0-9]*_[0-9]*.csv"))

# Check shard coverage
covered_posts = []

for file in post_files:
    start, end = map(int, file.stem.split("_")[1:3])
    covered_posts.extend(range(start, end + 1))

missing_posts = sorted(set(range(60)) - set(covered_posts))
overlapping_posts = len(covered_posts) - len(set(covered_posts))

print("Batch files:", len(post_files))
print("Missing shards:", missing_posts)
print("Overlapping shards:", overlapping_posts)

if missing_posts or overlapping_posts:
    raise ValueError("Submission shard coverage is incomplete.")

# Merge submissions
posts = pd.concat(
    [pd.read_csv(file, low_memory=False) for file in post_files],
    ignore_index=True
)

print("Before deduplication:", len(posts))

posts = posts.drop_duplicates(subset="id")

print("After deduplication:", len(posts))

# Convert timestamps
posts["created_at"] = pd.to_datetime(
    posts["created_at"], utc=True
)

# Save final dataset
posts.to_csv(
    output_dir / "stardew_posts_2020_12.csv",
    index=False
)

print("Date range:")
print(posts["created_at"].min())
print(posts["created_at"].max())

print("Final shape:", posts.shape)

#%% Step 6: Check parent node matching

# Construct IDs with Reddit prefixes
post_ids = set("t3_" + posts["id"].astype(str))
comment_ids = set("t1_" + comments["id"].astype(str))

# Identify parent node type
comments["parent_type"] = comments["parent_id"].str[:2]

# Check whether parent nodes exist in our datasets
comments["parent_found"] = (
    comments["parent_id"].isin(post_ids | comment_ids)
)

print("Total comments:", len(comments))

print("\nParent node types:")
print(comments["parent_type"].value_counts(dropna=False))

print("\nParent node matching:")
print(comments["parent_found"].value_counts())

print(
    "\nParent matching rate:",
    round(comments["parent_found"].mean() * 100, 2), "%"
)

print("\nMatching rate by parent type:")
print(
    comments.groupby("parent_type")["parent_found"]
    .agg(["count", "sum", "mean"])
)

#%% Step 7: Save parent matching results

comments.to_csv(
    output_dir / "stardew_comments_2020_12.csv",
    index=False
)

print("Comments:", comments.shape)
print("Posts:", posts.shape)
print("Parent matching rate:",
      round(comments["parent_found"].mean() * 100, 2), "%")

#%% Step 8: Data quality check

print("=== COMMENTS ===")
print("Missing values:")
print(comments.isna().sum())

print("\nDeleted/removed comment bodies:")
print(comments["body"].isin(["[deleted]", "[removed]"]).sum())

print("\nDeleted comment authors:")
print(comments["author"].isin(["[deleted]", "[removed]"]).sum())

print("\n=== POSTS ===")
print("Missing values:")
print(posts.isna().sum())

print("\nDeleted/removed post bodies:")
print(posts["selftext"].isin(["[deleted]", "[removed]"]).sum())

print("\nDeleted post authors:")
print(posts["author"].isin(["[deleted]", "[removed]"]).sum())

#%% Step 9: Prepare text and time series data

invalid_text = ["", "[deleted]", "[removed]"]

# Text: comments
text_comments = comments[
    comments["body"].notna()
    & ~comments["body"].str.strip().isin(invalid_text)
].copy()

# Text: posts
text_posts = posts.copy()

text_posts["selftext"] = (
    text_posts["selftext"]
    .fillna("")
    .replace(["[deleted]", "[removed]"], "")
)

text_posts["text"] = (
    text_posts["title"].fillna("")
    + " "
    + text_posts["selftext"]
).str.strip()

text_posts = text_posts[
    ~text_posts["text"].isin(invalid_text)
].copy()

# Time Series: daily activity counts
daily_posts = posts.groupby(
    posts["created_at"].dt.date
).size()

daily_comments = comments.groupby(
    comments["created_at"].dt.date
).size()

daily_activity = pd.DataFrame({
    "posts": daily_posts,
    "comments": daily_comments
}).fillna(0).astype(int)

daily_activity["total"] = (
    daily_activity["posts"] + daily_activity["comments"]
)

print("Valid text comments:", len(text_comments))
print("Valid text posts:", len(text_posts))
print("\nDaily activity:")
print(daily_activity.head())
print("\nDaily activity around release:")
print(daily_activity.loc[
    pd.to_datetime("2020-12-18").date():
    pd.to_datetime("2020-12-24").date()
])

#%% Step 10: Build user reply network data

# Map parent node IDs to authors
post_author_map = dict(
    zip("t3_" + posts["id"].astype(str), posts["author"])
)

comment_author_map = dict(
    zip("t1_" + comments["id"].astype(str), comments["author"])
)

parent_author_map = {
    **post_author_map,
    **comment_author_map
}

# Create reply edges
network_comments = comments[
    comments["parent_found"]
].copy()

network_comments["parent_author"] = (
    network_comments["parent_id"].map(parent_author_map)
)

# Remove deleted users and self-replies
invalid_authors = ["[deleted]", "[removed]", ""]

network_comments = network_comments[
    network_comments["author"].notna()
    & network_comments["parent_author"].notna()
    & ~network_comments["author"].isin(invalid_authors)
    & ~network_comments["parent_author"].isin(invalid_authors)
    & (network_comments["author"] != network_comments["parent_author"])
].copy()

print("Valid reply interactions:", len(network_comments))
print("Unique users:", pd.unique(
    pd.concat([
        network_comments["author"],
        network_comments["parent_author"]
    ])
).size)

print(network_comments[
    ["author", "parent_author", "created_at"]
].head())

#%% Step 11: Save visualization datasets

text_comments.to_csv(
    output_dir / "text_comments.csv",
    index=False
)

text_posts.to_csv(
    output_dir / "text_posts.csv",
    index=False
)

daily_activity.to_csv(
    output_dir / "daily_activity.csv",
    index_label="date"
)

network_comments.to_csv(
    output_dir / "network_replies.csv",
    index=False
)

print("Text comments:", len(text_comments))
print("Text posts:", len(text_posts))
print("Daily observations:", len(daily_activity))
print("Network interactions:", len(network_comments))

print("\nAll visualization datasets saved!")
# %%
