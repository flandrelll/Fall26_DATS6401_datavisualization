import duckdb

con = duckdb.connect("data/artscope.duckdb", read_only=True)

# Total images
print("Total images:")
print(con.execute("""
    SELECT COUNT(*) FROM artworks
""").fetchone()[0])

# Total styles
print("\nTotal styles:")
print(con.execute("""
    SELECT COUNT(DISTINCT style) FROM artworks
""").fetchone()[0])

# Images per style
print("\nImages per style:")
print(con.execute("""
    SELECT style, COUNT(*) AS count
    FROM artworks
    GROUP BY style
    ORDER BY count DESC
""").df())

con.close()