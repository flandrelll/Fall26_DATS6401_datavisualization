#%%
import streamlit as st
import pandas as pd
import altair as alt

st.title("Data Visualization of Baseball Statistics")

df = pd.read_csv("BaseballHeightWeight.csv")

#%%
# brief introduction to the dataset
st.header("About this Dataset")
st.write("""This dataset contains the basic information of baseball players including their positions, heights, weights, and ages etc. The data is visualized using a few different visual channels to show the relationship between height and weight and the player's positions.""")
# %%
# preview
st.header("Data Preview")
st.write(df.head(10))

#%%
# underlying question
st.header("Underlying Question")
st.write("""The underlying question is: How does the relationship between height and weight vary across baseball positions?
The three visualizations use different visual channels to encode player position while keeping height and weight on the x- and y-axes.""")
#%%
# chart 1
st.header("Chart 1: Position as Color")

chart1 = (
    alt.Chart(df)
    .mark_circle(size=70)
    .encode(
        x=alt.X(
            "Height(inches):Q",
            title="Height (inches)",
            scale=alt.Scale(zero=False)
        ),

        y=alt.Y(
            "Weight(pounds):Q",
            title="Weight (pounds)",
            scale=alt.Scale(zero=False)
        ),

        color=alt.Color(
            "Position:N",
            title="Position"
        ),

        tooltip=[
            alt.Tooltip("Name:N", title="Player"),
            alt.Tooltip("Team:N", title="Team"),
            alt.Tooltip("Position:N", title="Position"),
            alt.Tooltip("Height(inches):Q", title="Height"),
            alt.Tooltip("Weight(pounds):Q", title="Weight")
        ]
    )
    .properties(
        title="Height vs Weight — Position Encoded by Color",
        height=500
    )
)

st.altair_chart(chart1)

st.write("""Color was used in this visualization to encode the player's position. 
Position is a categorical variable, and color allows the viewer to distinguish groups quickly while
height and weight are encoded by spatial position, which is effective for quatitative comparisions.
This chart uses the Gestalt principle of similarity, in which the same color are natually perceived as belonging to the same group.""")

#%%
# chart 2
st.header("Chart 2: Position as Shape")

chart2 = (
    alt.Chart(df)
    .mark_point(size=70)
    .encode(
        x=alt.X(
            "Height(inches):Q",
            title="Height (inches)",
            scale=alt.Scale(zero=False)
        ),

        y=alt.Y(
            "Weight(pounds):Q",
            title="Weight (pounds)",
            scale=alt.Scale(zero=False)
        ),

        shape=alt.Shape(
            "Position:N",
            title="Position"
        ),

        tooltip=[
            alt.Tooltip("Name:N", title="Player"),
            alt.Tooltip("Team:N", title="Team"),
            alt.Tooltip("Position:N", title="Position"),
            alt.Tooltip("Height(inches):Q", title="Height"),
            alt.Tooltip("Weight(pounds):Q", title="Weight")
        ]
    )
    .properties(
        title="Height vs Weight — Position Encoded by Shape",
        height=500
    )
)

st.altair_chart(chart2)
st.write("""In this visualization, shape was used to distinguish categorical groups. Comparing to color, shape is less immediately distinguishable, as some of the shapes are very similar
to each other and is hard to be distinguished at the first glance. Additionally, there are many categories with overlapping data points, which makes it even harder to distinguish the shapes. 
The chart also uses the Gestalt principle of similarity, in which the same shape are natually perceived as belonging to the same group. However, it is less effective as color in this case.
""")

#%%
# chart 3
st.header("Chart 3: Position as Facets")

chart3 = (
    alt.Chart(df)
    .mark_circle(size=50)
    .encode(
        x=alt.X(
            "Height(inches):Q",
            title="Height (inches)",
            scale=alt.Scale(zero=False)
        ),
        y=alt.Y(
            "Weight(pounds):Q",
            title="Weight (pounds)",
            scale=alt.Scale(zero=False)
        ),
        tooltip=[
            alt.Tooltip("Name:N", title="Player"),
            alt.Tooltip("Team:N", title="Team"),
            alt.Tooltip("Position:N", title="Position"),
            alt.Tooltip("Height(inches):Q", title="Height"),
            alt.Tooltip("Weight(pounds):Q", title="Weight")
        ]
    )
    .properties(
        width=180,
        height=160)
    .facet(
        facet=alt.Facet(
            "Position:N",
            title="Position"
        ),
        columns=3
    )
)

st.altair_chart(chart3)
st.write("""In this visualization, we faceting separate player positions
into different subplots, which allows each group easy to identify and avoid overlapping.
However, comparing across the groups require more visual scanning.
This chart uses the Gestalt principle of proximity, in which points that are close to each other are perceived as belonging to the same group.
""")

#%%
st.header("Comparison and Conclusion")
st.write("""Overall, the color provides best balance for this dataset, as it allow the viewers to compare the overall relationship between height and weight
while allowing player positions to be distinguished. However, even color provides fast categorical distinction, when the number of categories is large, it is still hard to distinguish the colors.
Also, relying on color alone may reduce accessibility for certain viewers.
Shape can represent categorical groups, but the larger the number of positions go, the harder it is for viewers to distinguish the shapes. The overlapping of data points also makes it harder to distinguish the shapes.
Faceting provides the clearest separation between positions, but make comparisons across groups more difficult. """)

# %%
