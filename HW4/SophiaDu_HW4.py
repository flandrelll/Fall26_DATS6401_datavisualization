#%%
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


st.title("Air Passengers Time Series Visualization")


#%%
@st.cache_data
def load_data():
    df = pd.read_csv("HW4/AirPassengers.csv")
    df["Month"] = pd.to_datetime(df["Month"])
    return df


df = load_data()


#%%
st.header("About the Dataset")

st.write(
    '''The AirPassengers dataset contains monthly totals of international airline passengers from 1949 to 1960.'''
)

st.header("Data Preview")
st.write(df.head())


#%%
st.header("Trend Plot and Rolling Average")

resolution = st.selectbox(
    "Time Resolution",
    ["Monthly", "Quarterly", "Yearly"]
)


# change time resolution
if resolution == "Monthly":
    plot_df = df.copy()

elif resolution == "Quarterly":
    plot_df = (
        df.set_index("Month")
        .resample("QE")["#Passengers"]
        .mean()
        .reset_index()
    )

else:
    plot_df = (
        df.set_index("Month")
        .resample("YE")["#Passengers"]
        .mean()
        .reset_index()
    )


# widgets
if resolution == "Monthly":
    max_window = 12
elif resolution == "Quarterly":
    max_window = 8
else:
    max_window = 5

window = st.slider(
    "Rolling Window (periods)",
    min_value=1,
    max_value=max_window,
    value=min(4, max_window)
)

band_option = st.selectbox(
    "Uncertainty Band",
    ["Show", "Hide"]
)


# calculate rolling statistics
plot_df["Rolling Mean"] = (
    plot_df["#Passengers"]
    .rolling(window=window)
    .mean()
)

plot_df["Rolling STD"] = (
    plot_df["#Passengers"]
    .rolling(window=window)
    .std()
)

plot_df["Upper"] = (
    plot_df["Rolling Mean"]
    + plot_df["Rolling STD"]
)

plot_df["Lower"] = (
    plot_df["Rolling Mean"]
    - plot_df["Rolling STD"]
)


# create figure
fig = go.Figure()


# raw data
fig.add_trace(
    go.Scatter(
        x=plot_df["Month"],
        y=plot_df["#Passengers"],
        mode="lines",
        name=f"{resolution} Passengers"
    )
)


# uncertainty band
if band_option == "Show":

    # upper boundary
    fig.add_trace(
        go.Scatter(
            x=plot_df["Month"],
            y=plot_df["Upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        )
    )

    # lower boundary + fill
    fig.add_trace(
        go.Scatter(
            x=plot_df["Month"],
            y=plot_df["Lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="±1 Rolling STD"
        )
    )


# rolling mean
fig.add_trace(
    go.Scatter(
        x=plot_df["Month"],
        y=plot_df["Rolling Mean"],
        mode="lines",
        name=f"{window}-Period Rolling Mean"
    )
)


# layout
fig.update_layout(
    title=f"{resolution} Passengers and {window}-Period Rolling Mean",
    xaxis_title="Time",
    yaxis_title="Number of Passengers"
)


# display figure
st.plotly_chart(
    fig,
    use_container_width=True
)

#%%
st.header("Seasonality")

# create seasonal data
season_df = df.copy()

season_df["Year"] = season_df["Month"].dt.year
season_df["Month Name"] = season_df["Month"].dt.strftime("%b")


# create figure
season_fig = go.Figure()

for year in season_df["Year"].unique():
    year_df = season_df[season_df["Year"] == year]

    season_fig.add_trace(
        go.Scatter(
            x=year_df["Month Name"],
            y=year_df["#Passengers"],
            mode="lines+markers",
            name=str(year)
        )
    )


season_fig.update_layout(
    title="Seasonal Passenger Patterns by Year",
    xaxis_title="Month",
    yaxis_title="Number of Passengers"
)


st.plotly_chart(
    season_fig,
    use_container_width=True
)

#%%
st.header("Temporal Honesty Note")
st.write("""
The trend plot allows the user to view the data at monthly, quarterly, 
and yearly resolutions. Quarterly and yearly values are calculated by 
averaging the original monthly observations, so higher-level resolutions 
smooth short-term variation and should not be interpreted as individual 
monthly observations.

The seasonality plot keep the original monthly resolution because 
aggregating the data to quarterly or yearly values would hide the 
month-to-month seasonal pattern. 
""")
