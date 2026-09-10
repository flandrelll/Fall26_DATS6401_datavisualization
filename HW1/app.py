#%%
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("HW1 - Data Visualization of Baseball Statistics")

df = pd.read_csv("HW1/BaseballHeightWeight.csv")

#%%
# brief introduction to the dataset
st.header("About this Dataset")
st.write("""This dataset contains the height and weight of baseball players. The data is visualized using a scatter plot to show the relationship between height and weight.""")
# %%
# preview
st.header("Data Preview")
st.write(df.head(10))

#%%
# chart
st.header("Scatter Plot of Height vs Weight")
fig, ax = plt.subplots()
ax.scatter(df['Height(inches)'], df['Weight(pounds)'], alpha=0.5)

ax.set_xlabel("Height (inches)")
ax.set_ylabel("Weight (pounds)")
ax.set_title("Scatter Plot of Height vs Weight")
st.pyplot(fig)

# chart explanation
st.header("Chart Explanation")
st.write("""The scatter plot above shows the relationship between the height and weight of baseball players. 
Each point represents a player, with their height on the x-axis and weight on the y-axis. 
The plot helps to visualize how height and weight are correlated among baseball players.""")
