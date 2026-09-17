#%% 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#%%
st.title("Data Visualization of mpg Dataset: Heatmap and PCA")

# Load the dataset
df = sns.load_dataset('mpg')
#df.head()

st.header("About this Dataset")
st.write("""The mpg dataset contains information about various car models, including their miles per gallon (mpg), number of cylinders, displacement, horsepower, weight, acceleration, model year, origin, and name.
The dataset is used to analyze the relationship between different car attributes and their fuel efficiency.""")


#%%
# Heatmap
st.header("Heatmap of Correlation Matrix")
st.write("""A heatmap is a graphical representation of data where individual values are represented as colors. In this case, we will create a heatmap to visualize the correlation matrix of the numeric features in the mpg dataset.
The correlation matrix shows the pairwise correlation coefficients between the numeric features, which can help us understand the relationships between them.""")   

numeric_df = df.select_dtypes(include='number')
corr_matrix = numeric_df.corr()

distance_matrix = 1 - corr_matrix
dist = squareform(distance_matrix, checks=False)
link = linkage(dist, method='average')
order = leaves_list(link)
ordered_cols = corr_matrix.columns[order]

#print(ordered_cols)

corr_sorted = corr_matrix.loc[ordered_cols, ordered_cols]


#%%
fig = px.imshow(
    corr_sorted,
    color_continuous_scale='RdBu_r',
    text_auto=".2f",
    zmin=-1,
    zmax=1
)

st.plotly_chart(fig)
st.write(""" 
According to the heatmap, we can tell that horsepower, weight, cylinders and displacement are strongly positively correlated with each other,
with the correlation range from around 0.84 to 0.95. On the other hand, mpg is strongly negatively correlated with those variables, particularly with weight(-0.83) and displacement (-0.80), which
suggests that the larger, heavier, and more powerful vihicles have lower fuel efficiency overall. Model year is positively correlated with mpg, suggesting that the newer cars tend to be more fuel efficient. 
""")
#%%
# PCA: Step 1: Standardize the data
st.header("PCA Analysis")
st.write("""Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms the data into a new coordinate system, where the greatest variance by any projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.
This allows us to reduce the number of dimensions while retaining most of the variability in the data.
As the dataset is not standardized, we will standardize the numeric features before applying PCA, so that each feature contributes equally to the analysis.""")

#df.isna().sum()


#%%
# standardize the numeric features
pca_df = df.dropna().copy()
pca_num = pca_df.select_dtypes(include='number')

scaler = StandardScaler()
pca_scaled = scaler.fit_transform(pca_num)

#pca_scaled
#pca_scaled.mean(axis=0)
#pca_scaled.std(axis=0)


#%%
# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(pca_scaled)

#pca_result.shape
#pca.explained_variance_ratio_


#%%
#pca.components_
pc1_var = pca.explained_variance_ratio_[0] * 100
pc2_var = pca.explained_variance_ratio_[1] * 100

#%%
pca_df['PC1'] = pca_result[:, 0]
pca_df['PC2'] = pca_result[:, 1]

#pca_df.head()

#%%
# Create a categorical version of cylinders
pca_df['cylinders_category'] = pca_df['cylinders'].astype(str)

# Color-by widget
color_option = st.selectbox(
    "Color PCA plot by:",
    ["Origin", "Cylinders"]
)

if color_option == "Origin":
    color_by = "origin"
else:
    color_by = "cylinders_category"

# PCA plot
fig_pca = px.scatter(
    pca_df,
    x='PC1',
    y='PC2',
    color=color_by,
    hover_name='name',
    labels={
    'PC1': f'PC1 ({pc1_var:.2f}%)',
    'PC2': f'PC2 ({pc2_var:.2f}%)',
    'cylinders_category': 'Cylinders',
    'origin': 'Origin'
}
)

st.plotly_chart(fig_pca)

st.write("""
The first two principal components, PC1 and PC2, explain about 83.95% of the total variance. PC1 explains 71.58% of the variance, while PC2 explains 12.37%.
PC1 is mainly associated with displacement, horsepower, cylinders, and weight in a positive direction, while mpg is negatively associated with PC1. This suggests that PC1 represents the overall size and power of the vehicle, with larger and more powerful vehicles having lower fuel efficiency.
PC2 is mainly associated with model year, which has a much larger loading than the other variables. Therefore, PC2 mainly represents the model year of vehicles, where higher PC2 values generally represent newer cars. This is consistent with the positive correlation between model year and mpg, suggesting that newer cars tend to be more fuel efficient.
When the PCA plot is colored by origin, American cars extend further toward the positive side of PC1, while European and Japanese cars are more concentrated on the negative side. However, there is still overlap among the three origin groups. When colored by cylinders, 4-cylinder cars are mostly concentrated on the negative side of PC1, while 6-cylinder cars are more spread out, and 8-cylinder cars are mostly on the positive side of PC1. This pattern is consistent with the strong contribution of cylinders to PC1 and with PC1 representing overall vehicle size and power.
The PCA results align with the correlation heatmap and further summarize the major patterns among the numeric features related to fuel efficiency.
""")