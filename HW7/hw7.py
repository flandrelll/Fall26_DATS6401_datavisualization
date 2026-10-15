#%%
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, PCA

st.title("Steam Game Review Explorer")


#%%
@st.cache_data
def load_data():
    df = pd.read_csv("gamereview.csv")

    df = df.dropna(subset=["review"]).copy()
    df["review"] = df["review"].str.strip()
    df = df[df["review"] != ""]
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)

    return df


df = load_data()
docs = df["review"].tolist()

#%%
st.header("About the Dataset")

st.write(
    f"""
    This dataset contains {len(docs)} Steam game reviews
    covering {df["appid"].nunique()} games.
    Each review is treated as one document for text analysis.
    """
)
st.write(
    """
    Source: [Steam Review Aspect Dataset](
    https://github.com/ilos-vigil/steam-review-aspect-dataset
    ).
    This app uses the training split only.
    Findings describe this sample and may not represent all Steam reviews.
    """
)

#%%
st.header("Word Frequency")

cv = CountVectorizer(stop_words="english")
counts = cv.fit_transform(docs)

freq = pd.Series(
    counts.sum(axis=0).A1,
    index=cv.get_feature_names_out()
)

top_words = freq.nlargest(20).sort_values()

fig, ax = plt.subplots(figsize=(10, 6))

top_words.plot.barh(ax=ax, color="#2E6E8E")

ax.set_title("Top 20 Words in Steam Game Reviews")
ax.set_xlabel("Total Occurrences Across Reviews")
ax.set_ylabel("Word")
ax.set_xlim(left=0)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.write(
    """
    This chart shows the 20 most frequent words after lowercasing,
    tokenization, and English stop-word removal.
    Each occurrence is counted, including repeated words within one review.
    """
)

st.write(
    """
    What it reveals: Common vocabulary in this review sample.
    """)
st.write("""
    How it could mislead: Longer reviews contribute more words,
    and frequent words do not necessarily indicate positive opinions
    or greater importance.
    """
)

#%%
st.header("Topic Model")

tfv = TfidfVectorizer(
    stop_words="english",
    min_df=3,
    max_df=0.9,
    max_features=5000
)
T = tfv.fit_transform(docs)

nmf = NMF(
    n_components=5,
    init="nndsvda",
    solver="mu",
    random_state=42,
    max_iter=500
)

doc_topics = nmf.fit_transform(T)
words = tfv.get_feature_names_out()

selected_topic = st.selectbox(
    "Select a topic",
    options=range(5),
    format_func=lambda i: f"Topic {i + 1}"
)

topic_weights = pd.Series(
    nmf.components_[selected_topic],
    index=words
)

top_topic_words = topic_weights.nlargest(10).sort_values()

fig, ax = plt.subplots(figsize=(10, 5))
top_topic_words.plot.barh(ax=ax, color="#2E6E8E")

ax.set_title(f"Top 10 Words in Topic {selected_topic + 1}")
ax.set_xlabel("NMF Word Weight (not probability)")
ax.set_ylabel("Word")
ax.set_xlim(left=0)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Example Reviews")

top_review_indices = doc_topics[:, selected_topic].argsort()[-3:][::-1]

for i in top_review_indices:
    with st.expander(f"Review {i + 1}"):
        st.write(docs[i])

st.write(
    """
    What it reveals: NMF identifies groups of words that occur together,
    suggesting recurring themes. Example reviews help interpret each topic.
    How it could mislead: Five topics is an exploratory choice, but we don't know how many topics are
    actually there. Topics may overlap or reflect game names rather than broader themes.
    """
)

#%%
st.header("Embedding Projection")

pca = PCA(n_components=2, svd_solver="full")
pcs = pca.fit_transform(T.toarray())

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    pcs[:, 0],
    pcs[:, 1],
    s=25,
    alpha=0.5,
    color="#2E6E8E"
)

ax.set_title("Steam Reviews: TF–IDF → PCA")
ax.set_xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]:.1%} of variance)"
)
ax.set_ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]:.1%} of variance)"
)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.write(
    """
    Each point represents one review. Reviews are represented by
    TF–IDF vectors and projected to two dimensions using PCA.
    """
)

st.write(
    """
    What it reveals: Broad patterns of vocabulary similarity across reviews.
    How it could mislead: The projection lose some information, so nearby points
    may differ from the original space. TF–IDF ignores word order and does not
    reliably capture meaning, negation, or sarcasm. Visible groups should not
    be treated as proof of distinct topics.
    """
)

