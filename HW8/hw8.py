
#%%
import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from PIL import Image
from sklearn.cluster import KMeans
from skimage.color import rgb2gray
from skimage.filters import sobel


#%%
# Page Configuration

st.set_page_config(
    page_title="ArtScope",
    page_icon="🎨",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "artscope.duckdb"


#%%
# Load Data

@st.cache_data
def load_data():

    con = duckdb.connect(
        str(DB_PATH),
        read_only=True
    )

    df = con.execute("""
        SELECT *
        FROM artworks
        ORDER BY artwork_id
    """).df()

    con.close()

    return df


@st.cache_data(show_spinner=False)
def load_image(image_path):

    path = ROOT / image_path

    with Image.open(path) as img:
        return img.convert("RGB").copy()


def resize_image(image, max_size=400):

    img = image.copy()
    img.thumbnail((max_size, max_size))

    return np.array(img)


#%%
# Dominant Color Extraction

def extract_palette(image, k=5):

    arr = resize_image(image)
    pixels = arr.reshape(-1, 3)

    rng = np.random.default_rng(42)

    sample_size = min(8000, len(pixels))

    indices = rng.choice(
        len(pixels),
        size=sample_size,
        replace=False
    )

    sample = pixels[indices]

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(sample)

    labels = model.predict(pixels)

    counts = np.bincount(
        labels,
        minlength=k
    )

    centers = np.clip(
        np.rint(model.cluster_centers_),
        0,
        255
    ).astype(int)

    percentages = counts / counts.sum() * 100

    order = np.argsort(percentages)[::-1]

    colors = []

    for i in order:

        rgb = tuple(centers[i])

        hex_code = "#{:02X}{:02X}{:02X}".format(
            *rgb
        )

        colors.append({
            "hex": hex_code,
            "rgb": rgb,
            "percentage": percentages[i]
        })

    return colors


#%%
# Interactive Color Palette

def plot_palette(colors):

    fig = go.Figure()

    for color in colors:

        fig.add_trace(go.Bar(
            x=[color["percentage"]],
            y=["Palette"],
            orientation="h",
            marker_color=color["hex"],
            hovertemplate=(
                f"HEX: {color['hex']}<br>"
                f"RGB: {color['rgb']}<br>"
                "Percentage: %{x:.1f}%"
                "<extra></extra>"
            ),
            showlegend=False
        ))

    fig.update_layout(
        barmode="stack",
        height=140,
        xaxis=dict(
            visible=False,
            range=[0, 100]
        ),
        yaxis=dict(visible=False),
        margin=dict(
            l=0, r=0, t=10, b=0
        )
    )

    return fig


#%%
# Image Feature Extraction

def calculate_features(image):

    arr = resize_image(image).astype(float) / 255

    gray = rgb2gray(arr)

    max_rgb = arr.max(axis=2)
    min_rgb = arr.min(axis=2)

    saturation = np.divide(
        max_rgb - min_rgb,
        max_rgb,
        out=np.zeros_like(max_rgb),
        where=max_rgb != 0
    )

    edges = sobel(gray)

    return {
        "brightness": gray.mean(),
        "saturation": saturation.mean(),
        "contrast": gray.std(),
        "edge_strength": edges.mean()
    }


#%%
# RGB Histogram

def plot_rgb_histogram(image):

    arr = resize_image(image)

    fig = go.Figure()

    channels = [
        ("Red", 0, "red"),
        ("Green", 1, "green"),
        ("Blue", 2, "blue")
    ]

    for name, index, color in channels:

        hist = np.histogram(
            arr[:, :, index],
            bins=256,
            range=(0, 256)
        )[0]

        fig.add_trace(go.Scatter(
            x=np.arange(256),
            y=hist,
            name=name,
            line=dict(color=color),
            hovertemplate=(
                "Intensity: %{x}<br>"
                "Pixel Count: %{y}"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        xaxis_title="Pixel Intensity",
        yaxis_title="Pixel Count",
        height=350
    )

    return fig


#%%
# Grayscale Intensity Histogram

def plot_intensity_histogram(image):

    arr = resize_image(image).astype(float) / 255

    gray = rgb2gray(arr)

    hist, bins = np.histogram(
        gray * 255,
        bins=64,
        range=(0, 255)
    )

    centers = (bins[:-1] + bins[1:]) / 2

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=centers,
        y=hist,
        hovertemplate=(
            "Intensity: %{x:.1f}<br>"
            "Pixel Count: %{y}"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        xaxis_title="Grayscale Intensity",
        yaxis_title="Pixel Count",
        height=350
    )

    return fig


#%%
# Structural Analysis

def show_structure(image):

    st.header("Image Structure Analysis")

    st.info("""
    **How to Use**

    Explore the RGB channel distributions,
    grayscale intensity, and edge features
    of the selected artwork.
    """)

    st.subheader("RGB Channel Distribution")

    st.plotly_chart(
        plot_rgb_histogram(image),
        use_container_width=True
    )

    st.subheader("Grayscale Intensity Distribution")

    st.plotly_chart(
        plot_intensity_histogram(image),
        use_container_width=True
    )

    st.subheader("Sobel Edge Feature Map")

    arr = resize_image(image).astype(float) / 255

    gray = rgb2gray(arr)
    edges = sobel(gray)

    left, right = st.columns(2)

    with left:
        st.image(
            image,
            caption="Original Artwork",
            use_container_width=True
        )

    with right:
        st.image(
            edges,
            caption="Sobel Edge Map",
            use_container_width=True,
            clamp=True
        )

    st.caption("""
    Bright regions in the Sobel map indicate
    stronger local intensity changes.
    """)

    with st.expander(
        "Methodology & Interpretation — Image Structure"
    ):

        st.markdown("""
        **RGB Histogram**

        RGB histograms show the distribution of
        red, green, and blue pixel intensities.

        **Grayscale Intensity**

        The grayscale histogram summarizes
        the brightness distribution of an image.

        **Sobel Edge Detection**

        The Sobel operator estimates local image
        intensity gradients to identify edges
        and structural patterns.

        **Purpose**

        To explore color distributions, brightness,
        contrast, and visual structures.

        **Limitations**

        Sobel identifies local intensity changes,
        not artistic meaning or human attention.
        Image compression and resolution can
        influence the resulting edge map.
        """)


#%%
# Artwork Analysis Panel
# Reused by Art Explorer and Your Artwork

def analysis_panel(image, key, metadata=None):

    st.header("Selected Artwork Analysis")

    left, right = st.columns(
        [1, 1],
        gap="large"
    )

    # Original Artwork
    with left:

        st.image(
            image,
            caption="Original Artwork",
            use_container_width=True
        )

        if metadata is not None:

            st.markdown(
                f"### Artwork #{metadata['artwork_id']}"
            )

            st.write(
                f"**Artist:** {metadata['artist']}"
            )

            st.write(
                f"**Style:** {metadata['style']}"
            )

            st.write(
                f"**Genre:** {metadata['genre']}"
            )

            st.caption(
                f"Dimensions: "
                f"{metadata['width']} × "
                f"{metadata['height']} px"
            )

        else:

            st.markdown("### Uploaded Artwork")

            st.caption(
                f"Dimensions: "
                f"{image.width} × {image.height} px"
            )

    # Color Palette
    with right:

        st.subheader("Dominant Color Palette")

        st.write("""
        Adjust the number of dominant colors.
        Hover over each color block to view
        its HEX code, RGB value, and percentage.
        """)

        k = st.slider(
            "Number of Dominant Colors",
            min_value=3,
            max_value=10,
            value=5,
            key=f"palette_{key}"
        )

        colors = extract_palette(image, k)

        st.plotly_chart(
            plot_palette(colors),
            use_container_width=True
        )

        st.caption(
            "Dominant colors ordered by pixel proportion."
        )

        st.subheader("Visual Features")

        features = calculate_features(image)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Brightness",
            f"{features['brightness']:.3f}"
        )

        c2.metric(
            "Saturation",
            f"{features['saturation']:.3f}"
        )

        c3.metric(
            "Contrast",
            f"{features['contrast']:.3f}"
        )

        with st.expander(
            "Methodology & Interpretation — Color Palette"
        ):

            st.markdown("""
            **Algorithm: K-Means Clustering**

            Pixels are represented as RGB vectors
            and grouped into K clusters.

            Each cluster center represents a
            dominant color in the image.

            The percentage indicates the
            proportion of pixels assigned
            to each cluster.

            **Purpose**

            To summarize complex color compositions
            into an interpretable color palette.

            **Limitations**

            K-Means does not consider the spatial
            arrangement of colors.

            RGB distance also does not perfectly
            represent human color perception.
            """)

    st.divider()

    show_structure(image)


#%%
# Load Database

if not DB_PATH.exists():

    st.error(
        f"Database not found: {DB_PATH}"
    )
    st.stop()

df = load_data()

if df.empty:
    st.warning("No artworks found in the database.")
    st.stop()


#%%
# Session State
# Keeps the selected artwork across tabs

artwork_ids = df["artwork_id"].astype(int).tolist()

if (
    "selected_artwork_id" not in st.session_state
    or st.session_state.selected_artwork_id not in artwork_ids
):
    st.session_state.selected_artwork_id = artwork_ids[0]


#%%
# App Header

st.title("🎨 ArtScope")

st.caption(
    "Exploring Art Movements Through Visual Analytics"
)

tab0, tab1, tab2 = st.tabs([
    "Overview",
    "Art Explorer",
    "Your Artwork"
])


#%%
# TAB 0 — Overview

with tab0:

    st.header("About the Dataset")

    st.markdown("""
    ArtScope uses a subset of the **WikiArt Dataset**,
    accessed through
    [Hugging Face](https://huggingface.co/datasets/huggan/wikiart).

    The dataset contains digital reproductions
    of artworks from different artistic movements,
    artists, and genres.

    Each artwork includes:

    - **Image:** Digital reproduction of the artwork
    - **Style:** Associated artistic movement
    - **Artist:** Creator of the artwork
    - **Genre:** Subject category
    - **Dimensions:** Image width and height

    Images are stored locally, while their
    metadata is managed using DuckDB.
    """)

    # Dataset Statistics
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Artworks",
        len(df)
    )

    c2.metric(
        "Art Styles",
        df["style"].nunique()
    )

    c3.metric(
        "Artists",
        df["artist"].nunique()
    )

    # Class Distribution
    st.subheader("Distribution of Art Styles")

    style_counts = (
        df["style"]
        .value_counts()
        .rename_axis("Style")
        .reset_index(name="Count")
    )

    fig = px.bar(
        style_counts,
        x="Count",
        y="Style",
        orientation="h",
        title="Number of Artworks by Style"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()
    # About Application
    st.header("About ArtScope")

    st.markdown("""
    **ArtScope** is an interactive visual analytics
    application designed to explore the visual
    characteristics of different art movements.

    Rather than focusing on artistic classification,
    ArtScope examines measurable image features,
    including color composition, brightness,
    saturation, contrast, and structural patterns.

    The goal is to investigate whether artworks
    from different movements exhibit distinct
    visual characteristics.
    """)

    st.subheader("What Can You Explore?")

    st.markdown("""
    **Art Explorer**

    Analyze artworks from the WikiArt collection
    through interactive color palettes, intensity
    distributions, feature maps, and style comparisons.

    **Your Artwork**

    Upload your own image to explore its dominant
    colors and structural characteristics.
    """)

    with st.expander("Dataset Notes & Limitations"):

        st.markdown("""
        The current dataset is a small,
        non-random subset of WikiArt.

        Different styles have unequal numbers
        of artworks.

        Image characteristics may also be
        influenced by resolution, compression,
        and digitization quality.

        Results should therefore be interpreted
        as exploratory findings rather than
        definitive descriptions of art movements.
        """)

    # Interactive Montage
    st.header("Explore the Collection")

    st.subheader("Interactive Art Montage")

    st.info("""
    **How to Use**

    Select one or more art styles to explore
    their artworks.

    Adjust the number of images displayed
    for each style.

    Click **Select this Artwork** to choose
    a painting for further analysis.
    """)

    styles = sorted(
        df["style"].unique().tolist()
    )

    default_styles = [
        s for s in [
            "Impressionism",
            "Baroque",
            "Cubism"
        ]
        if s in styles
    ]

    selected_styles = st.multiselect(
        "Choose Art Styles",
        options=styles,
        default=default_styles
    )

    max_images = int(
        df["style"].value_counts().max()
    )

    n_images = st.slider(
        "Images per Style",
        min_value=1,
        max_value=max_images,
        value=min(4, max_images)
    )

    # Display Montage
    if not selected_styles:

        st.warning(
            "Please select at least one art style."
        )

    else:

        for style in selected_styles:

            st.subheader(
                style.replace("_", " ")
            )

            style_df = (
                df[df["style"] == style]
                .head(n_images)
            )

            cols = st.columns(3)

            for i, (_, row) in enumerate(
                style_df.iterrows()
            ):

                with cols[i % 3]:

                    image = load_image(
                        row["image_path"]
                    )

                    st.image(
                        image,
                        use_container_width=True
                    )

                    st.caption(
                        f"Artwork #{row['artwork_id']} · "
                        f"{row['artist']}"
                    )

                    if st.button(
                        "Select this Artwork",
                        key=f"select_{row['artwork_id']}"
                    ):

                        st.session_state.selected_artwork_id = int(
                            row["artwork_id"]
                        )

                        st.rerun()

    with st.expander(
        "Methodology & Interpretation — Image Montage"
    ):

        st.markdown("""
        **Method**

        Artworks are grouped according to
        their artistic style labels.

        Images are displayed in a structured
        grid to support visual exploration.

        **Purpose**

        To introduce the diversity of the
        dataset and allow users to observe
        differences in color, composition,
        and artistic expression.

        **Limitations**

        The current dataset contains a small
        non-random sample of artworks.

        These examples may not represent the
        full diversity of each art movement.
        """)

    # Selected Artwork Information
    st.divider()

    st.subheader("Currently Selected Artwork")

    selected_row = df[
        df["artwork_id"]
        == st.session_state.selected_artwork_id
    ].iloc[0]

    info_left, info_right = st.columns(
        [1, 2],
        gap="large"
    )

    with info_left:

        st.image(
            load_image(selected_row["image_path"]),
            use_container_width=True
        )

    with info_right:

        st.markdown(
            f"### Artwork #{selected_row['artwork_id']}"
        )

        st.write(
            f"**Artist:** {selected_row['artist']}"
        )

        st.write(
            f"**Style:** {selected_row['style']}"
        )

        st.write(
            f"**Genre:** {selected_row['genre']}"
        )

        st.caption(
            f"Dimensions: "
            f"{selected_row['width']} × "
            f"{selected_row['height']} px"
        )

        st.info("""
        Open **Art Explorer** to explore the
        selected artwork's color palette,
        image structure, and visual features.
        """)

    st.divider()



#%%
# TAB 1 — Art Explorer

with tab1:

    st.header("Art Explorer")

    st.info("""
    **How to Use**

    The artwork selected in Overview
    is automatically loaded here.

    You can also select another artwork
    using the dropdown menu.

    Explore its dominant colors,
    image intensity, structural features,
    and comparisons with other art styles.
    """)

    # Choose Artwork
    selected_id = st.selectbox(
        "Choose an Artwork",
        options=artwork_ids,
        index=artwork_ids.index(
            st.session_state.selected_artwork_id
        ),
        format_func=lambda x: (
            f"Artwork #{x} — "
            f"{df.loc[df['artwork_id'] == x, 'artist'].iloc[0]}"
        )
    )

    st.session_state.selected_artwork_id = selected_id

    selected_row = df[
        df["artwork_id"] == selected_id
    ].iloc[0]

    image = load_image(
        selected_row["image_path"]
    )

    # Original Artwork + Color Palette
    analysis_panel(
        image,
        key="gallery",
        metadata=selected_row
    )

    st.divider()

    # Style Comparison
    st.header("Art Movement Comparison")

    st.info("""
    **How to Use**

    Select one or more artistic movements
    and choose a visual feature to compare.

    Each observation represents one artwork.
    """)

    compare_styles = st.multiselect(
        "Choose Styles to Compare",
        options=styles,
        default=[
            s for s in [
                "Impressionism",
                "Baroque"
            ]
            if s in styles
        ]
    )

    metric = st.selectbox(
        "Choose Visual Feature",
        [
            "brightness",
            "saturation",
            "contrast",
            "edge_strength"
        ]
    )

    # Extract features for all artworks
    @st.cache_data
    def get_all_features():

        records = []

        for _, row in df.iterrows():

            img = load_image(
                row["image_path"]
            )

            features = calculate_features(img)

            records.append({
                "artwork_id": row["artwork_id"],
                "style": row["style"],
                **features
            })

        return pd.DataFrame(records)


    feature_df = get_all_features()

    compare_df = feature_df[
        feature_df["style"].isin(compare_styles)
    ]

    if not compare_df.empty:

        fig = px.box(
            compare_df,
            x="style",
            y=metric,
            color="style",
            points="all",
            title=(
                f"{metric.replace('_', ' ').title()} "
                "by Art Movement"
            )
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Art Style",
            yaxis_title=metric.replace("_", " ").title()
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Please select at least one style."
        )

    with st.expander(
        "Methodology & Interpretation — Style Comparison"
    ):

        st.markdown("""
        **Method**

        Visual features are extracted from
        each artwork and compared across
        art movements.

        The boxplot shows the distribution
        of the selected feature for each style.

        **Features**

        - Brightness: Mean grayscale intensity
        - Saturation: Mean color saturation
        - Contrast: Standard deviation of intensity
        - Edge Strength: Mean Sobel gradient magnitude

        **Purpose**

        To explore whether different artistic
        movements exhibit distinct measurable
        visual characteristics.

        **Limitations**

        Small sample sizes, class imbalance,
        subject matter, and digitization
        may influence the results.
        """)


#%%
# TAB 2 — Your Artwork

with tab2:

    st.header("Your Artwork")

    st.info("""
    **How to Use**

    Upload a JPG or PNG image.

    ArtScope will display the original artwork
    next to its dominant color palette.

    You can then explore its RGB distribution,
    grayscale intensity, and edge feature map.
    """)

    uploaded_file = st.file_uploader(
        "Upload Your Artwork",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        analysis_panel(
            image,
            key="upload"
        )

    else:

        st.write(
            "Upload an image to begin your visual analysis."
        )
