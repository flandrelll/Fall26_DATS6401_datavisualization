
#%%
import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import streamlit.components.v1 as components
from pathlib import Path
from pyvis.network import Network
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


#%% Page Configuration
st.set_page_config(
    page_title="Stardew Valley 1.5 Community Explorer",
    layout="wide"
)

st.title("Stardew Valley 1.5: Reddit Community Explorer")

st.write("""
An interactive exploration of Reddit discussions, community activity,
and user interactions surrounding the Stardew Valley 1.5 update.

Dataset: r/StardewValley, December 2020.
Release date: December 21, 2020.
""")


#%% Load Data
# Absolute path to the data folder
DATA_DIR = Path(__file__).resolve().parent / "stardew_data"

#%% Load Data
@st.cache_data
def load_data():
    daily = pd.read_csv(
        DATA_DIR / "daily_activity.csv",
        parse_dates=["date"]
    )

    text_comments = pd.read_csv(
        DATA_DIR / "text_comments.csv",
        parse_dates=["created_at"]
    )

    text_posts = pd.read_csv(
        DATA_DIR / "text_posts.csv",
        parse_dates=["created_at"]
    )

    network_replies = pd.read_csv(
        DATA_DIR / "network_replies.csv",
        parse_dates=["created_at"]
    )

    return daily, text_comments, text_posts, network_replies


daily, text_comments, text_posts, network_replies = load_data()

# Use UTC consistently for comparisons
for df, column in [
    (text_comments, "created_at"),
    (text_posts, "created_at"),
    (network_replies, "created_at")
]:
    df[column] = pd.to_datetime(df[column], utc=True)

daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None)

RELEASE_DATE = pd.Timestamp("2020-12-21", tz="UTC")


#%% Network Helper Functions
def build_period_graph(df, n_users):
    """
    Build an undirected reply network and select
    the top N users by weighted degree.

    Self-replies and missing user identities are excluded.
    """
    graph = nx.Graph()

    for author, parent_author in zip(
        df["author"],
        df["parent_author"]
    ):
        if pd.isna(author) or pd.isna(parent_author):
            continue

        if author == parent_author:
            continue

        if graph.has_edge(author, parent_author):
            graph[author][parent_author]["weight"] += 1
        else:
            graph.add_edge(
                author,
                parent_author,
                weight=1
            )

    weighted_degree = dict(graph.degree(weight="weight"))

    top_users = sorted(
        weighted_degree,
        key=weighted_degree.get,
        reverse=True
    )[:n_users]

    return graph.subgraph(top_users).copy()


def calculate_connectivity(graph):
    """
    Calculate density, connected components,
    and largest connected component percentage.
    """
    n = graph.number_of_nodes()

    if n == 0:
        return 0, 0, 0

    connected_groups = list(nx.connected_components(graph))

    density = nx.density(graph)
    component_count = len(connected_groups)

    largest_ratio = (
        max(len(group) for group in connected_groups)
        / n * 100
    )

    return density, component_count, largest_ratio


#%% Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Time Series",
    "Text Analysis",
    "Network"
])


#%% Overview
with tab1:
    st.header("Project Overview")

    st.write("""
    This project explores how the r/StardewValley community
    discussed and interacted around the release of
    Stardew Valley 1.5, which introduced Ginger Island
    on December 21, 2020.
    """)

    # Dataset Summary
    st.subheader("Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Posts", "7,674")

    with col2:
        st.metric("Comments", "62,755")

    with col3:
        st.metric("Observation Period", "31 Days")

    with col4:
        st.metric("Parent Matching Rate", "99.06%")

    st.caption("""
    Dataset: r/StardewValley | December 1–31, 2020 (UTC)
    Source: Arctic Shift Reddit Archive
    """)

    # Research Questions
    st.subheader("Research Questions")

    st.markdown("""
    This project investigates three aspects of community activity:

    1. **Time Series:** How did posting and commenting activity
       change around the release of Stardew Valley 1.5?

    2. **Text Analysis:** What topics and vocabulary appeared
       in discussions before and after the update?

    3. **Network Analysis:** How did user participation,
   interaction patterns, and network connectivity change
   around the update? Did newly observed users contribute
   to changes in the community structure?
    """)

    # Data Collection
    st.subheader("Data Collection")

    st.write("""
    The dataset was obtained from the Arctic Shift Reddit Archive,
    using archived Reddit submissions and comments from
    r/StardewValley during December 2020.

    The source data was stored in daily Parquet shards.
    DuckDB was used to query these files and extract only records
    belonging to r/StardewValley, without downloading the
    complete Reddit archive.

    The resulting records were combined, checked for duplicate IDs,
    and exported to CSV files for reproducible analysis.
    """)

    # Data Preprocessing
    st.subheader("Data Preprocessing")

    with st.expander("View Data Cleaning and Preparation"):

        st.markdown("#### 1. Data Validation")

        st.write("""
        Submission and comment batches were checked for missing
        source shards, overlapping shard ranges, duplicate IDs,
        and timestamp coverage.

        The final dataset contains 7,674 posts and 62,755 comments,
        covering December 1–31, 2020.
        """)

        st.markdown("#### 2. Text Preparation")

        st.write("""
        Missing, deleted, and removed comment bodies were excluded
        from the text-analysis dataset.

        Post titles were retained as valid text, including cases
        where the self-text was empty or unavailable.

        The resulting text datasets contain 60,203 comments
        and 7,674 posts.
        """)

        st.markdown("#### 3. Time Series Preparation")

        st.write("""
        Submission and comment timestamps were converted to UTC
        dates and aggregated into daily activity counts.

        The daily dataset contains 31 observations, with separate
        counts for posts, comments, and total activity.
        """)

        st.markdown("#### 4. Network Preparation")

        st.write("""
        Reddit parent identifiers were used to reconstruct
        reply relationships.

        A parent_id beginning with t1 refers to another comment,
        while t3 refers to a submission.

        Parent IDs were matched to their corresponding authors
        to identify who replied to whom.

        Of 62,755 comments, 62,163 had a matching parent record,
        producing a parent matching rate of 99.06%.

        After excluding interactions without valid user identities,
        52,938 reply interactions were retained for network analysis.
        """)

    # Scope and Limitations
    st.subheader("Scope & Limitations")

    st.write("""
    This project examines one subreddit over a single month.
    It does not represent the entire Stardew Valley player community.

    The release date divides the observations into two periods:
    December 1–20 (before) and December 21–31 (after).

    Changes observed around the release are descriptive associations
    and do not establish that the update alone caused them.

    Deleted content, unavailable authors, and archive coverage
    may also affect the analysis.
    """)

    st.markdown(
        "**Data Source:** "
        "[Arctic Shift Reddit Archive]"
        "(https://huggingface.co/datasets/open-index/arctic)"
    )


#%% Time Series
with tab2:
    st.header("Community Activity Over Time")

    st.write("""
    Stardew Valley's 1.5 update was released on December 21, 2020.
    This visualization explores how activity in r/StardewValley
    changed around the release.
    """)

    # Select activity type
    activity_type = st.selectbox(
        "Select Activity Type",
        ["total", "posts", "comments"],
        format_func=lambda x: {
            "total": "Total Activity",
            "posts": "Posts",
            "comments": "Comments"
        }[x]
    )

    # Interactive line chart
    fig = px.line(
        daily,
        x="date",
        y=activity_type,
        markers=True,
        title="Daily Reddit Activity — December 2020",
        labels={
            "date": "Date",
            activity_type: "Daily Activity Count"
        }
    )

    fig.add_vline(
        x="2020-12-21",
        line_dash="dash",
        line_color="red"
    )

    fig.update_layout(hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)

    st.caption("""
    The dashed vertical line marks the release date of
    Stardew Valley 1.5 (December 21, 2020).
    Daily activity is aggregated in UTC.
    """)

    # Before vs After Comparison
    st.subheader("Before vs. After Update")

    release_date_daily = pd.Timestamp("2020-12-21")

    before = daily[daily["date"] < release_date_daily]
    after = daily[daily["date"] >= release_date_daily]

    before_avg = before[activity_type].mean()
    after_avg = after[activity_type].mean()

    percent_change = (
        (after_avg - before_avg) / before_avg * 100
        if before_avg != 0 else 0
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Before Update (Daily Avg.)",
            f"{before_avg:,.0f}"
        )

    with col2:
        st.metric(
            "After Update (Daily Avg.)",
            f"{after_avg:,.0f}"
        )

    with col3:
        st.metric(
            "Change",
            f"{percent_change:+.1f}%"
        )

    st.caption("""
    Before: December 1–20. After: December 21–31.
    The release day is included in the after period.
    All dates are in UTC.
    """)


#%% Text Analysis
with tab3:
    st.header("Text Analysis: Community Discussions")

    st.write("""
    Explore the most frequently used words in r/StardewValley
    before and after the Stardew Valley 1.5 update.
    """)

    # User controls
    col1, col2, col3 = st.columns(3)

    with col1:
        source = st.selectbox(
            "Data Source",
            ["Comments", "Posts"]
        )

    with col2:
        period = st.selectbox(
            "Time Period",
            ["Before Update", "After Update", "Entire Month"]
        )

    with col3:
        top_n = st.slider(
            "Number of Words",
            min_value=10,
            max_value=30,
            value=15,
            step=5
        )

    # Select dataset
    if source == "Comments":
        text_df = text_comments.copy()
        text_column = "body"
    else:
        text_df = text_posts.copy()
        text_column = "text"

    # Filter by release date
    if period == "Before Update":
        text_df = text_df[
            text_df["created_at"] < RELEASE_DATE
        ]

    elif period == "After Update":
        text_df = text_df[
            text_df["created_at"] >= RELEASE_DATE
        ]

    # Prepare text
    docs = text_df[text_column].dropna().astype(str)

    # Custom stop words
    custom_stop_words = {
        "just", "like", "don", "ve", "ll", "didn",
        "doesn", "isn", "wasn", "wouldn", "couldn",
        "really", "think", "know", "got", "get",
        "https", "http", "www", "com", "reddit"
    }

    stop_words = list(
        ENGLISH_STOP_WORDS.union(custom_stop_words)
    )

    # Word frequency
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        max_features=1000
    )

    counts = vectorizer.fit_transform(docs)

    word_freq = pd.DataFrame({
        "word": vectorizer.get_feature_names_out(),
        "frequency": counts.sum(axis=0).A1
    })

    # Normalize by retained tokens
    total_words = counts.sum()

    word_freq["normalized_frequency"] = (
        word_freq["frequency"] / total_words * 1000
    )

    word_freq = word_freq.nlargest(
        top_n,
        "normalized_frequency"
    )

    # Horizontal bar chart
    fig = px.bar(
        word_freq.sort_values("normalized_frequency"),
        x="normalized_frequency",
        y="word",
        orientation="h",
        title=f"Top {top_n} Words — {source} ({period})",
        labels={
            "normalized_frequency": "Occurrences per 1,000 Retained Words",
            "word": "Word"
        },
        hover_data={
            "frequency": True,
            "normalized_frequency": ":.2f"
        }
    )

    fig.update_layout(
        height=550,
        yaxis_title=None
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Based on {len(docs):,} {source.lower()} "
        f"from the selected period. "
        "English and custom stop words are excluded. "
        "Normalized frequency is measured per 1,000 retained tokens."
    )

    # Text Preprocessing Explanation
    with st.expander("Text Preprocessing & Methodology"):

        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            st.markdown("#### Stop Word Removal")

            st.write("""
            Common English stop words are removed using
            scikit-learn's built-in list.

            Additional conversational fillers, contraction
            fragments, and URL tokens are excluded to
            emphasize meaningful discussion topics.
            """)

            st.caption(
                "Examples: just, like, really, don, ve, https, www"
            )

            st.markdown("#### Domain-Specific Vocabulary")

            st.write("""
            Game-related terms such as game, update, new,
            ginger, and island are retained to capture
            discussion surrounding the 1.5 release.
            """)

        with col_right:
            st.markdown("#### Normalized Frequency")

            st.write("""
            Word counts are expressed per 1,000 retained tokens
            to account for differences in text volume between periods.
            """)

            st.latex(
                r"f_{\mathrm{norm}} = "
                r"\frac{\mathrm{Word\ Count}}{\mathrm{Retained\ Tokens}}"
                r"\times 1000"
            )

            st.markdown("#### Limitations")

            st.write("""
            Stop word selection is subjective.
            Words are analyzed individually rather than as phrases,
            and the top 1,000 terms are selected separately
            for each period.
            """)

    # Interpretation and Discussion
    with st.expander(
        "Interpretation & Discussion",
        expanded=True
    ):
        st.markdown("#### Discussion of the 1.5 Update")

        st.write("""
        The frequent appearance of update-related vocabulary
        like "new" and "update" suggests that the new version
        became an important topic of conversation within the community.

        However, specific terms such as "Ginger Island"
        were mentioned less than general words such as
        "update" or "new".

        One possible explanation is that players were still
        getting familiar with the newly introduced content
        and its terminology.

        Alternatively, discussions may have focused on other
        features, gameplay changes, or avoiding spoilers.
        """)

        st.markdown("#### Community Support and Interaction")

        st.write("""
        Words such as "thank" and "love" may indicate
        a supportive and collaborative discussion environment.

        One possible interpretation is that players frequently
        asked questions about the update, shared information,
        or helped one another understand new game mechanics.

        However, word frequency alone cannot establish
        the conversational context in which these
        expressions appeared.
        """)

        st.markdown("#### Interpretation Limitations")

        st.caption("""
        These interpretations are exploratory hypotheses
        rather than confirmed findings.

        Word frequency does not capture context, sentiment,
        or relationships between words.

        Further analysis of phrases, topics, and reply interactions
        would be needed to evaluate these explanations.
        """)


#%% Network Visualization
with tab4:
    st.header("Reddit User Interaction Network")

    st.write("""
    This network visualizes reply interactions between users
    in r/StardewValley during December 2020.

    Each node represents a Reddit user, and each edge represents
    at least one reply between two users.
    """)

    # Interactive controls
    col1, col2 = st.columns(2)

    with col1:
        network_period = st.selectbox(
            "Time Period",
            ["Before Update", "After Update", "Entire Month"],
            key="network_period"
        )

    with col2:
        top_n_users = st.slider(
            "Number of Users to Display",
            min_value=30,
            max_value=200,
            value=100,
            step=10
        )

    # Split interactions into before and after periods
    before_replies = network_replies[
        network_replies["created_at"] < RELEASE_DATE
    ].copy()

    after_replies = network_replies[
        network_replies["created_at"] >= RELEASE_DATE
    ].copy()

    # Select the network period
    if network_period == "Before Update":
        network_df = before_replies

    elif network_period == "After Update":
        network_df = after_replies

    else:
        network_df = network_replies

    # Build selected network
    # Self-replies are excluded in build_period_graph()
    H = build_period_graph(
        network_df,
        top_n_users
    )

    # Calculate network metrics
    degree = dict(H.degree())

    communities = list(
        nx.community.greedy_modularity_communities(H)
    )

    community_map = {}

    for cid, community in enumerate(communities):
        for node in community:
            community_map[node] = cid

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Displayed Users",
            H.number_of_nodes()
        )

    with col2:
        st.metric(
            "Displayed Connections",
            H.number_of_edges()
        )

    with col3:
        st.metric(
            "Communities",
            len(communities)
        )

    # Create interactive PyVis network
    net = Network(
        height="650px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        cdn_resources="in_line",
        notebook=False
    )

    # Community colors
    palette = [
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#AA3377",
        "#66CCEE",
        "#BBBBBB"
    ]

    # Add nodes
    for node in H.nodes():
        node_degree = degree[node]
        cid = community_map[node]

        net.add_node(
            node,
            label="",
            title=(
                f"User: {node}"
                f"\nDegree: {node_degree}"
                f"\nCommunity: {cid + 1}"
            ),
            size=10 + node_degree * 3,
            color=palette[cid % len(palette)]
        )

    # Add edges
    for u, v, data in H.edges(data=True):
        net.add_edge(
            u,
            v,
            value=data["weight"],
            title=f"Replies: {data['weight']}",
            color="#cccccc"
        )

    # Force-directed layout
    net.barnes_hut(
        gravity=-2500,
        central_gravity=0.3,
        spring_length=130,
        spring_strength=0.03
    )

    # Display interactive network
    components.html(
        net.generate_html(notebook=False),
        height=670
    )

    st.caption("""
    Node size represents degree (number of connections)
    within the displayed network.

    Colors indicate communities detected using greedy
    modularity optimization.

    Edge thickness represents the number of reply interactions.
    """)

    #%% Network Interpretation & Hypothesis
    st.subheader("Network Interpretation & Hypothesis")

    # Calculate comparison networks using the same rules
    G_before = build_period_graph(
        before_replies,
        top_n_users
    )

    G_after = build_period_graph(
        after_replies,
        top_n_users
    )

    before_communities = len(
        list(nx.community.greedy_modularity_communities(G_before))
    )

    after_communities = len(
        list(nx.community.greedy_modularity_communities(G_after))
    )

    before_connections = G_before.number_of_edges()
    after_connections = G_after.number_of_edges()

    # Dynamic observations
    st.write(f"""
    Among the top **{top_n_users} users** selected in each period,
    the number of detected communities changed from
    **{before_communities}** before the update to
    **{after_communities}** after the update.

    The number of connections changed from
    **{before_connections}** to **{after_connections}**.

    One possible explanation is that the update attracted
    newly active users who participated in discussions outside
    previously established interaction groups.

    Their participation may have changed the composition
    of the network and contributed to a more distributed
    community structure.

    However, an increase in detected communities does not
    necessarily indicate network fragmentation.
    """)

    st.caption("""
    The top users are selected independently for each period,
    so the two networks do not necessarily contain the same people.

    The comparison also depends on the selected Top N value.
    """)

    #%% User Overlap Analysis
    st.subheader("User Overlap Analysis")

    st.write("""
    To investigate whether newly active users changed
    the community structure, we compare users who
    participated in reply interactions before and
    after the Stardew Valley 1.5 update.
    """)

    # Identify users on both sides of interactions
    before_users = (
        set(before_replies["author"].dropna())
        | set(before_replies["parent_author"].dropna())
    )

    after_users = (
        set(after_replies["author"].dropna())
        | set(after_replies["parent_author"].dropna())
    )

    # Compare user sets
    returning_users = before_users & after_users
    newly_observed_users = after_users - before_users

    after_total = len(after_users)

    returning_ratio = (
        len(returning_users) / after_total * 100
        if after_total > 0 else 0
    )

    new_ratio = (
        len(newly_observed_users) / after_total * 100
        if after_total > 0 else 0
    )

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Returning Users",
            f"{len(returning_users):,}"
        )

    with col2:
        st.metric(
            "Newly Observed Users",
            f"{len(newly_observed_users):,}"
        )

    with col3:
        st.metric(
            "Newly Observed Ratio",
            f"{new_ratio:.1f}%"
        )

    # Visualize user composition
    overlap_df = pd.DataFrame({
        "User Group": [
            "Returning Users",
            "Newly Observed Users"
        ],
        "Users": [
            len(returning_users),
            len(newly_observed_users)
        ]
    })

    fig_overlap = px.bar(
        overlap_df,
        x="User Group",
        y="Users",
        color="User Group",
        text="Users",
        title="User Composition After the 1.5 Update"
    )

    fig_overlap.update_layout(
        showlegend=False,
        yaxis_title="Number of Users",
        xaxis_title=None
    )

    st.plotly_chart(
        fig_overlap,
        use_container_width=True
    )

    # User Overlap Interpretation
    st.write(f"""
    Among users participating in reply interactions after
    the update, **{new_ratio:.1f}%** were not observed
    during the pre-update period, while
    **{returning_ratio:.1f}%** appeared in both periods.

    These results describe changes in the observed user
    composition, but do not establish that newly observed
    users caused the network to become more fragmented.
    """)

    st.caption("""
    Newly observed users are defined relative to the
    December 1–20 observation window.

    They are not necessarily new Reddit accounts
    or new members of the Stardew Valley community.

    This analysis includes all users in the retained
    reply-interaction dataset, rather than only the
    top users displayed in the network visualization.

    The before and after observation windows differ
    in length (20 days versus 11 days).
    """)

    #%% Network Connectivity Analysis
    st.subheader("Network Connectivity Analysis")

    st.write("""
    To investigate whether the interaction network became
    more fragmented after the 1.5 update, we compare
    network density, connected components, and the
    largest connected component.

    Both periods use the same Top N setting.
    """)

    # Use the same graphs as the network interpretation
    before_density, before_components, before_largest = (
        calculate_connectivity(G_before)
    )

    after_density, after_components, after_largest = (
        calculate_connectivity(G_after)
    )

    #%% Connectivity Comparison
    st.markdown("#### Before vs. After Connectivity Comparison")

    st.write(
        "The following metrics compare the top",
        top_n_users,
        "users in each period."
    )

    # Simple Markdown table to avoid previous WebSocket issues
    comparison_table = f"""
| Metric | Before Update | After Update |
|---|---:|---:|
| Network Density | {before_density:.4f} | {after_density:.4f} |
| Connected Components | {before_components} | {after_components} |
| Largest Component (%) | {before_largest:.1f}% | {after_largest:.1f}% |
"""

    st.markdown(comparison_table)

    #%% Network Connectivity Interpretation
    st.markdown("#### Network Connectivity Interpretation")

    st.write(f"""
    The connectivity analysis reveals changes in the
    interaction network following the Stardew Valley 1.5 update.

    Network density changed from **{before_density:.4f}**
    to **{after_density:.4f}**.

    The number of connected components changed from
    **{before_components}** to **{after_components}**,
    while the largest connected component changed from
    **{before_largest:.1f}%** to **{after_largest:.1f}%**
    of displayed users.
    """)

    # Interpret changes based on actual results
    if (
        after_components > before_components
        and after_largest < before_largest
    ):
        st.write("""
        The increase in connected components and decrease
        in the largest component ratio suggest some additional
        fragmentation in the observed network.

        However, this does not necessarily indicate
        a substantial breakdown of its main structure.
        """)

    elif (
        after_components < before_components
        and after_largest > before_largest
    ):
        st.write("""
        The decrease in connected components and increase
        in the largest component ratio suggest that the
        observed network became more connected.
        """)

    else:
        st.write("""
        The connectivity indicators provide mixed or
        limited evidence of changes in fragmentation.

        Additional analysis would be needed to determine
        whether the network became more dispersed.
        """)

    # Density interpretation
    if after_density > before_density:
        st.write("""
        Network density increased slightly, indicating
        a higher proportion of connected user pairs
        among the displayed users.
        """)

    elif after_density < before_density:
        st.write("""
        Network density decreased, indicating a lower
        proportion of connected user pairs among
        the displayed users.
        """)

    else:
        st.write("""
        Network density remained unchanged.
        """)

    #%% Connection to User Turnover
    st.markdown("#### Connection to User Turnover")

    st.write(f"""
    The user overlap analysis found that **{new_ratio:.1f}%**
    of post-update interaction participants were not observed
    in the pre-update period.

    Together, these findings suggest that the community
    experienced substantial changes in its observed user
    composition, alongside changes in connectivity.

    However, the analysis does not establish that newly
    observed users caused the connectivity changes.
    """)

    st.caption("""
    Connectivity metrics are calculated using the top N
    users selected independently for each period.

    The before and after observation windows differ in
    length (20 days versus 11 days), and the selected
    user populations are not identical.

    Therefore, these comparisons are descriptive
    rather than causal.
    """)

    #%% Network Methodology & Limitations
    with st.expander("Network Methodology & Limitations"):

        st.write("""
        **Network Construction**

        User-to-user connections were reconstructed using
        Reddit parent IDs.

        Replies to both submissions and comments were included.

        The graph is undirected: interactions between users
        A and B are combined regardless of reply direction.

        Self-replies and interactions with missing user
        identities are excluded from graph construction.

        **Node Selection**

        Users are ranked by their total number of reply
        interactions in the selected period.

        Only the top N users and the connections among
        them are displayed.

        **Community Detection**

        Greedy modularity optimization identifies groups
        of users that are more densely connected to one another.

        **Connectivity Metrics**

        Network density measures the proportion of possible
        user pairs that are connected.

        Connected components represent groups of users
        with no paths between different groups.

        The largest connected component ratio measures
        the percentage of displayed users belonging
        to the largest connected group.

        **Limitations**

        The displayed network is a subset of the full
        interaction network.

        Its community structure and degree values may
        differ from those of the complete network.

        A reply does not necessarily indicate agreement
        or a positive social relationship.

        The before and after periods have different
        observation lengths, and top users are selected
        independently in each period.

        Changes around the update are descriptive
        and do not establish causation.
        """)
