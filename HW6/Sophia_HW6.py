#%%
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities
import streamlit.components.v1 as components
from pyvis.network import Network

st.set_page_config(
    page_title="Florentine Families Network Explorer",
    layout="wide"
)

st.title("Florentine Families Network Explorer")

st.caption(
    "Exploring marriage relationships and social structures in Renaissance Florence"
)
#%%
@st.cache_data
def load_data():
    G = nx.florentine_families_graph()
    return G

G = load_data()

#%%
st.header("About the Dataset")

st.write("""
The Florentine Families network represents marriage relationships
among 15 prominent families in Renaissance Florence.
Each node represents a family, and each edge represents a
marriage connection between two families.
""")

st.write(
    "Number of nodes:", G.number_of_nodes(),
    "| Number of edges:", G.number_of_edges()
)

#%%
@st.cache_data
def calculate_metrics():
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    communities = list(greedy_modularity_communities(G))

    community_map = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            community_map[node] = cid

    return degree, betweenness, community_map

degree, betweenness, community_map = calculate_metrics()

#%%
st.header("Network Visualization")

centrality = st.selectbox(
    "Node Size Represents",
    ["Degree", "Betweenness Centrality"]
)

layout = st.selectbox(
    "Network Layout",
    ["Spring", "Circular"]
)


#%%
# Select centrality measure
if centrality == "Degree":
    node_values = degree
else:
    node_values = betweenness

# Create PyVis network
net = Network(
    height="600px",
    width="100%",
    cdn_resources="in_line",
    notebook=False
)

# Community colors
palette = ["#2E6E8E", "#d9534f", "#6aa84f", "#b07aa1"]

# Add nodes
for n in G.nodes():
    net.add_node(
        n,
        label=n,
        size=18 + node_values[n] * (
            5 if centrality == "Degree" else 60
        ),
        color=palette[community_map[n] % len(palette)],
        title=f"{n} | {centrality}: {node_values[n]:.3f}",
        font={"size": 18, "color": "#222222"}
    )

# Add edges
for u, v in G.edges():
    net.add_edge(u, v, color="#cccccc")

# Layout selector
if layout == "Spring":
    net.barnes_hut(
        gravity=-2000,
        central_gravity=0.5,
        spring_length=100,
        spring_strength=0.05
    )
else:
    pos = nx.circular_layout(G, scale=250)

    for node in net.nodes:
        node["x"], node["y"] = map(float, pos[node["id"]])
        node["physics"] = False

    net.toggle_physics(False)

# Display interactive network
components.html(
    net.generate_html(notebook=False),
    height=620
)

#%%
st.header("Visual Encodings and Interactivity")

st.write("""
Node size represents degree or betweenness centrality.
Degree measures the number of direct marriage connections,
while betweenness centrality measures how often a family lies
on shortest paths between other families.

Node color represents communities detected using the greedy
modularity algorithm. Families with the same color belong
to the same community.

The centrality selector allows users to compare different
measures of structural importance. The layout selector
shows how the same network appears under different layouts.

In this network, the Medici family stands out in both degree
and betweenness centrality, suggesting that it has many direct
marriage connections and also plays an important bridging role
between different parts of the network.

Comparing the two centrality measures reveals that having
many direct connections does not necessarily mean a family
plays an equally important bridging role.""")

st.header("Limitation")

st.write("""
The network only represents marriage relationships and does
not capture other forms of influence or relationships, such as wealth,
political power, or business alliances. Therefore, a family
with high centrality is not necessarily the most powerful
family in Renaissance Florence.
""")
