import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from DataNormalizer import DataNormalizer

NUMBER_OF_CLUSTERS = 6
graphlets_path = "graphlet_counts_dif.csv"

graphlet_counts = (
    DataNormalizer(pd.read_csv(graphlets_path).drop(columns=["Unnamed: 0"]))
    .percentual_normalization()
    .T
)
feature_matrix = graphlet_counts.values
network_names = graphlet_counts.index

kmeans = KMeans(n_clusters=NUMBER_OF_CLUSTERS, random_state=42)
labels = kmeans.fit_predict(feature_matrix)
distances = kmeans.transform(feature_matrix)

result_df = pd.DataFrame(
    {
        "Network": network_names,
        "Cluster": labels,
        "DistanceToCentroid": [
            distance[label] for label, distance in zip(labels, distances)
        ],
    }
)

result_df.to_csv("cluster_results.csv", index=False)

for network, label, distance in zip(network_names, labels, distances):
    print(f"{network} - cluster: {label} distance: {distance[label]}")

pca = PCA(n_components=3)
reduced_features = pca.fit_transform(feature_matrix)

cluster_centers = pca.transform(kmeans.cluster_centers_)

cluster_df = pd.DataFrame({"Cluster": labels, "Network": network_names})
center_df = pd.DataFrame(
    {
        "Cluster": range(NUMBER_OF_CLUSTERS),
        "Network": [f"Cluster Center {i}" for i in range(NUMBER_OF_CLUSTERS)],
    }
)

fig = px.scatter_3d(
    cluster_df,
    x=reduced_features[:, 0],
    y=reduced_features[:, 1],
    z=reduced_features[:, 2],
    color="Cluster",
    hover_name="Network",
    labels={"Cluster": "Cluster"},
    size_max=5,
    hover_data={"Cluster": False, "Network": True},
)

fig.add_trace(
    px.scatter_3d(
        center_df,
        x=cluster_centers[:, 0],
        y=cluster_centers[:, 1],
        z=cluster_centers[:, 2],
        color="Cluster",
        hover_name="Network",
        labels={"Cluster": "Cluster"},
        size_max=10,
    )
    .update_traces(marker=dict(symbol="cross"))
    .data[0]
)

fig.update_layout(
    title_text="KMeans Clustering (3D) with Hover Labels and Cluster Centers"
)
fig.show()
