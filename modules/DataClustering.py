import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from modules.DataNormaliser import DataNormaliser


class DataClustering:
    def __init__(self, input_df, num_clusters=6):
        self.num_clusters = num_clusters
        self.input_df = input_df
        self.graphlet_counts = None
        self.feature_matrix = None
        self.network_names = None
        self.labels = None
        self.distances = None
        self.result_df = None
        self.reduced_features = None
        self.cluster_centers = None
        self.cluster_df = None
        self.center_df = None
        self.kmeans = None

    def load_and_normalize_data(self):
        self.graphlet_counts = (
            DataNormaliser(self.input_df).percentual_normalization().T
        )
        self.feature_matrix = self.graphlet_counts.values
        self.network_names = self.graphlet_counts.index

    def perform_clustering(self):
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.labels = self.kmeans.fit_predict(self.feature_matrix)
        self.distances = self.kmeans.transform(self.feature_matrix)

        self.result_df = pd.DataFrame(
            {
                "Network": self.network_names,
                "Cluster": self.labels,
                "DistanceToCentroid": [
                    distance[label]
                    for label, distance in zip(self.labels, self.distances)
                ],
            }
        )

        self.result_df.to_csv("cluster_results.csv", index=False)

    def visualize_clusters_3d(self):
        pca = PCA(n_components=3)
        self.reduced_features = pca.fit_transform(self.feature_matrix)

        cluster_centers = pca.transform(self.kmeans.cluster_centers_)

        self.cluster_df = pd.DataFrame(
            {"Cluster": self.labels, "Network": self.network_names}
        )
        self.center_df = pd.DataFrame(
            {
                "Cluster": range(self.num_clusters),
                "Network": [f"Cluster Center {i}" for i in range(self.num_clusters)],
            }
        )

        fig = px.scatter_3d(
            self.cluster_df,
            x=self.reduced_features[:, 0],
            y=self.reduced_features[:, 1],
            z=self.reduced_features[:, 2],
            color="Cluster",
            hover_name="Network",
            labels={"Cluster": "Cluster"},
            size_max=5,
            hover_data={"Cluster": False, "Network": True},
        )

        fig.add_trace(
            px.scatter_3d(
                self.center_df,
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


# graphlets_path = "graphlet_counts_dif.csv"
# cluster_visualizer = KMeansClusterVisualizer(data_path=graphlets_path, num_clusters=6)
# cluster_visualizer.load_and_normalize_data()
# cluster_visualizer.perform_clustering()
# cluster_visualizer.visualize_clusters_3d()
