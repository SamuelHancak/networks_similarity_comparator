# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import plotly.express as px
# from modules.DataNormaliser import DataNormaliser


# class DataClustering:
#     def __init__(self, input_df, num_clusters=1):
#         self.num_clusters = num_clusters
#         self.input_df = input_df
#         self.graphlet_counts = None
#         self.feature_matrix = None
#         self.network_names = None
#         self.labels = None
#         self.distances = None
#         self.result_df = None
#         self.reduced_features = None
#         self.cluster_centers = None
#         self.cluster_df = None
#         self.center_df = None
#         self.kmeans = None

#     def __load_and_normalize_data(self):
#         self.graphlet_counts = (
#             DataNormaliser(self.input_df).percentual_normalization().T
#         )
#         self.feature_matrix = self.graphlet_counts.values
#         self.network_names = self.graphlet_counts.index

#     def __perform_clustering(self):
#         self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
#         self.labels = self.kmeans.fit_predict(self.feature_matrix)
#         self.distances = self.kmeans.transform(self.feature_matrix)

#         self.result_df = pd.DataFrame(
#             {
#                 "Network": self.network_names,
#                 "Cluster": self.labels,
#                 "DistanceToCentroid": [
#                     distance[label]
#                     for label, distance in zip(self.labels, self.distances)
#                 ],
#             }
#         )

#         self.result_df.to_csv("cluster_results.csv", index=False)

#     def __visualize_clusters_3d(self):
#         pca = PCA()
#         self.reduced_features = pca.fit_transform(self.feature_matrix)

#         cluster_centers = pca.transform(self.kmeans.cluster_centers_)

#         self.cluster_df = pd.DataFrame(
#             {"Cluster": self.labels, "Network": self.network_names}
#         )
#         self.center_df = pd.DataFrame(
#             {
#                 "Cluster": range(self.num_clusters),
#                 "Network": [f"Cluster Center {i}" for i in range(self.num_clusters)],
#             }
#         )

#         fig = px.scatter_3d(
#             self.cluster_df,
#             x=self.reduced_features[:, 0],
#             y=self.reduced_features[:, 1],
#             z=self.reduced_features[:, 2],
#             color="Cluster",
#             hover_name="Network",
#             labels={"Cluster": "Cluster"},
#             size_max=5,
#             hover_data={"Cluster": False, "Network": True},
#         )

#         fig.add_trace(
#             px.scatter_3d(
#                 self.center_df,
#                 x=cluster_centers[:, 0],
#                 y=cluster_centers[:, 1],
#                 z=cluster_centers[:, 2],
#                 color="Cluster",
#                 hover_name="Network",
#                 labels={"Cluster": "Cluster"},
#                 size_max=10,
#             )
#             .update_traces(marker=dict(symbol="cross"))
#             .data[0]
#         )

#         fig.update_layout(title_text="KMeans Clustering")
#         fig.show()

#     def clustering(self):
#         self.__load_and_normalize_data()
#         self.__perform_clustering()
#         self.__visualize_clusters_3d()


# graphlets_path = "graphlet_counts_dif.csv"
# cluster_visualizer = DataClustering(data_path=graphlets_path, num_clusters=6)
# cluster_visualizer.load_and_normalize_data()
# cluster_visualizer.perform_clustering()
# cluster_visualizer.visualize_clusters_3d()


# ----------

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import plotly.express as px
from modules.DataNormaliser import DataNormaliser


class DataClustering:
    def __init__(self, input_df, eps=100, min_samples=2):
        self.eps = eps
        self.min_samples = min_samples
        self.input_df = input_df
        self.graphlet_counts = None
        self.feature_matrix = None
        self.network_names = None
        self.labels = None
        self.distances = None
        self.result_df = None
        self.reduced_features = None
        self.cluster_df = None

    def __load_and_normalize_data(self):
        self.graphlet_counts = (
            DataNormaliser(self.input_df).percentual_normalization().T
        )
        self.feature_matrix = StandardScaler().fit_transform(
            self.graphlet_counts.values
        )
        self.network_names = self.graphlet_counts.index

    def __perform_clustering(self):
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = dbscan.fit_predict(self.feature_matrix)

        self.result_df = pd.DataFrame(
            {
                "Network": self.network_names,
                "Cluster": self.labels,
            }
        )

        self.result_df.to_csv("cluster_results.csv", index=False)

    def __visualize_clusters_3d(self):
        pca = PCA(n_components=3)
        self.reduced_features = pca.fit_transform(self.feature_matrix)

        self.cluster_df = pd.DataFrame(
            {"Cluster": self.labels, "Network": self.network_names}
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

        # Calculate and plot the cluster centers
        cluster_centers = pd.DataFrame(self.reduced_features, columns=["X", "Y", "Z"])
        cluster_centers["Cluster"] = self.labels
        cluster_centers = cluster_centers.groupby("Cluster").mean().reset_index()

        fig.add_trace(
            px.scatter_3d(
                cluster_centers,
                x="X",
                y="Y",
                z="Z",
                color="Cluster",
                hover_data={"Cluster": True},
            )
            .update_traces(marker=dict(symbol="cross"))
            .data[0]
        )

        fig.update_layout(
            title_text="DBSCAN Clustering (3D) with Hover Labels and Cluster Centers"
        )
        fig.show()

        fig.update_layout(title_text="DBSCAN Clustering (3D) with Hover Labels")
        fig.show()

    def clustering(self):
        self.__load_and_normalize_data()
        self.__perform_clustering()
        self.__visualize_clusters_3d()
