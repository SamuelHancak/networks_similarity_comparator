import pandas as pd
from itertools import combinations
import numpy as np
from modules.DataNormaliser import DataNormaliser
from sklearn.metrics.pairwise import cosine_distances


class NetworkDistances:
    def __init__(self, orbit_counts_df):
        self.orbit_counts_df = orbit_counts_df
        self.similarity_measures_df = pd.DataFrame()
        self.column_combinations = list(combinations(self.orbit_counts_df.columns, 2))
        self.orbit_counts_log_normal = DataNormaliser(
            orbit_counts_df
        ).log_scale_percentual_normalisation()
        self.orbit_counts_percentual_normal = DataNormaliser(
            orbit_counts_df
        ).percentual_normalisation()

    def computeRGFDist(self):
        computations = (
            DataNormaliser(self.orbit_counts_df)
            .log_scale_percentual_normalisation()
            .apply(
                lambda col: col.map(
                    lambda val: (-1 * (np.log10(val) if val > 0 else 0))
                    / (np.log10(col.sum()) if col.sum() > 1 else 1)
                )
            )
        )

        result_df = pd.DataFrame()
        for col1, col2 in self.column_combinations:
            zero_cols = computations.loc[
                (computations[col1] == 0) | (computations[col2] == 0)
            ].index.to_numpy()

            result_df[col2 + "---" + col1] = abs(
                computations[col2] - computations[col1]
            )
            if len(zero_cols) > 0:
                result_df.iloc[zero_cols, result_df.shape[1] - 1] = 0

        self.similarity_measures_df["RGFDist"] = result_df.sum()

    def computeSimpleDispersionDist(self):
        computations = self.orbit_counts_percentual_normal.apply(
            lambda col: col.map(lambda val: val / col.sum())
        )

        result_df = pd.DataFrame()
        for col1, col2 in self.column_combinations:
            result_df[col2 + "---" + col1] = (
                computations[col2] - computations[col1]
            ).abs()

        self.similarity_measures_df["SimDisp"] = result_df.sum() / 2

    def computeHellingerDist(self):
        computations = self.orbit_counts_percentual_normal.apply(
            lambda col: col.map(lambda val: np.sqrt(val))
        )

        result_df = pd.DataFrame()
        for col1, col2 in self.column_combinations:
            result_df[col2 + "---" + col1] = (
                computations[col2] - computations[col1]
            ) ** 2

        self.similarity_measures_df["Hellinger"] = np.sqrt(result_df.sum()) / np.sqrt(2)

    def computeMinkowskiDist(self):
        p = 2

        result_df = pd.DataFrame()
        for col1, col2 in self.column_combinations:
            result_df[col2 + "---" + col1] = (
                self.orbit_counts_percentual_normal[col2]
                - self.orbit_counts_percentual_normal[col1]
            ).abs() ** p

        self.similarity_measures_df["Minkowski"] = result_df.sum() ** (1 / p)

    def computeCosineDist(self):
        similarity_matrix = 1 - cosine_distances(self.orbit_counts_percentual_normal.T)

        result_df = pd.DataFrame()
        for col1, col2 in self.column_combinations:
            result_df[col2 + "---" + col1] = (
                similarity_matrix[
                    self.orbit_counts_percentual_normal.columns.get_loc(col2)
                ]
                - similarity_matrix[
                    self.orbit_counts_percentual_normal.columns.get_loc(col1)
                ]
            )

        self.similarity_measures_df["Cosine"] = result_df.abs().sum() / 2

    def computeJaccardSimilarity(self):
        computations = self.orbit_counts_df.copy()

        result_df = pd.DataFrame()
        for col1, col2 in self.column_combinations:
            set1 = set(computations[col1].index[computations[col1] > 0])
            set2 = set(computations[col2].index[computations[col2] > 0])

            intersection_size = len(set1.intersection(set2))
            union_size = len(set1.union(set2))

            result_df[col2 + "---" + col1] = (
                intersection_size / union_size if union_size != 0 else 0
            )

        self.similarity_measures_df["JaccardSimilarity"] = result_df.mean()
