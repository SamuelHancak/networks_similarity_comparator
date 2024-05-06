import pandas as pd
from itertools import combinations
import numpy as np
from modules.DataNormaliser import DataNormaliser
from sklearn.metrics.pairwise import cosine_distances


class NetworkDistances:
    def __init__(self, orbit_counts_df):
        self.orbit_counts_df = orbit_counts_df
        self.similarity_measures_df = pd.DataFrame()
        self.column_combinations = sorted(
            list(combinations(sorted(self.orbit_counts_df.columns), 2))
        )
        self.orbit_counts_percentual_normal = DataNormaliser(
            orbit_counts_df
        ).percentual_normalisation()

    def computeRGFDist(self):
        computations = self.orbit_counts_percentual_normal.apply(
            lambda col: col.map(lambda val: (-1 * (np.log10(val) if val > 0 else 0)))
        )

        result_df = pd.DataFrame()
        for col1, col2 in self.column_combinations:
            distance = np.abs(computations[col2] - computations[col1])
            zero_mask = (computations[col1] == 0) | (computations[col2] == 0)
            distance.loc[zero_mask] = 0

            result_df[col1 + "---" + col2] = distance

        self.similarity_measures_df["RGFDist"] = result_df.sum()

    def computeSimpleDispersionDist(self):
        computations = self.orbit_counts_percentual_normal.apply(
            lambda col: col.map(lambda val: val / col.sum())
        )

        result_df = pd.DataFrame()
        for col1, col2 in self.column_combinations:
            result_df[col1 + "---" + col2] = (
                computations[col1] - computations[col2]
            ).abs()

        self.similarity_measures_df["SimDisp"] = result_df.sum() / 2

    def computeHellingerDist(self):
        computations = self.orbit_counts_percentual_normal.apply(
            lambda col: col.map(lambda val: np.sqrt(val))
        )

        result_df = pd.DataFrame()
        for col1, col2 in self.column_combinations:
            result_df[col1 + "---" + col2] = (
                computations[col1] - computations[col2]
            ) ** 2

        self.similarity_measures_df["Hellinger"] = np.sqrt(result_df.sum()) / np.sqrt(2)

    def computeMinkowskiDist(self, p_value):
        p = p_value

        result_df = pd.DataFrame()
        for col1, col2 in self.column_combinations:
            result_df[col1 + "---" + col2] = (
                self.orbit_counts_percentual_normal[col1]
                - self.orbit_counts_percentual_normal[col2]
            ).abs() ** p

        self.similarity_measures_df[f"Minkowski(p={p_value})"] = result_df.sum() ** (
            1 / p
        )

    def computeCosineDist(self):
        similarity_matrix = cosine_distances(self.orbit_counts_percentual_normal.T)

        result_df = pd.DataFrame()
        for col1, col2 in self.column_combinations:
            result_df[col1 + "---" + col2] = [
                similarity_matrix[
                    self.orbit_counts_percentual_normal.columns.get_loc(col1),
                    self.orbit_counts_percentual_normal.columns.get_loc(col2),
                ]
            ]

        self.similarity_measures_df["Cosine"] = result_df.T
