import pandas as pd
import numpy as np
from itertools import combinations
from modules.DataNormaliser import DataNormaliser
import math


class NetworkDistances:
    def __init__(self):
        self.orbit_counts_df = pd.read_csv("output/graphlet_counts.csv").drop(
            columns=["Unnamed: 0"]
        )
        self.similarity_measures_df = pd.DataFrame()
        self.column_combinations = list(combinations(self.orbit_counts_df.columns, 2))

    def computeRGFDist(self):
        computations = (
            DataNormaliser(self.orbit_counts_df)
            .log_scale_normalisation2()
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

            result_df[col2 + "-" + col1] = abs(computations[col2] - computations[col1])
            if len(zero_cols) > 0:
                result_df.iloc[zero_cols, result_df.shape[1] - 1] = 0

        self.similarity_measures_df["RGFDist"] = result_df.sum()

    def computeHellingerDist(self):
        computations = (
            DataNormaliser(self.orbit_counts_df)
            .percentual_normalisation()
            .apply(lambda col: col.map(lambda val: np.sqrt(val)))
        )

        result_df = pd.DataFrame()
        for col1, col2 in self.column_combinations:
            result_df[col2 + "-" + col1] = (
                computations[col2] - computations[col1]
            ) ** 2

        self.similarity_measures_df["Hellinger"] = np.sqrt(result_df.sum()) / np.sqrt(2)
        print(self.similarity_measures_df)


NetworkDistances().computeHellingerDist()
NetworkDistances().computeRGFDist()
