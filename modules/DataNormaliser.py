import numpy as np


class DataNormaliser:
    def __init__(self, data):
        self.data = data

    def min_max_normalisation(self):
        df_scaled = self.data.copy()

        for column in df_scaled.columns:
            df_scaled[column] = (df_scaled[column] - df_scaled[column].min()) / (
                df_scaled[column].max() - df_scaled[column].min()
            )

        return df_scaled

    def log_scale_normalization(self):
        return self.data.apply(np.log)

    def log_scale_percentual_normalization(self):
        return self.percentual_normalization().apply(np.log1p)

    def percentual_normalization(self):
        return self.data.apply(lambda col: col / (col.sum() or 1))
