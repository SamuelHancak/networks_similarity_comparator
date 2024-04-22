import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


class DataNormaliser:
    def __init__(self, data):
        self.data = data

    def log_scale_percentual_normalisation(self):
        return self.percentual_normalisation().apply(np.log1p)

    def percentual_normalisation(self):
        return self.data.apply(lambda col: col / (col.sum() or 1))

    def discretize_data(self, n_bins=3):
        df_scaled = self.data.copy()
        discretizer = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy="kmeans", subsample=None
        )
        transformed_data = discretizer.fit_transform(df_scaled.iloc[:, 0:])
        df_scaled.iloc[:, 0:] = np.round(transformed_data)

        return df_scaled
