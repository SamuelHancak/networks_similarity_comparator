import pandas as pd
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
from sklearn.preprocessing import KBinsDiscretizer


class ROCCurveVisualiser:
    def __init__(
        self,
        input_df,
        compared_measure,
    ):
        self.input_df = input_df
        self.compared_measure = compared_measure

    def generate_roc_curve(self):
        discretizer = KBinsDiscretizer(
            n_bins=2, encode="ordinal", strategy="kmeans", subsample=None
        )
        true_values = discretizer.fit_transform(
            self.input_df[self.compared_measure].values.reshape(-1, 1)
        ).flatten()
        fig = go.Figure()

        for column in self.input_df.columns:
            fpr, tpr, _ = roc_curve(true_values, self.input_df[column])

            auc_value = auc(fpr, tpr)

            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"{column} (AUC = {auc_value:.2f})"
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(dash="dash"),
                showlegend=False,
            )
        )

        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            title="ROC Curve for Similarity Measures",
            showlegend=True,
            legend=dict(
                orientation="h",
                entrywidth=200,
                yanchor="bottom",
                y=1.01,
                xanchor="auto",
                x=1,
            ),
        )

        fig.show()
