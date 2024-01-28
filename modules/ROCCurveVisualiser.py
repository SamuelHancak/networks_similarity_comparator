from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go


class ROCCurveVisualiser:
    def __init__(
        self,
        input_df,
        compared_measure="Hellinger",
    ):
        self.input_df = input_df
        self.compared_measure = compared_measure

    def generate_roc_curve(self):
        true_values = (
            (
                self.input_df[self.compared_measure]
                < self.input_df[self.compared_measure].mean()
            )
            .astype(int)
            .values
        )
        fig = go.Figure()

        for column in self.input_df.columns:
            fpr, tpr, _ = roc_curve(true_values, self.input_df[column])

            fpr = 1 - fpr
            tpr = 1 - tpr

            auc_value = auc(fpr, tpr)

            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"{column} (AUC = {auc_value:.2f})"
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")
            )
        )

        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            title="ROC Curve for Similarity Measures",
            showlegend=True,
        )

        fig.show()


# file_path = "similarity_measures.csv"
# roc_plotter = SimilarityMeasuresROC(
#     file_path=file_path,
#     compared_measure="Minkowski",
# )
# roc_plotter.generate_roc_curve()