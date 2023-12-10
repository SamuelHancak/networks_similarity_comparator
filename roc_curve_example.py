import pandas as pd
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go

COMPARED_MEASURE = "Minkowski"
MAIN_THRESHOLD_VALUE = 0.5

input_df = (
    pd.read_csv("similarity_measures.csv").drop("Unnamed: 0", axis=1).astype("float")
)

true_values = (input_df[COMPARED_MEASURE] < MAIN_THRESHOLD_VALUE).astype(int).values

fig = go.Figure()

for column in input_df.columns:
    fpr, tpr, _ = roc_curve(true_values, input_df[column])

    fpr = 1 - fpr
    tpr = 1 - tpr

    auc_value = auc(fpr, tpr)

    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{column} (AUC = {auc_value:.2f})")
    )

fig.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash"))
)

fig.update_layout(
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    title="ROC Curve for Similarity Measures",
    showlegend=True,
)

fig.show()
