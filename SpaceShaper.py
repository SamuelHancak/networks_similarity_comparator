import pandas as pd


graphlet_counts_df = pd.read_csv(
    "neuralNetwork/train_data/graphlet_counts_final.csv"
).drop(columns=["Unnamed: 0"])
graphlet_counts_df.to_csv(
    "neuralNetwork/train_data/graphlet_counts_final_2.csv", index=False
)
print(graphlet_counts_df.head())
# similarity_measures_df = pd.read_csv(
#     "neuralNetwork/train_data/similarity_measures_final.csv"
# )
# similarity_measures_df.rename(columns={"Unnamed: 0": "Pair"}, inplace=True)
# similarity_measures_df.to_csv(
#     "neuralNetwork/train_data/similarity_measures_final_2.csv", index=False
# )

# print(similarity_measures_df.head())
