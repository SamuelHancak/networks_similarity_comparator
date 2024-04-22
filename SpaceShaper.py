# import pandas as pd


# graphlet_counts_df = pd.read_csv(
#     "neuralNetwork/train_data/graphlet_counts_final.csv"
# ).drop(columns=["Unnamed: 0"])
# graphlet_counts_df.to_csv(
#     "neuralNetwork/train_data/graphlet_counts_final_2.csv", index=False
# )
# print(graphlet_counts_df.head())


# similarity_measures_df = pd.read_csv(
#     "neuralNetwork/train_data/similarity_measures_final.csv"
# )
# similarity_measures_df.rename(columns={"Unnamed: 0": "Pair"}, inplace=True)
# similarity_measures_df.to_csv(
#     "neuralNetwork/train_data/similarity_measures_final_2.csv", index=False
# )

# print(similarity_measures_df.head())

"input/example_5000.in"


def remove_first_line(input_file):
    """
    Remove the first line from a file and write the result to a new file.
    :param input_file: The name of the input file.
    :param output_file: The name of the output file.
    """
    with open(input_file, "r") as file:
        lines = file.readlines()
    lines.pop(0)
    print(lines)
    # with open(output_file, 'w') as file:
    #     file.writelines(lines)


remove_first_line("input/example_5000.in")
