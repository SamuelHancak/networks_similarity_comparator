import os
import pandas as pd
import os

input_folder_path = "as-733"
output_folder_path = "as-733-out"

os.makedirs(output_folder_path, exist_ok=True)

for filename in os.listdir(input_folder_path):
    if filename.endswith(".txt"):
        input_file_path = os.path.join(input_folder_path, filename)
        output_file_path = os.path.join(
            output_folder_path, os.path.splitext(filename)[0] + ".in"
        )

        with open(input_file_path, "r") as input_file:
            lines = input_file.readlines()

        data_lines = [line.strip() for line in lines if not line.startswith("#")]

        with open(output_file_path, "w") as output_file:
            output_file.write(f"{lines[2][8:12]} {lines[2][20:24]}\n")

            for line in data_lines[1:]:
                values = line.split("\t")
                values = [str(int(value) - 1) for value in values]
                output_file.write(" ".join(values) + "\n")

        print(
            f"Conversion completed for {filename}. Data written to {output_file_path}"
        )


def convert_csv_to_in(csv_file_path, in_file_path):
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path, header=None, skiprows=1)
    verteces_count = df.max().max() + 1
    edges_count = len(df)

    # Create a new DataFrame with the new row
    new_row = pd.DataFrame([[verteces_count, edges_count]], columns=df.columns)

    # Concatenate the new row DataFrame with the original DataFrame
    df = pd.concat([new_row, df]).reset_index(drop=True)

    df.to_csv(in_file_path, header=False, index=False, sep=" ")


def convert_txt_to_in(txt_file_path, in_file_path):
    # Read txt file into a pandas DataFrame
    df = pd.read_csv(txt_file_path, header=None, sep="\s+")

    # Write DataFrame to a .in file
    df.to_csv(in_file_path, header=False, index=False, sep=" ")


def process_directory(input_dir, output_dir):
    # Get a list of all files in the directory
    files = os.listdir(input_dir)

    for file in files:
        # Construct the full file paths
        input_file_path = os.path.join(input_dir, file)
        output_file_path = os.path.join(output_dir, file).split(".")[0] + ".in"

        # print(input_file_path)
        # print(output_file_path)

        # Determine the file extension
        _, ext = os.path.splitext(file)

        # Convert the file based on its extension
        if ext == ".csv":
            convert_csv_to_in(input_file_path, output_file_path)
        elif ext == ".orca":
            convert_txt_to_in(input_file_path, output_file_path)


# input_dir = "output/rgSiete"
# output_dir = "output/rgSieteOut"
# process_directory(input_dir, output_dir)


csv_file_path = "output/musae_facebook_edges.csv"
in_file_path = "output/musae_facebook_edges.in"
convert_csv_to_in(csv_file_path, in_file_path)

# file_path = "output/dataScaleFree/sf-n4000-g2_2-kmin2-0.txt"
# file_path_out = "output/dataScaleFreeOut/sf-n4000-g2_2-kmin2-0.in"
# convert_txt_to_in(file_path, file_path_out)
