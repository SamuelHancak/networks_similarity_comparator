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
