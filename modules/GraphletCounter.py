import pandas as pd
import subprocess
import os

from modules.DataNormaliser import DataNormaliser
from modules.NetworkDistances import NetworkDistances
from modules.DataClustering import DataClustering

PROCESS_CALL_NAME = "./orca/orcao"
GRAPHLETS_COUNTS_FILE_NAME = "graphlet_counts.csv"
SIMILARITY_MEASURES_FILE_NAME = "similarity_measures.csv"


class GraphletCounter:
    def __init__(
        self,
        input_folder_path=None,
        output_folder_path=None,
        out_data=False,
        normalise=False,
    ):
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.out_data = out_data
        self.normalise = normalise
        self.__orbit_couts_df = pd.DataFrame()

    @staticmethod
    def __sum_columns(file_name, column_name):
        df = pd.read_csv(file_name, sep=" ", header=None)
        column_sums = df.sum(axis=0)
        final_sums = pd.DataFrame({f"{column_name}": [0] * 30})

        final_sums[column_name][0] = round(column_sums[0] / 2)
        final_sums[column_name][1] = column_sums[2]
        final_sums[column_name][2] = round(column_sums[3] / 3)
        final_sums[column_name][3] = round(column_sums[4] / 2)
        final_sums[column_name][4] = column_sums[7]
        final_sums[column_name][5] = round(column_sums[8] / 4)
        final_sums[column_name][6] = column_sums[9]
        final_sums[column_name][7] = round(column_sums[12] / 2)
        final_sums[column_name][8] = round(column_sums[14] / 4)
        final_sums[column_name][9] = column_sums[17]
        final_sums[column_name][10] = column_sums[18]
        final_sums[column_name][11] = column_sums[23]
        final_sums[column_name][12] = column_sums[25]
        final_sums[column_name][13] = column_sums[27]
        final_sums[column_name][14] = column_sums[33]
        final_sums[column_name][15] = round(column_sums[34] / 5)
        final_sums[column_name][16] = column_sums[35]
        final_sums[column_name][17] = column_sums[39]
        final_sums[column_name][18] = column_sums[44]
        final_sums[column_name][19] = column_sums[45]
        final_sums[column_name][20] = round(column_sums[50] / 2)
        final_sums[column_name][21] = column_sums[52]
        final_sums[column_name][22] = round(column_sums[55] / 2)
        final_sums[column_name][23] = column_sums[56]
        final_sums[column_name][24] = column_sums[61]
        final_sums[column_name][25] = column_sums[62]
        final_sums[column_name][26] = column_sums[65]
        final_sums[column_name][27] = column_sums[69]
        final_sums[column_name][28] = round(column_sums[70] / 2)
        final_sums[column_name][29] = round(column_sums[72] / 5)

        return final_sums

    def __read_folder_files(self):
        file_paths = []
        try:
            for root, _, files in os.walk(self.input_folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file_path.endswith(".out" if self.out_data else ".in"):
                        file_paths.append(file_path)

        except Exception as e:
            print(f"An error occurred: {e}")

        return file_paths

    def orca_counting(self):
        orbit_counts_df = pd.DataFrame()
        input_files = self.__read_folder_files()

        for file in input_files:
            file_name = file.split("/")[-1].replace(
                ".out" if self.out_data else ".in", ""
            )
            output_file = (
                file
                if self.out_data
                else (
                    f"{self.output_folder_path}/{file_name}.out"
                    if self.output_folder_path is not None
                    else f"{file_name}.out"
                )
            )

            if not self.out_data:
                subprocess.call([PROCESS_CALL_NAME, "node", "5", file, output_file])
                print("\n")

            orbit_counts_df = pd.concat(
                [
                    orbit_counts_df,
                    self.__sum_columns(file_name=output_file, column_name=file_name),
                ],
                axis=1,
            )

            if self.output_folder_path is None and not self.out_data:
                os.remove(output_file)

        self.__orbit_couts_df = orbit_counts_df

        if self.normalise:
            orbit_counts_df = DataNormaliser(orbit_counts_df).percentual_normalization()
            if self.output_folder_path is not None:
                orbit_counts_df.to_csv(
                    f"{self.output_folder_path}/norm-{GRAPHLETS_COUNTS_FILE_NAME}",
                    encoding="utf-8",
                )

        if self.output_folder_path is not None:
            self.__orbit_couts_df.to_csv(
                f"{self.output_folder_path}/{GRAPHLETS_COUNTS_FILE_NAME}",
                encoding="utf-8",
            )

        return orbit_counts_df

    def count_network_similarities(
        self,
        countRGFD=True,
        countSimDisp=True,
        countHellinger=True,
        countMinkowski=True,
        countCosine=True,
    ):
        dist = NetworkDistances(orbit_counts_df=self.__orbit_couts_df)

        if countRGFD:
            dist.computeRGFDist()

        if countSimDisp:
            dist.computeSimpleDispersionDist()

        if countHellinger:
            dist.computeHellingerDist()

        if countMinkowski:
            dist.computeMinkowskiDist()

        if countCosine:
            dist.computeCosineDist()

        if self.output_folder_path is not None:
            dist.similarity_measures_df.to_csv(
                f"{self.output_folder_path}/{SIMILARITY_MEASURES_FILE_NAME}",
                encoding="utf-8",
            )

        return dist.similarity_measures_df

    def get_orbit_counts_df(self):
        return self.__orbit_couts_df
