import os
from netrd.distance import NetSimile
import networkx as nx
import pandas as pd
import shutil


TARGET_DIRECTORY = "netSimFiles"


class NetSimileClass:
    def __init__(self, path):
        self.path = path
        self.result_df = pd.DataFrame()

    def remove_first_line(self):
        if not os.path.exists(TARGET_DIRECTORY):
            os.makedirs(TARGET_DIRECTORY)
        else:
            shutil.rmtree(TARGET_DIRECTORY)
            os.makedirs(TARGET_DIRECTORY)

        for file in os.listdir(self.path):
            if file.endswith(".in"):
                with open(os.path.join(self.path, file), "r") as f:
                    lines = f.readlines()
                with open(os.path.join(TARGET_DIRECTORY, file), "w") as f:
                    f.writelines(lines[1:])

    def calculate_net_simile(self):
        net_simile = NetSimile()
        files = os.listdir(TARGET_DIRECTORY)

        for i, file1 in enumerate(files):
            print(f"Processing file {file1}")
            for j, file2 in enumerate(files):
                if i < j:
                    G1 = nx.read_edgelist(os.path.join(TARGET_DIRECTORY, file1))
                    G2 = nx.read_edgelist(os.path.join(TARGET_DIRECTORY, file2))
                    distance = net_simile.dist(G1, G2)
                    file_names = sorted([file1.split(".")[0], file2.split(".")[0]])
                    self.result_df[file_names[0] + "---" + file_names[1]] = [distance]

        shutil.rmtree(TARGET_DIRECTORY)

    def compile_process(self):
        self.remove_first_line()
        self.calculate_net_simile()
        self.result_df = self.result_df.T
        self.result_df.columns = ["NetSimDist"]

        return self.result_df
