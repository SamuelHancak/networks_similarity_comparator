import tkinter as tk
from tkinter import ttk
import pandas as pd
from modules.DataVisualiser import CATEGORIES


class MeasuresViewer:
    def __init__(
        self, root, similarity_measures=pd.DataFrame(), graphlet_counts=pd.DataFrame()
    ):
        self.root = root
        self.similarity_measures = similarity_measures
        self.graphlet_counts = graphlet_counts

    def display_similarity_measures(self):
        second_window = tk.Toplevel(self.root)
        second_window.title("Similarity measures")

        if self.similarity_measures.index.name is None:
            self.similarity_measures.index.name = ""

        means = self.similarity_measures.mean()
        std_devs = self.similarity_measures.std()

        mean_val = ["Mean"] + list(means)
        stdDev_val = ["Standard deviation"] + list(std_devs)

        columns = [self.similarity_measures.index.name] + list(
            self.similarity_measures.columns
        )

        tree = ttk.Treeview(
            second_window,
            columns=columns,
            show="headings",
        )

        for col in columns:
            tree.heading(
                col,
                text=col,
                command=lambda c=col: self.sort_tree(tree, c, True),
            )
            tree.column(col, anchor="center")

        tree.insert("", "end", values=mean_val)
        tree.insert("", "end", values=stdDev_val)

        for index, row in self.similarity_measures.iterrows():
            tree.insert("", "end", values=[index] + list(row))

        tree.pack(fill="both", expand=True)

    def display_graphlet_counts(self):
        second_window = tk.Toplevel(self.root)
        second_window.title("Graphlet counts")
        graphlet_counts = self.graphlet_counts.copy()
        graphlet_counts.insert(0, "", CATEGORIES)

        tree = ttk.Treeview(
            second_window,
            columns=list(graphlet_counts.columns),
            show="headings",
        )

        for col in graphlet_counts.columns:
            tree.heading(
                col,
                text=col,
                command=lambda c=col: self.sort_tree(tree, c, False),
            )
            tree.column(col, anchor="center")

        for _, row in graphlet_counts.iterrows():
            tree.insert("", "end", values=list(row))

        tree.pack(fill="both", expand=True)

    def sort_tree(self, tree, column, is_similarity, reverse=False):
        data_to_sort = [
            (tree.set(child, column), child) for child in tree.get_children("")
        ]

        mean_row = None
        std_dev_row = None
        data_to_sort_excluding_mean_std = []

        if is_similarity:
            for value, child in data_to_sort:
                if tree.index(child) == 0:
                    mean_row = (value, child)
                elif tree.index(child) == 1:
                    std_dev_row = (value, child)
                else:
                    data_to_sort_excluding_mean_std.append((value, child))
        else:
            data_to_sort_excluding_mean_std = data_to_sort

        try:
            sorted_data = sorted(
                data_to_sort_excluding_mean_std,
                key=lambda x: float(x[0]),
                reverse=reverse,
            )
        except ValueError:
            sorted_data = sorted(
                data_to_sort_excluding_mean_std, key=lambda x: x[0], reverse=reverse
            )

        if mean_row:
            sorted_data.insert(0, mean_row)

        if std_dev_row:
            sorted_data.insert(1, std_dev_row)

        for index, item in enumerate(sorted_data):
            tree.move(item[1], "", index)

        tree.heading(
            column,
            command=lambda: self.sort_tree(tree, column, is_similarity, not reverse),
        )
