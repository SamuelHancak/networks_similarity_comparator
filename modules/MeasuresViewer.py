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

        columns = [self.similarity_measures.index.name] + list(
            self.similarity_measures.columns
        )

        tree = ttk.Treeview(
            second_window,
            columns=columns,
            show="headings",
        )

        for col in columns:
            tree.heading(col, text=col, command=lambda c=col: self.sort_tree(tree, c))
            tree.column(col, anchor="center")

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
            tree.heading(col, text=col, command=lambda c=col: self.sort_tree(tree, c))
            tree.column(col, anchor="center")

        for _, row in graphlet_counts.iterrows():
            tree.insert("", "end", values=list(row))

        tree.pack(fill="both", expand=True)

    def sort_tree(self, tree, column, reverse=False):
        current_data = [
            (tree.set(child, column), child) for child in tree.get_children("")
        ]
        try:
            sorted_data = sorted(
                current_data, key=lambda x: float(x[0]), reverse=reverse
            )
        except ValueError:
            sorted_data = sorted(current_data, key=lambda x: x[0], reverse=reverse)

        for index, item in enumerate(sorted_data):
            tree.move(item[1], "", index)

        tree.heading(column, command=lambda: self.sort_tree(tree, column, not reverse))
