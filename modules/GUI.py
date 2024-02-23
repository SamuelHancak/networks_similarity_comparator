from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
from tkinterdnd2 import DND_FILES, TkinterDnD

from modules.DataVisualiser import DataVisualiser
from modules.GraphletCounter import GraphletCounter
from modules.MeasuresViewer import MeasuresViewer
from modules.ROCCurveVisualiser import ROCCurveVisualiser
from modules.DataClustering import DataClustering
from neuralNetwork.NetworkClass import SiameseNetwork


class GUI:
    def __init__(self):
        self.root = None
        self.orbit_counts_df = None
        self.similarities_df = pd.DataFrame()
        self.g_counter = None

    def __open_folder_dialog(self, folder_listbox, process_data_btn=None):
        selected_folder = filedialog.askdirectory(
            initialdir="/",
            title="Select A Folder",
        )

        if selected_folder:
            folder_listbox.delete(0, END)
            folder_listbox.insert(END, selected_folder)

            if process_data_btn:
                process_data_btn.config(state=NORMAL)

    def __orca_counting(
        self,
        folder_listbox,
        folder_listbox_o,
        orca_data_val,
        freq_data_val,
    ):
        self.g_counter = GraphletCounter(
            folder_listbox.get(0, END)[0],
            folder_listbox_o.get(0, END)[0] if folder_listbox_o.size() > 0 else None,
            out_data=bool(orca_data_val.get()),
            freq_data=bool(freq_data_val.get()),
        )
        self.orbit_counts_df = self.g_counter.orca_counting()

    def __count_similarities(
        self,
        RGFD_dist_val,
        sim_disp_dist_val,
        hellinger_dist_val,
        minkowski_dist_val,
        cosine_dist_val,
    ):
        self.similarities_df = pd.concat(
            [
                self.similarities_df,
                self.g_counter.count_network_similarities(
                    RGFD_dist_val,
                    sim_disp_dist_val,
                    hellinger_dist_val,
                    minkowski_dist_val,
                    cosine_dist_val,
                ),
            ],
            axis=1,
        )

    def create_main_window(self):
        self.root = TkinterDnD.Tk()
        self.root.title("Networks similarity measures")

        frame = Frame(self.root)
        frame.pack(padx=10, pady=10)

        input_folder_label = Label(frame, text="Input folder")
        input_folder_label.grid(row=0, column=0, sticky=N)

        ouptut_folder_label = Label(frame, text="Output folder")
        ouptut_folder_label.grid(row=0, column=1, sticky=N)

        input_folder_listbox = Listbox(frame, selectmode=EXTENDED, height=1)
        input_folder_listbox.grid(
            row=1, column=0, sticky=NSEW, padx=[0, 5], pady=5, ipady=5
        )
        input_folder_listbox.drop_target_register(DND_FILES)
        input_folder_listbox.dnd_bind(
            "<<Drop>>",
            lambda event: [
                input_folder_listbox.delete(0, END),
                input_folder_listbox.insert(END, event.data),
                process_data_btn.config(state=NORMAL),
            ],
        )

        output_folder_listbox = Listbox(frame, selectmode=EXTENDED, height=1)
        output_folder_listbox.grid(
            row=1, column=1, sticky=NSEW, padx=[5, 0], pady=5, ipady=5
        )
        output_folder_listbox.drop_target_register(DND_FILES)
        output_folder_listbox.dnd_bind(
            "<<Drop>>",
            lambda event: [
                output_folder_listbox.delete(0, END),
                output_folder_listbox.insert(END, event.data),
            ],
        )

        add_folder_btn = Button(
            frame,
            text="Add Input Folder",
            command=lambda: self.__open_folder_dialog(
                input_folder_listbox, process_data_btn
            ),
            width=15,
        )
        add_folder_btn.grid(row=2, column=0, sticky=NSEW)

        remove_folder_btn = Button(
            frame,
            text="Remove Input Folder",
            command=lambda: [
                input_folder_listbox.delete(0),
                process_data_btn.config(state=DISABLED),
            ],
            width=15,
        )
        remove_folder_btn.grid(row=3, column=0, sticky=NSEW)

        add_folder_btn_o = Button(
            frame,
            text="Add Output Folder",
            command=lambda: self.__open_folder_dialog(output_folder_listbox),
            width=15,
        )
        add_folder_btn_o.grid(row=2, column=1, sticky=NSEW)

        remove_folder_btn_o = Button(
            frame,
            text="Remove Output Folder",
            command=lambda: [
                output_folder_listbox.delete(0),
            ],
            width=15,
        )
        remove_folder_btn_o.grid(row=3, column=1, sticky=NSEW)

        orca_data_val = IntVar()
        orca_data_checkbox = Checkbutton(
            frame,
            text="Data are ORCA outputs",
            variable=orca_data_val,
        )
        orca_data_checkbox.grid(row=4, column=0, pady=[5, 0], sticky=W)

        graphlet_freq_data_val = IntVar()
        graphlet_freq_data_checkbox = Checkbutton(
            frame,
            text="Data are graphlets frequencies",
            variable=graphlet_freq_data_val,
        )
        graphlet_freq_data_checkbox.grid(row=4, column=1, pady=[5, 0], sticky=W)

        process_data_btn = Button(
            frame,
            state=DISABLED,
            text="Process folders",
            command=lambda: [
                self.__orca_counting(
                    input_folder_listbox,
                    output_folder_listbox,
                    orca_data_val,
                    graphlet_freq_data_val,
                ),
                graphs_btn.config(
                    state=DISABLED if self.orbit_counts_df is None else NORMAL
                ),
                count_similarities_btn.config(
                    state=DISABLED if self.orbit_counts_df is None else NORMAL
                ),
                display_graphlet_counts_btn.config(
                    state=DISABLED if self.orbit_counts_df is None else NORMAL
                ),
                clustering_btn.config(
                    state=DISABLED if self.orbit_counts_df is None else NORMAL
                ),
                network_btn.config(
                    state=DISABLED if self.orbit_counts_df is None else NORMAL
                ),
            ],
            width=10,
        )
        process_data_btn.grid(row=5, column=0, sticky=NSEW, pady=5)

        graphs_btn = Button(
            frame,
            text="Show graphs",
            state=DISABLED,
            command=lambda: DataVisualiser(
                self.orbit_counts_df
            ).visualize_orbit_counts(),
            width=10,
        )
        graphs_btn.grid(row=5, column=1, sticky=NSEW, pady=5)

        separator = ttk.Separator(frame, orient=HORIZONTAL)
        separator.grid(row=6, column=0, columnspan=4, sticky=NSEW, pady=10)

        dis_measures_label = Label(frame, text="Select distance measures")
        dis_measures_label.grid(row=7, column=0, columnspan=4, sticky=N)

        RGFD_dist_val = IntVar()
        RGFD_dist_val.set(1)
        RGFD_dist_checkbox = Checkbutton(
            frame, text="RGFD distance", variable=RGFD_dist_val
        )
        RGFD_dist_checkbox.grid(row=8, column=0, pady=[5, 0], sticky=W)

        sim_disp_dist_val = IntVar()
        sim_disp_dist_val.set(1)
        sim_disp_dist_checkbox = Checkbutton(
            frame, text="Simple dispersion distance", variable=sim_disp_dist_val
        )
        sim_disp_dist_checkbox.grid(row=8, column=1, pady=[5, 0], sticky=W)

        hellinger_dist_val = IntVar()
        hellinger_dist_val.set(1)
        hellinger_dist_checkbox = Checkbutton(
            frame, text="Hellinger distance", variable=hellinger_dist_val
        )
        hellinger_dist_checkbox.grid(row=9, column=0, sticky=W)

        minkowski_dist_val = IntVar()
        minkowski_dist_val.set(1)
        minkowski_dist_checkbox = Checkbutton(
            frame, text="Minkowski distance", variable=minkowski_dist_val
        )
        minkowski_dist_checkbox.grid(row=9, column=1, sticky=W)

        cosine_dist_val = IntVar()
        cosine_dist_val.set(1)
        cosine_dist_checkbox = Checkbutton(
            frame, text="Cosine distance", variable=cosine_dist_val
        )
        cosine_dist_checkbox.grid(row=10, column=0, sticky=W)

        count_similarities_btn = Button(
            frame,
            text="Count similarities",
            state=DISABLED,
            command=lambda: [
                self.__count_similarities(
                    bool(RGFD_dist_val.get()),
                    bool(sim_disp_dist_val.get()),
                    bool(hellinger_dist_val.get()),
                    bool(minkowski_dist_val.get()),
                    bool(cosine_dist_val.get()),
                ),
                display_similarity_measures_btn.config(
                    state=DISABLED if self.similarities_df is None else NORMAL
                ),
                display_roc_curve_btn.config(
                    state=DISABLED if self.similarities_df is None else NORMAL
                ),
            ],
            width=10,
        )
        count_similarities_btn.grid(row=11, column=0, columnspan=4, sticky=NSEW)

        separator = ttk.Separator(frame, orient=HORIZONTAL)
        separator.grid(row=12, column=0, columnspan=4, sticky=NSEW, pady=10)

        display_graphlet_counts_btn = Button(
            frame,
            text="Display graphlet counts",
            state=DISABLED,
            command=lambda: MeasuresViewer(
                root=self.root,
                graphlet_counts=self.g_counter.get_orbit_counts_df(),
            ).display_graphlet_counts(),
            width=15,
        )
        display_graphlet_counts_btn.grid(row=13, column=0, sticky=NSEW)

        display_similarity_measures_btn = Button(
            frame,
            text="Display similarity values",
            state=DISABLED,
            command=lambda: MeasuresViewer(
                root=self.root,
                similarity_measures=self.similarities_df,
            ).display_similarity_measures(),
            width=15,
        )
        display_similarity_measures_btn.grid(row=13, column=1, sticky=NSEW)

        separator = ttk.Separator(frame, orient=HORIZONTAL)
        separator.grid(row=14, column=0, columnspan=4, sticky=NSEW, pady=10)

        def __networking(self):
            similarity_scores = SiameseNetwork(
                self.g_counter.get_orbit_counts_df()
            ).predict_similarity()
            self.similarities_df["NeuralNetwork"] = similarity_scores

            display_similarity_measures_btn.config(
                state=DISABLED if self.similarities_df is None else NORMAL
            ),

        network_btn = Button(
            frame,
            text="Neural network",
            state=DISABLED,
            command=lambda: __networking(self),
            width=15,
        )
        network_btn.grid(row=15, column=0, sticky=NSEW)

        def __clustering(self):
            self.similarities_df = DataClustering(
                input_df=self.g_counter.get_orbit_counts_df(),
                similarity_measures_df=self.similarities_df,
            ).clustering()

            display_similarity_measures_btn.config(
                state=DISABLED if self.similarities_df is None else NORMAL
            ),

        clustering_btn = Button(
            frame,
            text="Clustering",
            state=DISABLED,
            command=lambda: __clustering(self),
            width=15,
        )
        clustering_btn.grid(row=15, column=1, sticky=NSEW)

        separator = ttk.Separator(frame, orient=HORIZONTAL)
        separator.grid(row=16, column=0, columnspan=4, sticky=NSEW, pady=10)

        display_roc_curve_btn = Button(
            frame,
            text="Display ROC curve",
            state=DISABLED,
            command=lambda: ROCCurveVisualiser(
                input_df=self.similarities_df
            ).generate_roc_curve(),
            width=15,
        )
        display_roc_curve_btn.grid(row=17, column=0, columnspan=4, sticky=NSEW)

        self.root.mainloop()
