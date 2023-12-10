from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from modules.DataVisualizer import DataVisualizer
from modules.GraphletCounter import GraphletCounter
from modules.MeasuresViewer import MeasuresViewer


class GUI:
    def __init__(self):
        self.root = None
        self.orbit_counts_df = None
        self.similarities_df = None
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
        normalisation_val,
    ):
        self.g_counter = GraphletCounter(
            folder_listbox.get(0, END)[0],
            folder_listbox_o.get(0, END)[0] if folder_listbox_o.size() > 0 else None,
            out_data=bool(orca_data_val.get()),
            normalise=bool(normalisation_val.get()),
        )
        self.orbit_counts_df = self.g_counter.orca_counting()

    def __count_similarities(
        self,
        RGFD_dist_val,
        sim_disp_dist_val,
        hellinger_dist_val,
        minkowski_dist_val,
    ):
        self.similarities_df = self.g_counter.count_network_similarities(
            RGFD_dist_val,
            sim_disp_dist_val,
            hellinger_dist_val,
            minkowski_dist_val,
        )

    def create_main_window(self):
        self.root = TkinterDnD.Tk()
        self.root.title("Graph similarity measures")

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

        normalisation_val = IntVar()
        normalisation_checkbox = Checkbutton(
            frame, text="Normalise data", variable=normalisation_val
        )
        normalisation_checkbox.grid(row=4, column=1, pady=[5, 0], sticky=W)

        process_data_btn = Button(
            frame,
            state=DISABLED,
            text="Process folders",
            command=lambda: [
                self.__orca_counting(
                    input_folder_listbox,
                    output_folder_listbox,
                    orca_data_val,
                    normalisation_val,
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
            ],
            width=10,
        )
        process_data_btn.grid(row=5, column=0, sticky=NSEW, pady=5)

        graphs_btn = Button(
            frame,
            text="Show graphs",
            state=DISABLED,
            command=lambda: DataVisualizer(
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
                ),
                display_similarity_measures_btn.config(
                    state=DISABLED if self.similarities_df is None else NORMAL
                ),
            ],
            width=10,
        )
        count_similarities_btn.grid(row=11, column=0, columnspan=4)

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

        self.root.mainloop()