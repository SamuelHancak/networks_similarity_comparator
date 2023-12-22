import plotly.graph_objects as go

CATEGORIES = [
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "G6",
    "G7",
    "G8",
    "G9",
    "G10",
    "G11",
    "G12",
    "G13",
    "G14",
    "G15",
    "G16",
    "G17",
    "G18",
    "G19",
    "G20",
    "G21",
    "G22",
    "G23",
    "G24",
    "G25",
    "G26",
    "G27",
    "G28",
    "G29",
    "G30",
]


class DataVisualiser:
    def __init__(self, orbit_counts):
        self.orbit_counts = orbit_counts
        self.categories = CATEGORIES
        self.fig = None

    def create_polar_scatter(self):
        self.fig = go.Figure()

        for col in self.orbit_counts.columns:
            self.fig.add_trace(
                go.Scatterpolar(
                    r=self.orbit_counts[col],
                    theta=self.categories,
                    fill="toself",
                    name=col,
                )
            )

            self.fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
                showlegend=True,
            )

        self.fig.show()

    def create_bar_chart(self):
        self.fig = go.Figure()

        for col in self.orbit_counts:
            self.fig.add_trace(
                go.Bar(x=self.categories, y=self.orbit_counts[col], name=col)
            )
            self.fig.update_layout(
                barmode="group",
                xaxis_title="Categories",
                yaxis_title="Values",
                showlegend=True,
            )

        self.fig.show()

    def create_line_chart(self):
        self.fig = go.Figure()

        for col in self.orbit_counts:
            self.fig.add_trace(
                go.Scatter(
                    x=self.categories,
                    y=self.orbit_counts[col],
                    mode="lines+markers",
                    name=col,
                )
            )

        self.fig.update_layout(
            xaxis_title="Categories",
            yaxis_title="Values",
            showlegend=True,
        )
        self.fig.update_yaxes(type="log")

        self.fig.show()

    def visualize_orbit_counts(self):
        # self.create_polar_scatter()
        self.create_bar_chart()
        self.create_line_chart()
