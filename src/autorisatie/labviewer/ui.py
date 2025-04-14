# Import packages
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from . import settings, hist, correlation, review

# Initialise the App
app = Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="MUMC Lab Analysis Dashboard"
)

# Header Components
header = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("MUMC Lab Analysis Dashboard", 
                   className="display-4 text-primary mb-3",
                   style={"font-weight": "bold"}),
            html.P(
                "A comprehensive tool for analyzing and visualizing laboratory test results. "
                "This dashboard provides statistical insights, historical trends, and correlation "
                "analysis for laboratory measurements.",
                className="lead text-muted mb-4"
            ),
            html.Hr(className="my-4")
        ], width=12)
    ])
], fluid=True, className="py-3 bg-light")

# App Layout
app.layout = dbc.Container(
    [
        dcc.Store(id='memory-output'),
        header,
        dbc.Tabs(
            [
                dbc.Tab(
                    dbc.Card(
                        settings.page(),
                        className="border-0 rounded-0"
                    ),
                    label="Data Settings",
                    tab_style={"marginLeft": "0px"},
                    label_style={"color": "#2C3E50"},
                    active_label_style={"color": "#2C3E50", "font-weight": "bold"},
                ),
                dbc.Tab(
                    dbc.Card(
                        hist.page(),
                        className="border-0 rounded-0"
                    ),
                    label="Distribution Analysis",
                    label_style={"color": "#2C3E50"},
                    active_label_style={"color": "#2C3E50", "font-weight": "bold"},
                ),
                dbc.Tab(
                    dbc.Card(
                        correlation.page(),
                        className="border-0 rounded-0"
                    ),
                    label="Correlation Analysis",
                    label_style={"color": "#2C3E50"},
                    active_label_style={"color": "#2C3E50", "font-weight": "bold"},
                ),
                dbc.Tab(
                    dbc.Card(
                        review.page(),
                        className="border-0 rounded-0"
                    ),
                    label="Order Review",
                    label_style={"color": "#2C3E50"},
                    active_label_style={"color": "#2C3E50", "font-weight": "bold"},
                ),
            ],
            className="mt-4"
        )
    ],
    fluid=True,
    className="px-4"
)

# Run the App
if __name__ == "__main__":
    app.run(debug=True)