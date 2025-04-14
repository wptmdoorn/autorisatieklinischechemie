import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import os, sqlite3
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc

# list all files in data clean and construct options for checklist
try:
    file_options = [
        {"label": fi,
         "value": f"../data/clean/{fi}"} for fi in os.listdir("../data/clean")
    ]
    print("Available files:", file_options)
except Exception as e:
    print(f"Error loading file options: {str(e)}")
    file_options = []

def connect_db(db_path: str) -> sqlite3.Connection:
    """Create a database connection."""
    try:
        print(f'loading db... {db_path}')
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda x: x.decode(errors='ignore')  # Ignore decoding errors
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {str(e)}")
        raise

def page():
    return dbc.Container([
        # Settings Section
        dbc.Row([
            dbc.Col([
                html.H4("Settings", className="mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Switch(
                                    id="test-mode-switch",
                                    label="Test Mode (Load first 1000 rows only)",
                                    value=False
                                ),
                                html.Small(
                                    "Enable this to load a smaller dataset for testing purposes.",
                                    className="text-muted d-block mt-2"
                                )
                            ])
                        ])
                    ])
                ], className="mb-4")
            ], width=12)
        ]),

        # Data Loading Section
        dbc.Row([
            dbc.Col([
                html.H4("Data Source", className="mb-4"),
                html.P(
                    "Select the database files to analyze:",
                    className="text-muted"
                ),
                dbc.Card([
                    dbc.CardBody([
                        dcc.Checklist(
                            id="checklist",
                            options=file_options,
                            labelStyle={"display": "block", "margin": "10px 0"}
                        ),
                        dbc.Button(
                            "Load Data",
                            id="load-data-button",
                            color="primary",
                            className="mt-3"
                        )
                    ])
                ], className="mb-4"),
                dcc.Loading(
                    id="loading-status",
                    type="circle",
                    children=html.Div(
                        id="data-load-status",
                        className="mt-3"
                    )
                )
            ], width=12)
        ])
    ], fluid=True, className="py-3")

@dash.callback(
    [Output("memory-output", "data"),
     Output("data-load-status", "children")],
    [Input("load-data-button", "n_clicks")],
    [State("checklist", "value"),
     State("test-mode-switch", "value")],
    prevent_initial_call=True
)
def load_data(n_clicks, selected_files, test_mode):
    if not selected_files:
        return None, dbc.Alert(
            [
                html.I(className="fas fa-exclamation-circle me-2"),
                "Please select at least one database file."
            ],
            color="warning",
            className="mt-3"
        )
    
    try:
        all_dfs = []
        total_records = 0
        
        for db_path in selected_files:
            print('loading data from:', db_path)
            conn = connect_db(db_path)
            
            # Step 1: Retrieve data from wide_blood_draws with specific columns
            query = """
            SELECT Patientnummer, DrawCount, Labnummer, 
                   AFSE, ALTSE, CHOSE, GGTSE, HB, KALSE, KRESE, LEUC, 
                   NATSE, TRISE, ALBSE, ASTSE, CRPSE, TRC
            FROM wide_blood_draws
            """
            
            # Add LIMIT clause if test mode is enabled
            if test_mode:
                query += " LIMIT 1000"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            print(f"Loaded {len(df)} records from {db_path}")
            total_records += len(df)
            all_dfs.append(df)

        # Concatenate all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Convert to numeric, handling errors
        numeric_columns = ['AFSE', 'ALTSE', 'CHOSE', 'GGTSE', 'HB', 'KALSE', 'KRESE', 'LEUC', 
                          'NATSE', 'TRISE', 'ALBSE', 'ASTSE', 'CRPSE', 'TRC']
        combined_df[numeric_columns] = combined_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Optimize memory usage
        for col in numeric_columns:
            combined_df[col] = pd.to_numeric(combined_df[col], downcast='float')
        
        records = combined_df.to_dict('records')
        print(f"Processed data into {len(records)} total records")
        
        # Add test mode indicator to status message
        mode_text = " (Test Mode)" if test_mode else ""
        return records, dbc.Alert(
            [
                html.I(className="fas fa-check-circle me-2"),
                f"Successfully loaded {total_records} records from {len(selected_files)} database(s){mode_text}"
            ],
            color="success",
            className="mt-3"
        )
        
    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        return None, dbc.Alert(
            [
                html.I(className="fas fa-exclamation-circle me-2"),
                f"Error loading data: {str(e)}"
            ],
            color="danger",
            className="mt-3"
        )

# Add CSS for custom radio styling
custom_radio_css = """
<style>
.custom-radio .form-check {
    padding: 10px;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    margin-bottom: 8px;
}
.custom-radio .form-check:hover {
    background-color: #f8f9fa;
}
</style>
"""
