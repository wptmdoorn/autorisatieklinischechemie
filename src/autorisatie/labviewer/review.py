import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly
import json
import torch
from sklearn.preprocessing import StandardScaler
import os
from ..model.autoencoder.model import get_pseudo_probability_score_per_test, Autoencoder, get_feasibility_score
import pickle

# Cache the model and scaler loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {}
scalers = {}

def load_models():
    """Load all available models and their scalers"""
    global models, scalers
    
    # Autoencoder model 1
    if 'autoencoder1' not in models:
        try:
            model = Autoencoder(input_dim=14, encoding_dim=10)
            model_path = os.path.join('..', 'out', 'model', 'autoencoder_lab_model.pth')
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                models['autoencoder1'] = model
                print("Successfully loaded autoencoder model 1")
            else:
                print("Warning: Autoencoder model 1 file not found at", model_path)
        except Exception as e:
            print(f"Error loading autoencoder model 1: {str(e)}")
    
    if 'autoencoder1' not in scalers:
        try:
            with open('../out/model/autoencoder_scaler.pkl', 'rb') as f:
                scalers['autoencoder1'] = pickle.load(f)
        except Exception as e:
            print(f"Error loading autoencoder scaler 1: {str(e)}")
    
    # Autoencoder model 2 (placeholder)
    if 'autoencoder2' not in models:
        try:
            model = Autoencoder(input_dim=14, encoding_dim=10)
            model_path = os.path.join('..', 'out', 'model', 'autoencoder_lab_model.pth')
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                models['autoencoder2'] = model
                print("Successfully loaded autoencoder model 2")
            else:
                print("Warning: Autoencoder model 2 file not found at", model_path)
        except Exception as e:
            print(f"Error loading autoencoder model 2: {str(e)}")
    
    if 'autoencoder2' not in scalers:
        try:
            with open('../out/model/autoencoder_scaler.pkl', 'rb') as f:
                scalers['autoencoder2'] = pickle.load(f)
        except Exception as e:
            print(f"Error loading autoencoder scaler 2: {str(e)}")
    
    return models, scalers

# Load models at startup
models, scalers = load_models()

lab_columns = ['AFSE', 'ALTSE', 'CHOSE', 'GGTSE', 'HB', 'KALSE', 'KRESE', 'LEUC', 'NATSE', 'TRISE', 
               'ALBSE', 'ASTSE', 'CRPSE', 'TRC']
metadata_fields = ['Patientnummer', 'DrawCount', 'Labnummer']

# Prevent callback retriggers for unchanged data
@dash.callback(
    [Output('current-order', 'data'),
     Output('order-display', 'children'),
     Output('current-index', 'data')],
    [Input('next-button', 'n_clicks'),
     Input('prev-button', 'n_clicks'),
     Input('memory-output', 'data')],
    [State('current-index', 'data')],
    prevent_initial_call=True
)
def update_order(next_clicks, prev_clicks, data, current_idx):
    if data is None:
        return no_update, no_update, no_update
    
    # Initialize current index if None
    if current_idx is None:
        current_idx = 0
    
    # Determine which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Update index based on button click
    if button_id == 'next-button' and current_idx < len(data) - 1:
        current_idx += 1
    elif button_id == 'prev-button' and current_idx > 0:
        current_idx -= 1
    
    # Get current order
    current_order = data[current_idx]
    
    # Create metadata display
    metadata_display = dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Order Information", className="mb-3 text-muted"),
                    html.Table(
                        [html.Tr([
                            html.Td(html.Strong(field + ": "), 
                                  style={"paddingRight": "15px"}),
                            html.Td(str(current_order.get(field, "N/A")))
                        ]) for field in metadata_fields],
                        className="table table-sm table-borderless"
                    )
                ])
            ])
        ]),
        className="mb-3"
    )
    
    # Create input fields for lab values
    input_fields = [
        dbc.Row([
            dbc.Col(
                html.Label(col, className="fw-bold"),
                width=3
            ),
            dbc.Col(
                dbc.Input(
                    type="number",
                    value=current_order.get(col),
                    id={'type': 'lab-input', 'index': col},
                    className="mb-2",
                    debounce=True
                ),
                width=6
            ),
            dbc.Col(
                html.Div(id={'type': 'prob-display', 'index': col},
                        className="text-center"),
                width=3
            )
        ], className="mb-2 align-items-center")
        for col in lab_columns if col in current_order and pd.notna(current_order[col])
    ]
    
    # Combine metadata and input fields
    display_elements = [
        metadata_display,
        html.H6("Lab Values", className="mb-3"),
        *input_fields
    ]
    
    return current_order, display_elements, current_idx

@dash.callback(
    [Output({'type': 'prob-display', 'index': ALL}, 'children'),
     Output('feasibility-score', 'children')],
    [Input({'type': 'lab-input', 'index': ALL}, 'value'),
     Input('model-checklist', 'value')],
    [State({'type': 'lab-input', 'index': ALL}, 'id')],
    prevent_initial_call=True
)
def calculate_feasibility(values, selected_models, ids):
    if not values or not ids or not selected_models:
        return [no_update] * len(ids), no_update
    
    # Create data array for model
    data = np.full((1, len(lab_columns)), -1, dtype=float)
    value_dict = {id_obj['index']: val for id_obj, val in zip(ids, values) if val is not None}
    
    for i, col in enumerate(lab_columns):
        if col in value_dict:
            try:
                data[0, i] = float(value_dict[col])
            except (ValueError, TypeError):
                continue
    
    # Calculate probabilities for each selected model
    model_scores = {}
    for model_name in selected_models:
        if model_name in models and model_name in scalers:
            with torch.no_grad():
                prob_scores, _ = get_pseudo_probability_score_per_test(
                    data, models[model_name], scalers[model_name], device
                )
                model_scores[model_name] = prob_scores[0]
    
    # Create probability displays for each test
    prob_displays = []
    for id_obj in ids:
        col = id_obj['index']
        if col in lab_columns:
            idx = lab_columns.index(col)
            model_badges = []
            for model_name, scores in model_scores.items():
                prob = scores[idx]
                if np.isnan(prob):
                    model_badges.append(dbc.Badge(f"{model_name[-1]}: N/A", color="secondary", className="ms-1"))
                else:
                    color = "success" if prob >= 0.7 else "warning" if prob >= 0.4 else "danger"
                    model_badges.append(dbc.Badge(f"{model_name[-1]}: {prob*100:.1f}%", color=color, className="ms-1"))
            prob_displays.append(html.Div(model_badges))
        else:
            prob_displays.append(html.Span(""))
    
    # Calculate overall feasibility scores
    feasibility_scores = {}
    for model_name, scores in model_scores.items():
        overall_score = get_feasibility_score(data, models[model_name], scalers[model_name], device)[0][0] * 100
        feasibility_scores[model_name] = overall_score
    
    # Create feasibility display
    feasibility_display = [
        html.H4("Feasibility Analysis", className="text-center mb-3"),
        html.Div([
            html.Div([
                html.H5(f"Model {model_name[-1]}", className="text-center mb-2"),
                html.H1(f"{score:.1f}%", className="text-center display-4 mb-3"),
            ], className="mb-4") for model_name, score in feasibility_scores.items()
        ]),
        html.Div([
            html.P("Probability Legend:", className="mb-2"),
            dbc.Badge("≥ 70%", color="success", className="me-2"),
            dbc.Badge("40-70%", color="warning", className="me-2"),
            dbc.Badge("< 40%", color="danger"),
        ], className="text-center mb-3"),
        html.P("Based on selected model predictions",
              className="text-muted text-center")
    ]
    
    return prob_displays, feasibility_display

def page():
    return dbc.Container([
        dcc.Store(id='current-order'),
        dcc.Store(id='current-index'),
        
        dbc.Row([
            dbc.Col([
                html.H4("Order Review", className="mb-4"),
                html.P(
                    "Review and analyze laboratory orders using multiple feasibility models.",
                    className="text-muted mb-4"
                ),
                
                # Tabs for different views
                dbc.Tabs([
                    # Tab 1: Settings
                    dbc.Tab([
                        dbc.Card([
                            dbc.CardHeader("Model Selection"),
                            dbc.CardBody([
                                dbc.Checklist(
                                    id='model-checklist',
                                    options=[
                                        {'label': 'Autoencoder Model 1', 'value': 'autoencoder1'},
                                        {'label': 'Autoencoder Model 2', 'value': 'autoencoder2'}
                                    ],
                                    value=['autoencoder1'],  # Default to first model
                                    inline=True,
                                    className="mb-3"
                                ),
                                html.P("Select which models to use for feasibility analysis",
                                      className="text-muted small")
                            ])
                        ], className="mb-4")
                    ], label="Settings", tab_id="tab-settings"),
                    
                    # Tab 2: Review
                    dbc.Tab([
                        dbc.Row([
                            # Left column - Lab Values
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader([
                                        html.H4("Lab Order Review", className="mb-0"),
                                        html.P("Edit values to see real-time feasibility", 
                                              className="text-muted mb-0 small")
                                    ]),
                                    dbc.CardBody([
                                        dcc.Loading(
                                            id="loading-review",
                                            type="circle",
                                            children=html.Div(id='order-display')
                                        ),
                                        html.Div([
                                            dbc.Button("Previous", id='prev-button', 
                                                     color="secondary", className="me-2"),
                                            dbc.Button("Next", id='next-button', 
                                                     color="secondary"),
                                        ], className="mt-3 d-flex justify-content-between")
                                    ])
                                ])
                            ], width=8),
                            
                            # Right column - Feasibility Score
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        dcc.Loading(
                                            id="loading-feasibility",
                                            type="circle",
                                            children=html.Div(id='feasibility-score')
                                        )
                                    ])
                                ])
                            ], width=4)
                        ], className="mt-4")
                    ], label="Review", tab_id="tab-review")
                ], id="review-tabs", active_tab="tab-settings")
            ], width=12)
        ])
    ], fluid=True, className="py-3") 