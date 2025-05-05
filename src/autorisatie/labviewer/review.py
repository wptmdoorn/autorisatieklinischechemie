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
from ..model.autoencoder.inference import AutoencoderPredictor
from ..model.correlation.inference import CorrelationPredictor
from ..model.vae.inference import VAEPredictor
import pickle
from .utils.config import load_test_config

# Load test configuration
lab_columns, test_display_names = load_test_config()

# Cache the model and scaler loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {}

# Dictionary mapping model names to their predictor classes
model_dict = {
    'Autoencoder': AutoencoderPredictor,
    'Correlation': CorrelationPredictor,
    'VAE': VAEPredictor
}

scalers = {}

def load_models():
    """Load all available models and their scalers"""
    global models, scalers
    
    # Initialize models dictionary
    models = {}
    
    # Load each model type
    for model_name, predictor_class in model_dict.items():
        try:
            # Initialize predictor
            predictor = predictor_class()
            models[model_name] = predictor
            print(f"Successfully loaded {model_name} model")
        except Exception as e:
            print(f"Error loading {model_name} model: {str(e)}")
    
    return models

# Load models at startup
models = load_models()

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
                html.Label(test_display_names[col], className="fw-bold"),
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
        if model_name in models:
            try:
                results = models[model_name].predict(data)
                model_scores[model_name] = {
                    'test_scores': results['test_scores'][0],
                    'total_score': results['total_score'][0]
                }
            except Exception as e:
                print(f"Error making prediction with {model_name}: {str(e)}")
                continue
    
    # Create probability displays for each test
    prob_displays = []
    for id_obj in ids:
        col = id_obj['index']
        if col in lab_columns:
            idx = lab_columns.index(col)
            model_badges = []
            for model_name, scores in model_scores.items():
                prob = scores['test_scores'][idx]
                if np.isnan(prob):
                    model_badges.append(dbc.Badge(f"{model_name}: N/A", color="secondary", className="ms-1"))
                else:
                    color = "success" if prob >= 0.7 else "warning" if prob >= 0.4 else "danger"
                    model_badges.append(dbc.Badge(f"{model_name}: {prob*100:.1f}%", color=color, className="ms-1"))
            prob_displays.append(html.Div(model_badges))
        else:
            prob_displays.append(html.Span(""))
    
    # Create feasibility display
    feasibility_display = [
        html.H4("Feasibility Analysis", className="text-center mb-3"),
        html.Div([
            html.Div([
                html.H5(f"{model_name}", className="text-center mb-2"),
                html.H1(f"{scores['total_score']*100:.1f}%" if not np.isnan(scores['total_score']) else "N/A", 
                       className="text-center display-4 mb-3"),
            ], className="mb-4") for model_name, scores in model_scores.items()
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
                                        {'label': model_name, 'value': model_name}
                                        for model_name in model_dict.keys()
                                    ],
                                    value=list(model_dict.keys())[:1],  # Default to first model
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