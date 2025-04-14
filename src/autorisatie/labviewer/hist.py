import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output
import os
import plotly.express as px
import pandas as pd
import plotly
import dash_bootstrap_components as dbc

plotly.io.json.config.default_engine = 'orjson'

@dash.callback(
    Output('memory-graph', 'figure'),
    [Input('memory-output', 'data'),
     Input('test-dropdown', 'value')],
    prevent_initial_call=True
)
def hist(data, test_str):
    if data is None or not test_str:
        return no_update
    
    df = pd.DataFrame(data)
    
    # Create histogram with optimized settings
    fig = px.histogram(
        df, 
        x=test_str,
        nbins=50,  # Optimize number of bins
        template='plotly_white'  # Lighter template for faster rendering
    )
    
    # Optimize the layout
    fig.update_layout(
        title=f"Distribution of {test_str}",
        showlegend=False,  # Disable legend if not needed
        margin=dict(l=40, r=40, t=60, b=40),  # Optimize margins
        plot_bgcolor='white'
    )
    
    return fig

@dash.callback(
    Output('test-dropdown', 'options'),
    Input('memory-output', 'data'),
    prevent_initial_call=True
)
def dropdown_check(data):
    if data is None:
        return no_update
    
    df = pd.DataFrame(data)
    # Only include numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    return sorted(list(numeric_cols))

def page():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H4("Distribution Analysis", className="mb-4"),
                html.P(
                    "Analyze the distribution of individual laboratory measurements.",
                    className="text-muted mb-4"
                ),
                dbc.Card([
                    dbc.CardHeader("Select Laboratory Test"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dcc.Dropdown(
                                    id='test-dropdown',
                                    placeholder="Select a lab test to analyze",
                                    className="mb-3"
                                )
                            ])
                        ])
                    ])
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("Distribution Plot"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-hist",
                            type="circle",
                            children=dcc.Graph(
                                id='memory-graph',
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                                }
                            )
                        )
                    ])
                ])
            ], width=12)
        ])
    ], fluid=True, className="py-3")
