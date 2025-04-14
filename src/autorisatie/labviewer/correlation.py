import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import plotly
from scipy.stats import linregress
from scipy.optimize import curve_fit
import dash_bootstrap_components as dbc
import numpy as np

plotly.io.json.config.default_engine = 'orjson'

# Callback to update the scatter plot with correlation and regression
@dash.callback(
    Output('corr-graph', 'figure'),
    [Input('memory-output', 'data'),
     Input('x-dropdown', 'value'),
     Input('y-dropdown', 'value')],
    prevent_initial_call=True
)
def correlation_plot(data, x_var, y_var):
    if data is None or x_var is None or y_var is None:
        return no_update
    
    df = pd.DataFrame(data)
    
    if x_var not in df.columns or y_var not in df.columns:
        return no_update
    
    df = df[[x_var, y_var]].apply(pd.to_numeric, errors='coerce').dropna()
    x = df[x_var]
    y = df[y_var]
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value**2
    n = len(df)
    line_eq = f"y = {slope:.3f}x + {intercept:.3f}"

    # Create scatter plot with regression line
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data Points'))
    fig.add_trace(go.Scatter(x=x, y=slope * x + intercept, mode='lines', 
                            name='Regression Line',
                            line=dict(color='red', dash='dash')))

    # Add annotation with statistics
    stats_text = (f"R = {r_value:.3f}<br>"
                 f"R² = {r_squared:.3f}<br>"
                 f"n = {n}<br>"
                 f"{line_eq}")

    fig.update_layout(
        title=f"Correlation between {x_var} and {y_var}",
        xaxis_title=x_var,
        yaxis_title=y_var,
        plot_bgcolor='white',
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref='paper',
                yref='paper',
                text=stats_text,
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1,
                borderpad=10,
                font=dict(size=12)
            )
        ]
    )
    
    return fig

@dash.callback(
    Output('correlation-matrix', 'figure'),
    Input('memory-output', 'data'),
    prevent_initial_call=True
)
def update_correlation_matrix(data):
    if data is None:
        return no_update
    
    df = pd.DataFrame(data)
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    df_numeric = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Calculate correlation matrix
    corr_matrix = df_numeric.corr()
    
    # Create correlation matrix plot
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        zmin=-1,
        zmax=1,
        colorscale='RdBu',
        text=np.round(corr_matrix.values, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
    ))
    
    # Update layout
    fig.update_layout(
        title="Correlation Matrix (R values)<br><sub>Click any cell to see detailed correlation plot</sub>",
        height=800,
        width=800,
        plot_bgcolor='white',
        xaxis={'side': 'bottom', 'tickangle': 45},
        yaxis={'autorange': 'reversed'},
    )
    
    return fig

# Callback to populate dropdowns with column names
@dash.callback(
    [Output('x-dropdown', 'options'),
     Output('y-dropdown', 'options')],
    Input('memory-output', 'data'),
    prevent_initial_call=True
)
def update_dropdowns(data):
    if data is None:
        return no_update, no_update
    df = pd.DataFrame(data)
    # Only include numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    options = sorted(list(numeric_cols))
    return options, options

# New callback to handle matrix clicks
@dash.callback(
    [Output('correlation-tabs', 'active_tab'),
     Output('x-dropdown', 'value'),
     Output('y-dropdown', 'value')],
    Input('correlation-matrix', 'clickData'),
    prevent_initial_call=True
)
def handle_matrix_click(clickData):
    if clickData is None:
        return no_update, no_update, no_update
    
    # Extract x and y variables from the click data
    x_var = clickData['points'][0]['x']
    y_var = clickData['points'][0]['y']
    
    # Don't update if clicked on same variable (diagonal)
    if x_var == y_var:
        return no_update, no_update, no_update
    
    # Switch to pairwise analysis tab and update dropdowns
    return "tab-pairwise", x_var, y_var

# Model fitting functions
def fit_models(x, y):
    models = {}
    x_clean = x[~np.isnan(y)]
    y_clean = y[~np.isnan(y)]
    
    if len(x_clean) < 2:
        return {}
    
    try:
        # Linear: y = ax + b
        slope, intercept, r_value, _, _ = linregress(x_clean, y_clean)
        models['Linear'] = {
            'r_squared': r_value**2,
            'equation': f'y = {slope:.3f}x + {intercept:.3f}',
            'predict': lambda x: slope * x + intercept
        }
        
        # Polynomial degree 2: y = ax² + bx + c
        coef = np.polyfit(x_clean, y_clean, 2)
        y_pred = np.polyval(coef, x_clean)
        r2 = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
        models['Polynomial (2)'] = {
            'r_squared': r2,
            'equation': f'y = {coef[0]:.3f}x² + {coef[1]:.3f}x + {coef[2]:.3f}',
            'predict': lambda x: coef[0] * x**2 + coef[1] * x + coef[2]
        }
        
        # Polynomial degree 3: y = ax³ + bx² + cx + d
        coef = np.polyfit(x_clean, y_clean, 3)
        y_pred = np.polyval(coef, x_clean)
        r2 = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
        models['Polynomial (3)'] = {
            'r_squared': r2,
            'equation': f'y = {coef[0]:.3f}x³ + {coef[1]:.3f}x² + {coef[2]:.3f}x + {coef[3]:.3f}',
            'predict': lambda x: coef[0] * x**3 + coef[1] * x**2 + coef[2] * x + coef[3]
        }
        
        # Exponential: y = ae^(bx)
        def exp_func(x, a, b):
            return a * np.exp(b * x)
        
        popt, _ = curve_fit(exp_func, x_clean, y_clean, p0=[1, 0.1])
        y_pred = exp_func(x_clean, *popt)
        r2 = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
        models['Exponential'] = {
            'r_squared': r2,
            'equation': f'y = {popt[0]:.3f}e^({popt[1]:.3f}x)',
            'predict': lambda x: exp_func(x, *popt)
        }
        
        # Logarithmic: y = a*ln(x) + b
        def log_func(x, a, b):
            return a * np.log(x) + b
        
        # Only fit if all x values are positive
        if np.all(x_clean > 0):
            popt, _ = curve_fit(log_func, x_clean, y_clean)
            y_pred = log_func(x_clean, *popt)
            r2 = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
            models['Logarithmic'] = {
                'r_squared': r2,
                'equation': f'y = {popt[0]:.3f}ln(x) + {popt[1]:.3f}',
                'predict': lambda x: log_func(x, *popt)
            }
        
        # Power law: y = ax^b
        def power_func(x, a, b):
            return a * np.power(x, b)
        
        # Only fit if all x and y values are positive
        if np.all(x_clean > 0) and np.all(y_clean > 0):
            popt, _ = curve_fit(power_func, x_clean, y_clean)
            y_pred = power_func(x_clean, *popt)
            r2 = 1 - np.sum((y_clean - y_pred)**2) / np.sum((y_clean - np.mean(y_clean))**2)
            models['Power Law'] = {
                'r_squared': r2,
                'equation': f'y = {popt[0]:.3f}x^{popt[1]:.3f}',
                'predict': lambda x: power_func(x, *popt)
            }
            
    except Exception as e:
        print(f"Error fitting models: {str(e)}")
    
    return models

# Callback for advanced regression analysis
@dash.callback(
    [Output('model-comparison-table', 'children'),
     Output('model-comparison-plot', 'figure')],
    [Input('memory-output', 'data'),
     Input('x-dropdown-advanced', 'value'),
     Input('y-dropdown-advanced', 'value')],
    prevent_initial_call=True
)
def update_model_comparison(data, x_var, y_var):
    if data is None or x_var is None or y_var is None:
        return no_update, no_update
    
    df = pd.DataFrame(data)
    if x_var not in df.columns or y_var not in df.columns:
        return no_update, no_update
    
    # Prepare data
    df = df[[x_var, y_var]].apply(pd.to_numeric, errors='coerce').dropna()
    x = df[x_var].values
    y = df[y_var].values
    
    # Fit all models
    models = fit_models(x, y)
    if not models:
        return "Unable to fit models to this data.", no_update
    
    # Sort models by R-squared
    sorted_models = sorted(models.items(), key=lambda x: x[1]['r_squared'], reverse=True)
    
    # Create table
    table_header = [
        html.Thead(html.Tr([
            html.Th("Model Type"),
            html.Th("R²"),
            html.Th("Equation")
        ]))
    ]
    
    table_body = [html.Tbody([
        html.Tr([
            html.Td(model_name),
            html.Td(f"{model_info['r_squared']:.4f}"),
            html.Td(model_info['equation'])
        ]) for model_name, model_info in sorted_models
    ])]
    
    table = dbc.Table(table_header + table_body, bordered=True, hover=True, striped=True)
    
    # Create plot with top 5 models
    fig = go.Figure()
    
    # Add scatter plot of actual data
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        name='Data Points',
        marker=dict(size=6)
    ))
    
    # Add lines for top 5 models
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    x_range = np.linspace(min(x), max(x), 200)
    
    for (model_name, model_info), color in zip(sorted_models[:5], colors):
        try:
            y_pred = model_info['predict'](x_range)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_pred,
                mode='lines',
                name=f"{model_name} (R²={model_info['r_squared']:.4f})",
                line=dict(color=color, dash='dash')
            ))
        except Exception as e:
            print(f"Error plotting {model_name}: {str(e)}")
    
    fig.update_layout(
        title=f"Model Comparison for {x_var} vs {y_var}",
        xaxis_title=x_var,
        yaxis_title=y_var,
        plot_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    return table, fig

# The layout of the page
def page():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H4("Correlation Analysis", className="mb-4"),
                html.P(
                    "Analyze relationships between laboratory measurements using correlation analysis.",
                    className="text-muted mb-4"
                ),
                
                # Tabs for different correlation views
                dbc.Tabs([
                    # Tab 1: Pairwise Analysis
                    dbc.Tab([
                        dbc.Card([
                            dbc.CardHeader("Select Variables"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("X Variable:"),
                                        dcc.Dropdown(
                                            id='x-dropdown',
                                            placeholder="Select first variable",
                                            className="mb-3"
                                        )
                                    ], md=6),
                                    dbc.Col([
                                        html.Label("Y Variable:"),
                                        dcc.Dropdown(
                                            id='y-dropdown',
                                            placeholder="Select second variable",
                                            className="mb-3"
                                        )
                                    ], md=6),
                                ])
                            ])
                        ], className="mb-4"),
                        
                        dbc.Card([
                            dbc.CardHeader("Correlation Plot"),
                            dbc.CardBody([
                                dcc.Loading(
                                    id="loading-correlation",
                                    type="circle",
                                    children=dcc.Graph(
                                        id='corr-graph',
                                        config={
                                            'displayModeBar': True,
                                            'displaylogo': False,
                                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                                        }
                                    )
                                )
                            ])
                        ])
                    ], label="Pairwise Analysis", tab_id="tab-pairwise"),
                    
                    # Tab 2: Correlation Matrix
                    dbc.Tab([
                        dbc.Card([
                            dbc.CardHeader("All Pairwise Correlations"),
                            dbc.CardBody([
                                html.P([
                                    "This matrix shows the correlation coefficient (R) between all pairs of measurements. ",
                                    html.Strong("Click any cell to see the detailed correlation plot.")
                                ], className="text-muted mb-3"),
                                dcc.Loading(
                                    id="loading-matrix",
                                    type="circle",
                                    children=dcc.Graph(
                                        id='correlation-matrix',
                                        config={
                                            'displayModeBar': True,
                                            'displaylogo': False,
                                            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                            'toImageButtonOptions': {
                                                'format': 'png',
                                                'filename': 'correlation_matrix',
                                                'height': 800,
                                                'width': 800,
                                                'scale': 2
                                            }
                                        }
                                    )
                                )
                            ])
                        ])
                    ], label="Correlation Matrix", tab_id="tab-matrix"),
                    
                    # Tab 3: Advanced Regression Analysis
                    dbc.Tab([
                        dbc.Card([
                            dbc.CardHeader("Select Variables"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("X Variable:"),
                                        dcc.Dropdown(
                                            id='x-dropdown-advanced',
                                            placeholder="Select first variable",
                                            className="mb-3"
                                        )
                                    ], md=6),
                                    dbc.Col([
                                        html.Label("Y Variable:"),
                                        dcc.Dropdown(
                                            id='y-dropdown-advanced',
                                            placeholder="Select second variable",
                                            className="mb-3"
                                        )
                                    ], md=6),
                                ])
                            ])
                        ], className="mb-4"),
                        
                        dbc.Card([
                            dbc.CardHeader("Model Comparison"),
                            dbc.CardBody([
                                html.P([
                                    "This analysis fits various regression models to find the best relationship between variables. ",
                                    "The table shows all models sorted by R², and the plot displays the top 5 best-fitting models."
                                ], className="text-muted mb-3"),
                                html.Div(id="model-comparison-table", className="mb-4"),
                                dcc.Loading(
                                    id="loading-advanced",
                                    type="circle",
                                    children=dcc.Graph(
                                        id='model-comparison-plot',
                                        config={
                                            'displayModeBar': True,
                                            'displaylogo': False,
                                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                                        }
                                    )
                                )
                            ])
                        ])
                    ], label="Advanced Regression", tab_id="tab-advanced"),
                    
                ], id="correlation-tabs", active_tab="tab-pairwise")
            ], width=12)
        ])
    ], fluid=True, className="py-3")

# Add callback to populate advanced dropdowns
@dash.callback(
    [Output('x-dropdown-advanced', 'options'),
     Output('y-dropdown-advanced', 'options')],
    Input('memory-output', 'data'),
    prevent_initial_call=True
)
def update_advanced_dropdowns(data):
    if data is None:
        return no_update, no_update
    df = pd.DataFrame(data)
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    options = sorted(list(numeric_cols))
    return options, options
