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
from .model import get_pseudo_probability_score_per_test, Autoencoder, get_feasibility_score
import pickle, sqlite3

# Initialize model and scaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder(input_dim=14, encoding_dim=10)
model_path = os.path.join('..', 'out', 'model', 'autoencoder_lab_model.pth')

try:
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Successfully loaded autoencoder model")
    else:
        print("Warning: Model file not found at", model_path)
except Exception as e:
    print(f"Error loading model: {str(e)}")

with open('../out/model/autoencoder_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

lab_columns = ['AFSE', 'ALTSE', 'CHOSE', 'GGTSE', 'HB', 'KALSE', 'KRESE', 'LEUC', 'NATSE', 'TRISE', 
               'ALBSE', 'ASTSE', 'CRPSE', 'TRC']

# Stap 1: Laad en voorbewerk de data
conn = sqlite3.connect("../data/clean/2020.db")
conn.text_factory = lambda x: x.decode(errors='ignore')
query = conn.execute("SELECT * FROM wide_blood_draws LIMIT 1000")
cols = [column[0] for column in query.description]
df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
conn.close()

# Selecteer alleen labwaarden
lab_columns = ['AFSE', 'ALTSE', 'CHOSE', 'GGTSE', 'HB', 'KALSE', 'KRESE', 'LEUC', 'NATSE', 'TRISE', 
               'ALBSE', 'ASTSE', 'CRPSE', 'TRC']
data = df[lab_columns].head(8)
data = data.apply(pd.to_numeric, errors='coerce').fillna(-1)

# Create data array for model
print(data.to_numpy())

feasibility_scores, mse_values = get_feasibility_score(data.to_numpy(), model, scaler, device)
print("Totale feasibility scores (eerste 5 samples):", feasibility_scores)