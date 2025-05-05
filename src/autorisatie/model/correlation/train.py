import numpy as np
import pandas as pd
import sqlite3
from scipy.stats import lognorm
import pickle
import os
from pathlib import Path
from .model import CorrelationModel

# Lab test columns
LAB_COLUMNS = [
    'AFSE', 'ALTSE', 'CHOSE', 'GGTSE', 'HB', 'KALSE', 'KRESE', 'LEUC',
    'NATSE', 'TRISE', 'ALBSE', 'ASTSE', 'CRPSE', 'TRC'
]

def prepare_data(data_path):
    """Prepare data for training."""
    # Load data
    conn = sqlite3.connect(data_path)
    conn.text_factory = lambda x: x.decode(errors='ignore')
    query = conn.execute("SELECT * FROM wide_blood_draws")
    cols = [column[0] for column in query.description]
    df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    conn.close()

    # Select lab values
    data = df[LAB_COLUMNS]
    data = data.apply(pd.to_numeric, errors='coerce').fillna(-1)
    return data.values

def fit_distributions(data):
    """Fit log-normal distributions to each lab test."""
    n_features = len(LAB_COLUMNS)
    dist_params = []
    
    for i in range(n_features):
        valid_data = data[data[:, i] != -1, i]
        if len(valid_data) > 1:
            # Add small constant to avoid negative/zero values
            valid_data = valid_data + 1e-10
            shape, loc, scale = lognorm.fit(valid_data, floc=0)  # loc=0 for simplicity
            dist_params.append((shape, loc, scale))
        else:
            dist_params.append(None)
            print(f"Warning: No valid data for {LAB_COLUMNS[i]}")
    
    return dist_params

def calculate_correlation_matrix(data):
    """Calculate correlation matrix between lab tests."""
    n_features = len(LAB_COLUMNS)
    corr_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Select samples where both tests are not -1
            mask = (data[:, i] != -1) & (data[:, j] != -1)
            if np.sum(mask) > 1:
                corr = np.corrcoef(data[mask, i], data[mask, j])[0, 1]
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
            else:
                corr_matrix[i, j] = 0
                corr_matrix[j, i] = 0
    
    return corr_matrix

def train_model(data_path):
    """Train the correlation model."""
    # Prepare data
    data = prepare_data(data_path)
    
    # Fit distributions
    dist_params = fit_distributions(data)
    
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(data)
    
    # Create model
    model = CorrelationModel(dist_params, corr_matrix, LAB_COLUMNS)
    
    # Save model parameters
    os.makedirs('../out/model', exist_ok=True)
    with open('../out/model/correlation_dist_params.pkl', 'wb') as f:
        pickle.dump(dist_params, f)
    with open('../out/model/correlation_matrix.pkl', 'wb') as f:
        pickle.dump(corr_matrix, f)
    
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train the correlation model')
    parser.add_argument('--data', type=str, required=True, help='Path to the database file')
    args = parser.parse_args()
    
    train_model(args.data) 