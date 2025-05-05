import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os
from pathlib import Path
import optuna
from optuna.trial import Trial
import sqlite3
from tqdm import tqdm
from .model import Autoencoder

# Lab test columns
LAB_COLUMNS = [
    'AFSE', 'ALTSE', 'CHOSE', 'GGTSE', 'HB', 'KALSE', 'KRESE', 'LEUC',
    'NATSE', 'TRISE', 'ALBSE', 'ASTSE', 'CRPSE', 'TRC'
]

def prepare_data(data_path, batch_size, test_mode=False):
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

    # Normalize data
    data_normalized = data.values.copy()
    scaler = StandardScaler()

    for i in range(len(LAB_COLUMNS)):
        mask_col = data_normalized[:, i] != -1
        if np.sum(mask_col) > 1:
            data_normalized[mask_col, i] = scaler.fit_transform(
                data_normalized[mask_col, i].reshape(-1, 1)
            ).flatten()

    # Convert to PyTorch tensor
    data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
    dataset = TensorDataset(data_tensor, data_tensor)
    
    if test_mode:
        # Use a smaller subset for testing
        dataset = torch.utils.data.Subset(dataset, range(25000))
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler

def create_model(trial: Trial, input_dim: int) -> Autoencoder:
    """Create a model with hyperparameters suggested by Optuna."""
    # Define the hyperparameter search space
    encoding_dim = trial.suggest_int('encoding_dim', 4, 16)
    hidden_dims = []
    n_layers = trial.suggest_int('n_layers', 2, 4)
    
    # Generate hidden layer dimensions
    prev_dim = input_dim
    for i in range(n_layers):
        hidden_dim = trial.suggest_int(f'hidden_dim_{i}', 32, 256)
        hidden_dims.append(hidden_dim)
        prev_dim = hidden_dim
    
    # Create encoder layers
    encoder_layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        encoder_layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.ReLU()
        ])
        prev_dim = hidden_dim
    encoder_layers.extend([
        nn.Linear(prev_dim, encoding_dim),
        nn.ReLU()
    ])
    
    # Create decoder layers
    decoder_layers = []
    prev_dim = encoding_dim
    for hidden_dim in reversed(hidden_dims):
        decoder_layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.ReLU()
        ])
        prev_dim = hidden_dim
    decoder_layers.append(nn.Linear(prev_dim, input_dim))
    
    # Create model
    model = Autoencoder(input_dim, encoding_dim)
    model.encoder = nn.Sequential(*encoder_layers)
    model.decoder = nn.Sequential(*decoder_layers)
    
    return model

def objective(trial: Trial, data_path: str, test_mode: bool = False):
    """Objective function for Optuna optimization."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters to optimize
    batch_size = trial.suggest_int('batch_size', 128, 1024, step=128)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    num_epochs = trial.suggest_int('num_epochs', 50, 200)
    patience = trial.suggest_int('patience', 5, 20)
    min_delta = trial.suggest_float('min_delta', 1e-5, 1e-3, log=True)
    
    # Prepare data
    train_loader, val_loader, _ = prepare_data(data_path, batch_size, test_mode)
    
    # Create model
    model = create_model(trial, input_dim=len(LAB_COLUMNS))
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            mask = (inputs != -1).float()
            loss = criterion(outputs, targets)
            loss = (loss * mask).sum() / torch.clamp(mask.sum(), min=1e-10)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                mask = (inputs != -1).float()
                loss = criterion(outputs, targets)
                loss = (loss * mask).sum() / torch.clamp(mask.sum(), min=1e-10)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
        # Report intermediate value
        trial.report(val_loss, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_loss

def optimize_hyperparameters(data_path: str, n_trials: int = 100, test_mode: bool = False):
    """Run hyperparameter optimization."""
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, data_path, test_mode), n_trials=n_trials)
    
    # Save best hyperparameters
    best_params = study.best_params
    os.makedirs('out/model', exist_ok=True)
    with open('out/model/autoencoder_best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    
    print("Best hyperparameters:", best_params)
    print("Best validation loss:", study.best_value)
    
    return best_params

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Optimize autoencoder hyperparameters')
    parser.add_argument('--data', type=str, required=True, help='Path to the database file')
    parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--test', action='store_true', help='Run in test mode with smaller dataset')
    args = parser.parse_args()
    
    optimize_hyperparameters(args.data, args.trials, args.test) 