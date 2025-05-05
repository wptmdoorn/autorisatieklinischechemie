import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
from .foundation_model import FoundationModel
from .foundation_train import prepare_data, LabTestDataset

def objective(trial: Trial, data_path: str, test_mode: bool = False):
    """Objective function for Optuna optimization."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters to optimize
    sequence_length = trial.suggest_int('sequence_length', 5, 20)
    d_model = trial.suggest_int('d_model', 128, 512, step=64)
    nhead = trial.suggest_int('nhead', 4, 16, step=4)
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 2, 8)
    num_decoder_layers = trial.suggest_int('num_decoder_layers', 2, 8)
    dim_feedforward = trial.suggest_int('dim_feedforward', 512, 2048, step=256)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    
    # Prepare data
    train_dataset, val_dataset, _ = prepare_data(
        data_path, sequence_length, test_mode
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    model = FoundationModel(
        input_dim=14,  # Number of lab tests
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    max_epochs = 50  # Limit epochs for hyperparameter tuning
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            sequences, test_indices = batch
            sequences = sequences.to(device)
            test_indices = test_indices.to(device)
            
            # Forward pass
            outputs = model(sequences, test_indices)
            
            # Calculate loss
            mask = (sequences != -1).float()
            loss = criterion(outputs, sequences)
            loss = (loss * mask).sum() / torch.clamp(mask.sum(), min=1e-10)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                sequences, test_indices = batch
                sequences = sequences.to(device)
                test_indices = test_indices.to(device)
                
                outputs = model(sequences, test_indices)
                
                mask = (sequences != -1).float()
                loss = criterion(outputs, sequences)
                loss = (loss * mask).sum() / torch.clamp(mask.sum(), min=1e-10)
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:  # Shorter patience for hyperparameter tuning
                break
        
        # Report intermediate value
        trial.report(avg_val_loss, epoch)
        
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
    os.makedirs('../out/model', exist_ok=True)
    with open('../out/model/foundation_best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    
    print("Best hyperparameters:", best_params)
    print("Best validation loss:", study.best_value)
    
    return best_params

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Optimize foundation model hyperparameters')
    parser.add_argument('--data', type=str, required=True, help='Path to the database file')
    parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--test', action='store_true', help='Run in test mode with smaller dataset')
    args = parser.parse_args()
    
    optimize_hyperparameters(args.data, args.trials, args.test) 