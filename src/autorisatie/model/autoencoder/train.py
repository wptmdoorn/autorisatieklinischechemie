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
from tqdm import tqdm
from .model import Autoencoder

# Training parameters
TRAINING_PARAMS = {
    'batch_size': 512,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'patience': 10,  # Early stopping patience
    'min_delta': 0.0001,  # Minimum improvement for early stopping
    'encoding_dim': 10,
    'input_dim': 14,  # Number of lab tests
}

# Lab test columns
LAB_COLUMNS = [
    'AFSE', 'ALTSE', 'CHOSE', 'GGTSE', 'HB', 'KALSE', 'KRESE', 'LEUC',
    'NATSE', 'TRISE', 'ALBSE', 'ASTSE', 'CRPSE', 'TRC'
]

def prepare_data(data_path, test_mode=False):
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

    # Save scaler
    os.makedirs('out/model', exist_ok=True)
    with open('out/model/autoencoder_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Convert to PyTorch tensor
    data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
    dataset = TensorDataset(data_tensor, data_tensor)
    
    if test_mode:
        # Use a smaller subset for testing
        dataset = torch.utils.data.Subset(dataset, range(1000))
    
    dataloader = DataLoader(dataset, batch_size=TRAINING_PARAMS['batch_size'], shuffle=True)
    return dataloader, scaler

def train_model(data_path, test_mode=False):
    """Train the autoencoder model with early stopping."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data
    dataloader, scaler = prepare_data(data_path, test_mode)
    
    # Initialize model
    model = Autoencoder(
        input_dim=TRAINING_PARAMS['input_dim'],
        encoding_dim=TRAINING_PARAMS['encoding_dim']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_PARAMS['learning_rate'])
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    
    for epoch in range(TRAINING_PARAMS['num_epochs']):
        model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{TRAINING_PARAMS["num_epochs"]}'):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            mask = (inputs != -1).float()
            loss = criterion(outputs, targets)
            loss = (loss * mask).sum() / torch.clamp(mask.sum(), min=1e-10)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        # Early stopping check
        if avg_loss < best_loss - TRAINING_PARAMS['min_delta']:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'out/model/autoencoder_lab_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= TRAINING_PARAMS['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f'Epoch [{epoch+1}/{TRAINING_PARAMS["num_epochs"]}], Loss: {avg_loss:.4f}')
    
    # Calculate and save max MSE values
    model.load_state_dict(torch.load('out/model/autoencoder_lab_model.pth'))
    model.eval()
    
    mse_values = []
    mse_per_test_values = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, _ = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            mask = (inputs != -1).float()
            
            # Calculate MSE
            mse = ((outputs - inputs) ** 2 * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-10)
            mse_values.append(mse.cpu().numpy())
            
            # Calculate MSE per test
            mse_per_test = ((outputs - inputs) ** 2 * mask).cpu().numpy()
            mse_per_test_values.append(mse_per_test)
    
    mse_values = np.concatenate(mse_values)
    max_mse = np.max(mse_values[np.isfinite(mse_values)]) if np.any(np.isfinite(mse_values)) else 1.0
    
    mse_per_test_values = np.concatenate(mse_per_test_values, axis=0)
    max_mse_per_test = np.max(mse_per_test_values, axis=0, where=mse_per_test_values != 0, initial=1.0)
    max_mse_per_test = np.where(max_mse_per_test == 0, 1.0, max_mse_per_test)
    
    # Save max MSE values
    with open('out/model/autoencoder_max_mse.pkl', 'wb') as f:
        pickle.dump(max_mse, f)
    with open('out/model/autoencoder_max_mse_per_test.pkl', 'wb') as f:
        pickle.dump(max_mse_per_test, f)
    
    return model, scaler, max_mse, max_mse_per_test

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train the autoencoder model')
    parser.add_argument('--data', type=str, required=True, help='Path to the database file')
    parser.add_argument('--test', action='store_true', help='Run in test mode with smaller dataset')
    args = parser.parse_args()
    
    train_model(args.data, args.test) 