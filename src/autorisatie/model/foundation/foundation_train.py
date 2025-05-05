import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os
from pathlib import Path
from tqdm import tqdm
import sqlite3
from .foundation_model import FoundationModel

# Lab test columns
LAB_COLUMNS = [
    'AFSE', 'ALTSE', 'CHOSE', 'GGTSE', 'HB', 'KALSE', 'KRESE', 'LEUC',
    'NATSE', 'TRISE', 'ALBSE', 'ASTSE', 'CRPSE', 'TRC'
]

class LabTestDataset(Dataset):
    def __init__(self, data, test_indices, sequence_length=10):
        self.data = data
        self.test_indices = test_indices
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # Get sequence of values
        sequence = self.data[idx:idx + self.sequence_length]
        test_idx = self.test_indices[idx:idx + self.sequence_length]
        
        # Convert to tensors
        sequence = torch.tensor(sequence, dtype=torch.float32)
        test_idx = torch.tensor(test_idx, dtype=torch.long)
        
        return sequence, test_idx

def prepare_data(data_path, sequence_length=10, test_mode=False):
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
    os.makedirs('../out/model', exist_ok=True)
    with open('../out/model/foundation_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Create test indices (0 to num_tests-1)
    test_indices = np.arange(len(LAB_COLUMNS))
    test_indices = np.tile(test_indices, (len(data_normalized), 1))

    if test_mode:
        # Use a smaller subset for testing
        data_normalized = data_normalized[:1000]
        test_indices = test_indices[:1000]

    # Create dataset
    dataset = LabTestDataset(data_normalized, test_indices, sequence_length)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset, scaler

def train_model(
    data_path,
    sequence_length=10,
    d_model=256,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=1024,
    dropout=0.1,
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=100,
    patience=10,
    min_delta=1e-4,
    test_mode=False
):
    """Train the foundation model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data
    train_dataset, val_dataset, scaler = prepare_data(
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
    
    # Initialize model
    model = FoundationModel(
        input_dim=len(LAB_COLUMNS),
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
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
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
        train_losses.append(avg_train_loss)
        
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
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), '../out/model/foundation_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open('../out/model/foundation_training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    return model, scaler

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train the foundation model')
    parser.add_argument('--data', type=str, required=True, help='Path to the database file')
    parser.add_argument('--sequence_length', type=int, default=10, help='Length of input sequences')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Minimum improvement for early stopping')
    parser.add_argument('--test', action='store_true', help='Run in test mode with smaller dataset')
    args = parser.parse_args()
    
    train_model(
        args.data,
        sequence_length=args.sequence_length,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        test_mode=args.test
    ) 