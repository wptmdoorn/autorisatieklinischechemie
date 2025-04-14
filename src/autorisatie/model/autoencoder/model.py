import numpy as np
import pandas as pd
import sqlite3
import torch, pickle
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import lognorm

lab_columns = ['AFSE', 'ALTSE', 'CHOSE', 'GGTSE', 'HB', 'KALSE', 'KRESE', 'LEUC', 'NATSE', 'TRISE', 
               'ALBSE', 'ASTSE', 'CRPSE', 'TRC']

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def get_pseudo_probability_score_per_test(data_new, model, scaler, device, max_mse_per_test=None):
    """
    Bereken een pseudo-waarschijnlijkheidsscore per labtest, consistent met get_feasibility_score.
    """
    data_new_normalized = data_new.copy()
    for i in range(data_new.shape[1]):
        mask_col = data_new[:, i] != -1
        if np.sum(mask_col) > 0:
            try:
                data_new_normalized[mask_col, i] = scaler.transform(
                    data_new[mask_col, i].reshape(-1, 1)
                ).flatten()
            except Exception as e:
                print(f"Fout bij normaliseren kolom {i}: {e}")
                data_new_normalized[mask_col, i] = 0
    
    # Debugging: print genormaliseerde waarde
    natse_idx = lab_columns.index('NATSE')
    print(f"Genormaliseerde waarde voor NATSE (sample 0): {data_new_normalized[0, natse_idx]:.4f}")
    
    valid_samples = np.any(data_new_normalized != -1, axis=1)
    if not np.all(valid_samples):
        print(f"Waarschuwing: {np.sum(~valid_samples)} samples hebben alleen -1 waarden")
    
    data_tensor = torch.tensor(data_new_normalized, dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        reconstructions = model(data_tensor)
    
    mask = (data_tensor != -1).float()
    mse_per_test = (reconstructions - data_tensor) ** 2 * mask
    mse_per_test = mse_per_test.cpu().numpy()
    
    # Debugging: print MSE voor NATSE
    print(f"MSE voor NATSE (sample 0): {mse_per_test[0, natse_idx]:.4f}")
    
    # Laad max_mse_per_test als niet meegegeven
    if max_mse_per_test is None:
        print("Waarschuwing: max_mse_per_test niet opgegeven, gebruik opgeslagen waarde")
        with open('../out/model/autoencoder_max_mse_per_test.pkl', 'rb') as f:
            max_mse_per_test = pickle.load(f)
    
    # Bereken pseudo-waarschijnlijkheidsscore
    probability_scores = np.zeros_like(mse_per_test)
    for i in range(mse_per_test.shape[1]):
        probability_scores[:, i] = np.where(
            mse_per_test[:, i] == 0,
            np.nan,
            1 - (mse_per_test[:, i] / max_mse_per_test[i])
        )
    
    probability_scores = np.clip(probability_scores, 0, 1)
    print(f"Gebruikte max_mse_per_test: {max_mse_per_test}")
    return probability_scores, mse_per_test

def get_feasibility_score(data_new, model, scaler, device, max_mse=None):
    data_new_normalized = data_new.copy()
    for i in range(data_new.shape[1]):
        mask_col = data_new[:, i] != -1
        if np.sum(mask_col) > 0:
            try:
                data_new_normalized[mask_col, i] = scaler.transform(
                    data_new[mask_col, i].reshape(-1, 1)
                ).flatten()
            except:
                data_new_normalized[mask_col, i] = 0
    
    valid_samples = np.any(data_new_normalized != -1, axis=1)
    if not np.all(valid_samples):
        print(f"Waarschuwing: {np.sum(~valid_samples)} samples hebben alleen -1 waarden")
    
    data_tensor = torch.tensor(data_new_normalized, dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        reconstructions = model(data_tensor)
    
    mask = (data_tensor != -1).float()
    mse = ((reconstructions - data_tensor) ** 2 * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-10)
    mse = mse.cpu().numpy()
    
    mse = np.where(np.isnan(mse), np.inf, mse)
    if max_mse is None:
        print("Waarschuwing: max_mse niet opgegeven, gebruik opgeslagen waarde")
        with open('../out/model/autoencoder_max_mse.pkl', 'rb') as f:
            max_mse = pickle.load(f)

    feasibility_score = np.where(mse == np.inf, 0.0, 1 - (mse / max_mse))
    feasibility_score = np.clip(feasibility_score, 0, 1)
    
    print(f"Gebruikte max_mse: {max_mse:.4f}")
    print(f"MSE voor eerste 5 samples: {mse[:5]}")
    return feasibility_score, mse