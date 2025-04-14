import numpy as np
import pandas as pd
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pickle
from scipy.stats import lognorm

# Stap 1: Laad en voorbewerk de data
conn = sqlite3.connect("data/clean/2020.db")
conn.text_factory = lambda x: x.decode(errors='ignore')
query = conn.execute("SELECT * FROM wide_blood_draws")
cols = [column[0] for column in query.description]
df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
conn.close()

# Selecteer alleen labwaarden
lab_columns = ['AFSE', 'ALTSE', 'CHOSE', 'GGTSE', 'HB', 'KALSE', 'KRESE', 'LEUC', 'NATSE', 'TRISE', 
               'ALBSE', 'ASTSE', 'CRPSE', 'TRC']
data = df[lab_columns]

# Vervang tekst en NaN door -1
data = data.apply(pd.to_numeric, errors='coerce').fillna(-1)

# Stap 2: Normaliseer de data
n_features = len(lab_columns)
data_normalized = data.values.copy()
scaler = StandardScaler()

for i in range(n_features):
    mask_col = data_normalized[:, i] != -1
    if np.sum(mask_col) > 1:
        data_normalized[mask_col, i] = scaler.fit_transform(
            data_normalized[mask_col, i].reshape(-1, 1)
        ).flatten()

with open('out/model/autoencoder_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler opgeslagen als 'autoencoder_scaler.pkl'")

# Check op volledig gemaskeerde samples
valid_samples = np.any(data_normalized != -1, axis=1)
if not np.all(valid_samples):
    print(f"Waarschuwing: {np.sum(~valid_samples)} samples hebben alleen -1 waarden")
    data_normalized = data_normalized[valid_samples]

# Zet data om naar PyTorch tensor
data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
dataset = TensorDataset(data_tensor, data_tensor)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

# Stap 3: Definieer de autoencoder
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

# Instantieer het model
input_dim = n_features
encoding_dim = 10  # Vergroot van 5 naar 10
model = Autoencoder(input_dim, encoding_dim)

# Stap 4: Definieer loss en optimizer
criterion = nn.MSELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Stap 5: Train het model
num_epochs = 20  # Vergroot van 10 naar 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Verzamel MSE-waarden
mse_values = []
mse_per_test_values = []
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
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
            
        # Bereken MSE
        mse = ((outputs - inputs) ** 2 * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-10)
        mse_values.append(mse.detach().numpy())
            
        # Bereken MSE per test
        mse_per_test = ((outputs - inputs) ** 2 * mask).detach().numpy()
        mse_per_test_values.append(mse_per_test)
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
 # Bereken max_mse
mse_values = np.concatenate(mse_values)
max_mse = np.max(mse_values[np.isfinite(mse_values)]) if np.any(np.isfinite(mse_values)) else 1.0
with open('out/model/autoencoder_max_mse.pkl', 'wb') as f:
    pickle.dump(max_mse, f)
    
# Bereken max_mse_per_test
mse_per_test_values = np.concatenate(mse_per_test_values, axis=0)
max_mse_per_test = np.max(mse_per_test_values, axis=0, where=mse_per_test_values != 0, initial=1.0)
max_mse_per_test = np.where(max_mse_per_test == 0, 1.0, max_mse_per_test)
with open('out/model/autoencoder_max_mse_per_test.pkl', 'wb') as f:
    pickle.dump(max_mse_per_test, f)

# Stap 6: Sla het model op
torch.save(model.state_dict(), 'out/model/autoencoder_lab_model.pth')
print("Model opgeslagen als 'autoencoder_lab_model.pth'")

# Stap 7: Functie voor totale feasibility score
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
    mse = mse.detach().numpy()
    
    mse = np.where(np.isnan(mse), np.inf, mse)
    if max_mse is None:
        max_mse = np.max(mse[mse != np.inf]) if np.any(mse != np.inf) else 1.0
    feasibility_score = np.where(mse == np.inf, 0.0, 1 - (mse / max_mse))
    feasibility_score = np.clip(feasibility_score, 0, 1)
    
    return feasibility_score, mse

# Stap 8: Functie voor pseudo-waarschijnlijkheid per labtest
def get_pseudo_probability_score_per_test(data_new, model, scaler, device, min_sigma=0.1):
    """
    Bereken een pseudo-waarschijnlijkheidsscore per labtest, met strengere log-normale foutverdeling.
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
    print(f"Genormaliseerde waarde voor NATSE (sample 0): {data_new_normalized[0, lab_columns.index('NATSE')]:.4f}")
    
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
    natse_idx = lab_columns.index('NATSE')
    print(f"MSE voor NATSE (sample 0): {mse_per_test[0, natse_idx]:.4f}")
    
    # Schat parameters voor log-normale verdeling
    sigma_per_test = np.std(np.log1p(mse_per_test + 1e-10), axis=0, where=mse_per_test != 0)
    sigma_per_test = np.where(sigma_per_test == 0, min_sigma, np.maximum(sigma_per_test, min_sigma))
    
    # Bereken pseudo-waarschijnlijkheidsscore
    probability_scores = np.zeros_like(mse_per_test)
    for i in range(mse_per_test.shape[1]):
        errors = mse_per_test[:, i] + 1e-10
        # Strengere score: schaal errors met een factor om outliers zwaarder te straffen
        scaled_errors = errors / (sigma_per_test[i] ** 2)  # Kwadratische straf
        prob = lognorm.cdf(scaled_errors, s=sigma_per_test[i], scale=1.0)
        probability_scores[:, i] = np.where(
            mse_per_test[:, i] == 0,
            np.nan,
            1 - prob
        )
    
    probability_scores = np.clip(probability_scores, 0, 1)
    return probability_scores, mse_per_test

# Stap 9: Test inferentie
# Originele data
print(data.values)
feasibility_scores, mse_values = get_feasibility_score(data.values, model, scaler, device)
print("Totale feasibility scores (eerste 5 samples):", feasibility_scores[:5])

# Pseudo-waarschijnlijkheden voor originele data
prob_scores, mse_per_test = get_pseudo_probability_score_per_test(data.values, model, scaler, device)
print("Pseudo-waarschijnlijkheden voor sample 0:")
for i, test in enumerate(lab_columns):
    print(f"{test}: {prob_scores[0, i]:.3f}" if not np.isnan(prob_scores[0, i]) else f"{test}: ontbrekend")

# Test met NATSE = 1400
abnormal_data = data.values.copy()
abnormal_data[0, lab_columns.index('NATSE')] = 1400
prob_scores_abnormal, mse_per_test_abnormal = get_pseudo_probability_score_per_test(
    abnormal_data, model, scaler, device
)
print("Pseudo-waarschijnlijkheden voor sample 0 (NATSE = 1400):")
for i, test in enumerate(lab_columns):
    print(f"{test}: {prob_scores_abnormal[0, i]:.3f}" if not np.isnan(prob_scores_abnormal[0, i]) else f"{test}: ontbrekend")