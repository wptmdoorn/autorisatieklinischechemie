import numpy as np
import pandas as pd
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pickle

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

# Sla de scaler op
with open('out/model/vae_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler opgeslagen als 'scaler.pkl'")

# Check op volledig gemaskeerde samples
valid_samples = np.any(data_normalized != -1, axis=1)
if not np.all(valid_samples):
    print(f"Waarschuwing: {np.sum(~valid_samples)} samples hebben alleen -1 waarden")
    data_normalized = data_normalized[valid_samples]

# Zet data om naar PyTorch tensor
data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
dataset = TensorDataset(data_tensor, data_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Stap 3: Definieer de VAE
class VAE(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim * 2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        # Log-variances voor reconstructie
        self.logvar_decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim)
        )
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        recon_logvar = self.logvar_decoder(z)
        return recon, mu, logvar, recon_logvar

# Stap 4: Definieer de VAE loss-functie
def vae_loss(recon, x, mu, logvar, recon_logvar, mask):
    # Zorg dat log(2π) een tensor is
    log_2pi = torch.tensor(np.log(2 * np.pi), device=x.device, dtype=x.dtype)
    
    # Reconstructieloss (Gaussische log-waarschijnlijkheid)
    recon_loss = 0.5 * (log_2pi + recon_logvar + 
                        ((x - recon) ** 2 / torch.exp(recon_logvar))) * mask
    
    # Controleer op nan/inf in recon_loss
    if torch.isnan(recon_loss).any() or torch.isinf(recon_loss).any():
        print("Waarschuwing: nan/inf gedetecteerd in recon_loss")
        recon_loss = torch.where(
            torch.isnan(recon_loss) | torch.isinf(recon_loss),
            torch.zeros_like(recon_loss),
            recon_loss
        )
    
    recon_loss = recon_loss.sum() / torch.clamp(mask.sum(), min=1e-10)
    
    # KL-divergentie
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    kl_loss = kl_loss.mean()
    
    return recon_loss + kl_loss

# Stap 5: Instantieer en train het model
input_dim = n_features
encoding_dim = 5
model = VAE(input_dim, encoding_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        inputs, _ = batch
        inputs = inputs.to(device)
        
        recon, mu, logvar, recon_logvar = model(inputs)
        mask = (inputs != -1).float()
        loss = vae_loss(recon, inputs, mu, logvar, recon_logvar, mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

# Stap 6: Sla het model op
torch.save(model.state_dict(), 'out/model/vae_lab_model.pth')
print("Model opgeslagen als 'vae_lab_model.pth'")

# Stap 7: Functie voor waarschijnlijkheidsscore per labtest
def get_probability_score_per_test(data_new, model, scaler, device):
    """
    Bereken de waarschijnlijkheidsscore per labtest voor nieuwe data.
    """
    # Normaliseer nieuwe data
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
    
    # Debugging: print genormaliseerde waarde voor AFSE
    print(f"Genormaliseerde AFSE (sample 0): {data_new_normalized[0, 0]:.4f}")
    
    # Check op volledig gemaskeerde samples
    valid_samples = np.any(data_new_normalized != -1, axis=1)
    if not np.all(valid_samples):
        print(f"Waarschuwing: {np.sum(~valid_samples)} samples hebben alleen -1 waarden")
    
    # Zet om naar tensor
    data_tensor = torch.tensor(data_new_normalized, dtype=torch.float32).to(device)
    
    # Voorspel reconstructie
    model.eval()
    with torch.no_grad():
        recon, mu, logvar, recon_logvar = model(data_tensor)
    
    # Bereken log-waarschijnlijkheid per test
    mask = (data_tensor != -1).float()
    log_prob = -0.5 * (torch.log(2 * np.pi) + recon_logvar + 
                       ((data_tensor - recon) ** 2 / torch.exp(recon_logvar))) * mask
    
    # Converteer naar waarschijnlijkheidsscore
    probability_scores = torch.exp(log_prob / torch.clamp(mask.sum(dim=1, keepdim=True), min=1e-10))
    probability_scores = probability_scores.cpu().numpy()
    probability_scores = np.where(mask.cpu().numpy() == 0, np.nan, probability_scores)
    probability_scores = np.clip(probability_scores, 0, 1)
    
    return probability_scores

# Stap 8: Inferentie-test
# Laad scaler en model
with open('out/model/vae_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
model = VAE(input_dim=14, encoding_dim=5)
model.load_state_dict(torch.load('out/model/vae_lab_model.pth'))
model.to(device)
model.eval()

# Test met originele data
test_data = data.values.copy()
prob_scores = get_probability_score_per_test(test_data, model, scaler, device)
print("Waarschijnlijkheidsscore voor AFSE (sample 0, origineel):", 
      prob_scores[0, 0] if not np.isnan(prob_scores[0, 0]) else "ontbrekend")

# Test met AFSE = 10.000
test_data_abnormal = test_data.copy()
test_data_abnormal[0, 0] = 10000
prob_scores_abnormal = get_probability_score_per_test(test_data_abnormal, model, scaler, device)
print("Waarschijnlijkheidsscore voor AFSE (sample 0, AFSE=10.000):", 
      prob_scores_abnormal[0, 0] if not np.isnan(prob_scores_abnormal[0, 0]) else "ontbrekend")