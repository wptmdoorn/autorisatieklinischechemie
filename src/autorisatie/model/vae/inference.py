import torch
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from .model import VAE
from sklearn.preprocessing import StandardScaler

class VAEPredictor:
    def __init__(self, model_dir='../out/model'):
        """Initialize the predictor with saved model and scaler."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = Path(model_dir)
        
        # Load model
        self.model = VAE(input_dim=14, encoding_dim=10)
        model_path = self.model_dir / 'vae_lab_model.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device),
                                   strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Load scaler
        scaler_path = self.model_dir / 'vae_scaler.pkl'
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load max MSE values
        max_mse_path = self.model_dir / 'vae_max_mse.pkl'
        max_mse_per_test_path = self.model_dir / 'vae_max_mse_per_test.pkl'
        
        if not max_mse_path.exists() or not max_mse_per_test_path.exists():
            raise FileNotFoundError("Max MSE files not found")
            
        with open(max_mse_path, 'rb') as f:
            self.max_mse = pickle.load(f)
        with open(max_mse_per_test_path, 'rb') as f:
            self.max_mse_per_test = pickle.load(f)
        
        # Lab test columns
        self.lab_columns = [
            'AFSE', 'ALTSE', 'CHOSE', 'GGTSE', 'HB', 'KALSE', 'KRESE', 'LEUC',
            'NATSE', 'TRISE', 'ALBSE', 'ASTSE', 'CRPSE', 'TRC'
        ]

    def _prepare_input(self, data):
        """Prepare input data for prediction."""
        if isinstance(data, pd.DataFrame):
            # Ensure all required columns are present
            missing_cols = set(self.lab_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            data = data[self.lab_columns]
        
        # Convert to numpy array if needed
        data = np.array(data, dtype=np.float32)
        
        # Handle missing values
        data = np.where(np.isnan(data), -1, data)
        
        # Normalize data
        data_normalized = data.copy()
        for i in range(len(self.lab_columns)):
            mask_col = data_normalized[:, i] != -1
            if np.sum(mask_col) > 0:
                try:
                    data_normalized[mask_col, i] = self.scaler.transform(
                        data[mask_col, i].reshape(-1, 1)
                    ).flatten()
                except:
                    data_normalized[mask_col, i] = 0
        
        return torch.tensor(data_normalized, dtype=torch.float32).to(self.device)

    def predict(self, data):
        """
        Make predictions on input data.
        
        Args:
            data: Input data as pandas DataFrame or numpy array
            
        Returns:
            dict: Dictionary containing:
                - 'total_score': Overall feasibility score for each sample
                - 'test_scores': Individual scores for each lab test
                - 'mse': Mean squared error for each sample
                - 'mse_per_test': MSE for each lab test
        """
        # Prepare input
        data_tensor = self._prepare_input(data)
        
        # Make predictions
        with torch.no_grad():
            reconstructions, mu, log_var = self.model(data_tensor)
            
            # Calculate reconstruction loss
            mask = (data_tensor != -1).float()
            mse = ((reconstructions - data_tensor) ** 2 * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-10)
            mse = mse.cpu().numpy()
            
            # Calculate MSE per test
            mse_per_test = ((reconstructions - data_tensor) ** 2 * mask).cpu().numpy()
            
            # Calculate KL divergence
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            kl_div = kl_div.cpu().numpy()
            
            # Combine reconstruction loss and KL divergence
            total_loss = mse + kl_div
            
            # Calculate scores
            total_score = np.where(total_loss == np.inf, 0.0, 1 - (total_loss / self.max_mse))
            total_score = np.clip(total_score, 0, 1)
            
            test_scores = np.zeros_like(mse_per_test)
            for i in range(mse_per_test.shape[1]):
                test_scores[:, i] = np.where(
                    mse_per_test[:, i] == 0,
                    np.nan,
                    1 - (mse_per_test[:, i] / self.max_mse_per_test[i])
                )
            test_scores = np.clip(test_scores, 0, 1)
        
        return {
            'total_score': total_score,
            'test_scores': test_scores,
            'mse': mse,
            'mse_per_test': mse_per_test,
            'kl_div': kl_div
        } 