import torch
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from .model import Autoencoder
from sklearn.preprocessing import StandardScaler

def load_model_and_scaler():
    """Load the trained model and scaler."""
    # Load best hyperparameters
    try:
        with open('../out/model/autoencoder_best_params.pkl', 'rb') as f:
            params = pickle.load(f)
    except FileNotFoundError:
        print("No optimized hyperparameters found. Using default values.")
        params = {
            'encoding_dim': 10,
            'n_layers': 3,
            'hidden_dim_0': 128,
            'hidden_dim_1': 64,
            'hidden_dim_2': 32
        }
    
    # Load scaler
    with open('../out/model/autoencoder_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load max MSE values
    with open('../out/model/autoencoder_max_mse.pkl', 'rb') as f:
        max_mse = pickle.load(f)
    with open('../out/model/autoencoder_max_mse_per_test.pkl', 'rb') as f:
        max_mse_per_test = pickle.load(f)
    
    # Create model with optimized architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder(input_dim=14, encoding_dim=params['encoding_dim'])
    
    # Set up encoder layers
    encoder_layers = []
    prev_dim = 14
    for i in range(params['n_layers']):
        hidden_dim = params[f'hidden_dim_{i}']
        encoder_layers.extend([
            torch.nn.Linear(prev_dim, hidden_dim),
            torch.nn.ReLU()
        ])
        prev_dim = hidden_dim
    encoder_layers.extend([
        torch.nn.Linear(prev_dim, params['encoding_dim']),
        torch.nn.ReLU()
    ])
    model.encoder = torch.nn.Sequential(*encoder_layers)
    
    # Set up decoder layers
    decoder_layers = []
    prev_dim = params['encoding_dim']
    for i in range(params['n_layers']-1, -1, -1):
        hidden_dim = params[f'hidden_dim_{i}']
        decoder_layers.extend([
            torch.nn.Linear(prev_dim, hidden_dim),
            torch.nn.ReLU()
        ])
        prev_dim = hidden_dim
    decoder_layers.append(torch.nn.Linear(prev_dim, 14))
    model.decoder = torch.nn.Sequential(*decoder_layers)
    
    # Load trained weights
    model.load_state_dict(torch.load('../out/model/autoencoder_lab_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, scaler, max_mse, max_mse_per_test

def preprocess_data(data, scaler):
    """Preprocess the input data."""
    # Convert to numpy array if it's a pandas DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # Normalize data
    data_normalized = data.copy()
    for i in range(data.shape[1]):
        mask_col = data_normalized[:, i] != -1
        if np.sum(mask_col) > 1:
            data_normalized[mask_col, i] = scaler.transform(
                data_normalized[mask_col, i].reshape(-1, 1)
            ).flatten()
    
    return data_normalized

def detect_anomalies(data, model, scaler, max_mse, max_mse_per_test, threshold=0.95):
    """Detect anomalies in the input data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preprocess data
    data_normalized = preprocess_data(data, scaler)
    
    # Convert to PyTorch tensor
    data_tensor = torch.tensor(data_normalized, dtype=torch.float32).to(device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(data_tensor)
    
    # Calculate MSE
    mask = (data_tensor != -1).float()
    mse = ((outputs - data_tensor) ** 2 * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-10)
    mse_per_test = ((outputs - data_tensor) ** 2 * mask).cpu().numpy()
    
    # Normalize MSE values
    mse_normalized = mse.cpu().numpy() / max_mse
    mse_per_test_normalized = mse_per_test / max_mse_per_test
    
    # Calculate anomaly scores
    anomaly_scores = np.max(mse_per_test_normalized, axis=1)
    
    # Detect anomalies
    anomalies = anomaly_scores > threshold
    
    return anomalies, anomaly_scores, mse_normalized, mse_per_test_normalized

def get_anomaly_details(data, anomalies, anomaly_scores, mse_per_test_normalized):
    """Get detailed information about detected anomalies."""
    # Convert to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # Get anomalous samples
    anomalous_data = data[anomalies].copy()
    anomalous_scores = anomaly_scores[anomalies]
    anomalous_mse = mse_per_test_normalized[anomalies]
    
    # Create detailed report
    details = []
    for idx, (_, row) in enumerate(anomalous_data.iterrows()):
        # Find which tests contributed most to the anomaly
        test_scores = anomalous_mse[idx]
        top_tests = np.argsort(test_scores)[-3:][::-1]  # Get top 3 contributing tests
        
        detail = {
            'sample_index': row.name,
            'anomaly_score': anomalous_scores[idx],
            'top_contributing_tests': [
                {
                    'test_name': data.columns[test_idx],
                    'value': row.iloc[test_idx],
                    'contribution': test_scores[test_idx]
                }
                for test_idx in top_tests
            ]
        }
        details.append(detail)
    
    return details

def analyze_data(data, threshold=0.95):
    """Analyze data for anomalies and return detailed results."""
    # Load model and scaler
    model, scaler, max_mse, max_mse_per_test = load_model_and_scaler()
    
    # Detect anomalies
    anomalies, anomaly_scores, mse_normalized, mse_per_test_normalized = detect_anomalies(
        data, model, scaler, max_mse, max_mse_per_test, threshold
    )
    
    # Get detailed information about anomalies
    anomaly_details = get_anomaly_details(
        data, anomalies, anomaly_scores, mse_per_test_normalized
    )
    
    return {
        'anomalies_detected': anomalies.sum(),
        'total_samples': len(data),
        'anomaly_rate': anomalies.mean(),
        'anomaly_details': anomaly_details
    }

class AutoencoderPredictor:
    def __init__(self, model_dir='../out/model'):
        """Initialize the predictor with saved model and scaler."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = Path(model_dir)
        
        # Load model
        self.model = Autoencoder(input_dim=14, encoding_dim=10)
        model_path = self.model_dir / 'autoencoder_lab_model.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load scaler
        scaler_path = self.model_dir / 'autoencoder_scaler.pkl'
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load max MSE values
        max_mse_path = self.model_dir / 'autoencoder_max_mse.pkl'
        max_mse_per_test_path = self.model_dir / 'autoencoder_max_mse_per_test.pkl'
        
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
            reconstructions = self.model(data_tensor)
        
        # Calculate MSE
        mask = (data_tensor != -1).float()
        mse = ((reconstructions - data_tensor) ** 2 * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-10)
        mse = mse.cpu().numpy()
        
        # Calculate MSE per test
        mse_per_test = ((reconstructions - data_tensor) ** 2 * mask).cpu().numpy()
        
        # Calculate scores
        total_score = np.where(mse == np.inf, 0.0, 1 - (mse / self.max_mse))
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
            'mse_per_test': mse_per_test
        } 