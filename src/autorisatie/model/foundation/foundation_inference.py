import torch
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from .foundation_model import FoundationModel

def load_model_and_scaler():
    """Load the trained model and scaler."""
    # Load best hyperparameters
    try:
        with open('../out/model/foundation_best_params.pkl', 'rb') as f:
            params = pickle.load(f)
    except FileNotFoundError:
        print("No optimized hyperparameters found. Using default values.")
        params = {
            'sequence_length': 10,
            'd_model': 256,
            'nhead': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'dim_feedforward': 1024,
            'dropout': 0.1
        }
    
    # Load scaler
    with open('../out/model/foundation_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Create model with optimized architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FoundationModel(
        input_dim=14,  # Number of lab tests
        d_model=params['d_model'],
        nhead=params['nhead'],
        num_encoder_layers=params['num_encoder_layers'],
        num_decoder_layers=params['num_decoder_layers'],
        dim_feedforward=params['dim_feedforward'],
        dropout=params['dropout']
    )
    
    # Load trained weights
    model.load_state_dict(torch.load('../out/model/foundation_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, scaler, params

def preprocess_data(data, scaler, sequence_length):
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
    
    # Create sequences
    sequences = []
    test_indices = []
    
    for i in range(len(data_normalized) - sequence_length + 1):
        sequence = data_normalized[i:i + sequence_length]
        test_idx = np.arange(sequence.shape[1])
        test_idx = np.tile(test_idx, (sequence_length, 1))
        
        sequences.append(sequence)
        test_indices.append(test_idx)
    
    return np.array(sequences), np.array(test_indices)

def detect_anomalies(data, model, scaler, params, threshold=0.95):
    """Detect anomalies in the input data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preprocess data
    sequences, test_indices = preprocess_data(data, scaler, params['sequence_length'])
    
    # Convert to PyTorch tensors
    sequences = torch.tensor(sequences, dtype=torch.float32).to(device)
    test_indices = torch.tensor(test_indices, dtype=torch.long).to(device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(sequences, test_indices)
    
    # Calculate MSE
    mask = (sequences != -1).float()
    mse = ((outputs - sequences) ** 2 * mask).sum(dim=(1, 2)) / torch.clamp(mask.sum(dim=(1, 2)), min=1e-10)
    mse_per_test = ((outputs - sequences) ** 2 * mask).cpu().numpy()
    
    # Calculate anomaly scores
    anomaly_scores = mse.cpu().numpy()
    
    # Detect anomalies
    anomalies = anomaly_scores > threshold
    
    return anomalies, anomaly_scores, mse_per_test

def predict_next_values(data, model, scaler, params, num_predictions=1):
    """Predict future values based on current sequence."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preprocess data
    sequences, test_indices = preprocess_data(data, scaler, params['sequence_length'])
    
    # Convert to PyTorch tensors
    sequences = torch.tensor(sequences, dtype=torch.float32).to(device)
    test_indices = torch.tensor(test_indices, dtype=torch.long).to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model.predict_next_values(sequences, test_indices, num_predictions)
    
    # Convert predictions back to original scale
    predictions = predictions.cpu().numpy()
    predictions_original = np.zeros_like(predictions)
    
    for i in range(predictions.shape[2]):
        mask = predictions[:, :, i] != -1
        if np.sum(mask) > 0:
            predictions_original[mask, i] = scaler.inverse_transform(
                predictions[mask, i].reshape(-1, 1)
            ).flatten()
    
    return predictions_original

def analyze_data(data, threshold=0.95):
    """Analyze data for anomalies and return detailed results."""
    # Load model and scaler
    model, scaler, params = load_model_and_scaler()
    
    # Detect anomalies
    anomalies, anomaly_scores, mse_per_test = detect_anomalies(
        data, model, scaler, params, threshold
    )
    
    # Get detailed information about anomalies
    anomaly_details = get_anomaly_details(
        data, anomalies, anomaly_scores, mse_per_test
    )
    
    return {
        'anomalies_detected': anomalies.sum(),
        'total_samples': len(data),
        'anomaly_rate': anomalies.mean(),
        'anomaly_details': anomaly_details
    }

def get_anomaly_details(data, anomalies, anomaly_scores, mse_per_test):
    """Get detailed information about detected anomalies."""
    # Convert to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # Get anomalous samples
    anomalous_data = data[anomalies].copy()
    anomalous_scores = anomaly_scores[anomalies]
    anomalous_mse = mse_per_test[anomalies]
    
    # Create detailed report
    details = []
    for idx, (_, row) in enumerate(anomalous_data.iterrows()):
        # Find which tests contributed most to the anomaly
        test_scores = anomalous_mse[idx].mean(axis=0)  # Average over sequence length
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

class FoundationPredictor:
    def __init__(self, model_dir='../out/model'):
        """Initialize the predictor with saved model and scaler."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = Path(model_dir)
        
        # Load model and scaler
        self.model, self.scaler, self.params = load_model_and_scaler()
        
        # Lab test columns
        self.lab_columns = [
            'AFSE', 'ALTSE', 'CHOSE', 'GGTSE', 'HB', 'KALSE', 'KRESE', 'LEUC',
            'NATSE', 'TRISE', 'ALBSE', 'ASTSE', 'CRPSE', 'TRC'
        ]
    
    def predict(self, data, num_predictions=1):
        """
        Make predictions on input data.
        
        Args:
            data: Input data as pandas DataFrame or numpy array
            num_predictions: Number of future values to predict
            
        Returns:
            dict: Dictionary containing:
                - 'predictions': Predicted future values
                - 'anomaly_scores': Anomaly scores for each sample
                - 'anomalies': Boolean array indicating anomalies
        """
        # Prepare input
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
        
        # Get predictions
        predictions = predict_next_values(data, self.model, self.scaler, self.params, num_predictions)
        
        # Detect anomalies
        anomalies, anomaly_scores, _ = detect_anomalies(
            data, self.model, self.scaler, self.params
        )
        
        return {
            'predictions': predictions,
            'anomaly_scores': anomaly_scores,
            'anomalies': anomalies
        } 