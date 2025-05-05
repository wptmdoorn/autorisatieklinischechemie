import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from .model import CorrelationModel

class CorrelationPredictor:
    def __init__(self, model_dir='../out/model'):
        """Initialize the predictor with saved model parameters."""
        self.model_dir = Path(model_dir)
        
        # Load model parameters
        dist_params_path = self.model_dir / 'correlation_dist_params.pkl'
        corr_matrix_path = self.model_dir / 'correlation_matrix.pkl'
        
        if not dist_params_path.exists() or not corr_matrix_path.exists():
            raise FileNotFoundError("Model parameter files not found")
            
        with open(dist_params_path, 'rb') as f:
            dist_params = pickle.load(f)
        with open(corr_matrix_path, 'rb') as f:
            corr_matrix = pickle.load(f)
        
        # Lab test columns
        self.lab_columns = [
            'AFSE', 'ALTSE', 'CHOSE', 'GGTSE', 'HB', 'KALSE', 'KRESE', 'LEUC',
            'NATSE', 'TRISE', 'ALBSE', 'ASTSE', 'CRPSE', 'TRC'
        ]
        
        # Initialize model
        self.model = CorrelationModel(dist_params, corr_matrix, self.lab_columns)

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
        
        return data

    def predict(self, data):
        """
        Make predictions on input data.
        
        Args:
            data: Input data as pandas DataFrame or numpy array
            
        Returns:
            dict: Dictionary containing:
                - 'total_score': Overall feasibility score for each sample
                - 'test_scores': Individual scores for each lab test
        """
        # Prepare input
        data_array = self._prepare_input(data)
        
        # Make predictions
        test_scores = self.model.predict(data_array)
        
        # Calculate total score as mean of non-nan test scores
        total_score = np.nanmean(test_scores, axis=1)
        
        return {
            'total_score': total_score,
            'test_scores': test_scores
        } 