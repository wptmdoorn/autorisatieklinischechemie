import numpy as np
from scipy.stats import lognorm

class CorrelationModel:
    def __init__(self, dist_params, corr_matrix, lab_columns, corr_threshold=0.3):
        """
        Initialize the correlation model.
        
        Args:
            dist_params: List of (shape, loc, scale) tuples for each lab test's log-normal distribution
            corr_matrix: Correlation matrix between lab tests
            lab_columns: List of lab test names
            corr_threshold: Minimum absolute correlation to consider
        """
        self.dist_params = dist_params
        self.corr_matrix = corr_matrix
        self.lab_columns = lab_columns
        self.corr_threshold = corr_threshold
        self.n_features = len(lab_columns)

    def _get_base_probability(self, values, test_idx):
        """Calculate base probability from log-normal distribution."""
        if self.dist_params[test_idx] is None:
            return np.nan
        
        shape, loc, scale = self.dist_params[test_idx]
        values = values + 1e-10  # Avoid negative/zero values
        return 1 - lognorm.cdf(values, s=shape, loc=loc, scale=scale)

    def _get_correlation_adjustment(self, data, test_idx, valid_mask):
        """Calculate correlation-based adjustment to probability."""
        corr_scores = np.ones(np.sum(valid_mask))
        
        for j in range(self.n_features):
            if test_idx == j or self.dist_params[j] is None:
                continue
                
            if abs(self.corr_matrix[test_idx, j]) > self.corr_threshold:
                valid_pair_mask = (data[:, test_idx] != -1) & (data[:, j] != -1)
                if np.sum(valid_pair_mask) == 0:
                    continue
                    
                shape_j, loc_j, scale_j = self.dist_params[j]
                values_j = data[valid_pair_mask, j] + 1e-10
                prob_j = lognorm.cdf(values_j, s=shape_j, loc=loc_j, scale=scale_j)
                
                # Weight by correlation strength
                corr_weight = abs(self.corr_matrix[test_idx, j]) / np.sum(
                    abs(self.corr_matrix[test_idx, self.corr_matrix[test_idx] > self.corr_threshold])
                )
                corr_scores[valid_pair_mask[valid_mask]] *= (1 - prob_j) ** corr_weight
        
        return corr_scores

    def predict(self, data):
        """
        Calculate probability scores for each lab test.
        
        Args:
            data: numpy array of shape (n_samples, n_features) with -1 for missing values
            
        Returns:
            probability_scores: numpy array of shape (n_samples, n_features) with probabilities
        """
        n_samples = data.shape[0]
        probability_scores = np.zeros_like(data, dtype=float)
        
        for i in range(self.n_features):
            if self.dist_params[i] is None:
                probability_scores[:, i] = np.nan
                continue
            
            valid_mask = data[:, i] != -1
            if np.sum(valid_mask) == 0:
                probability_scores[:, i] = np.nan
                continue
            
            # Get base probability from log-normal distribution
            values = data[valid_mask, i]
            base_scores = self._get_base_probability(values, i)
            
            # Get correlation-based adjustment
            corr_scores = self._get_correlation_adjustment(data, i, valid_mask)
            
            # Combine scores
            probability_scores[valid_mask, i] = base_scores * corr_scores
            probability_scores[~valid_mask, i] = np.nan
        
        return np.clip(probability_scores, 0, 1) 