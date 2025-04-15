import pandas as pd
import numpy as np
from .inference import AutoencoderPredictor

def test_predictor():
    """Test the autoencoder predictor with sample data."""
    # Initialize predictor
    predictor = AutoencoderPredictor()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'AFSE': [10, 20, np.nan],
        'ALTSE': [30, 40, 50],
        'CHOSE': [5, 6, 7],
        'GGTSE': [25, 30, 35],
        'HB': [8.5, 9.0, 9.5],
        'KALSE': [4.0, 4.1, 4.2],
        'KRESE': [100, 110, 120],
        'LEUC': [5.0, 6.0, 7.0],
        'NATSE': [140, 142, 144],
        'TRISE': [1.0, 1.1, 1.2],
        'ALBSE': [40, 42, 44],
        'ASTSE': [25, 30, 35],
        'CRPSE': [5, 6, 7],
        'TRC': [200, 220, 240]
    })
    
    # Make predictions
    results = predictor.predict(sample_data)
    
    # Print results
    print("\nSample Data:")
    print(sample_data)
    
    print("\nTotal Feasibility Scores:")
    for i, score in enumerate(results['total_score']):
        print(f"Sample {i+1}: {score:.4f}")
    
    print("\nTest-specific Scores:")
    for i, test in enumerate(predictor.lab_columns):
        print(f"\n{test}:")
        for j, score in enumerate(results['test_scores'][:, i]):
            print(f"  Sample {j+1}: {score:.4f}" if not np.isnan(score) else f"  Sample {j+1}: missing")
    
    print("\nMSE Values:")
    for i, mse in enumerate(results['mse']):
        print(f"Sample {i+1}: {mse:.4f}")

if __name__ == "__main__":
    test_predictor() 