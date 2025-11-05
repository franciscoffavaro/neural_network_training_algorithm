"""Data loading and preprocessing utilities."""
import numpy as np
import pandas as pd

# supports ODS format for data input

def load_data(file_path):
    """Load and normalize data from ODS file."""
    try:
        df = pd.read_excel(file_path, engine='odf')
        print(f"Loaded {df.shape[0]} samples with {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.reshape(-1, 1)
        
        # Min-max normalization
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X = (X - X_min) / (X_max - X_min + 1e-8)
        
        y_min, y_max = y.min(), y.max()
        y = (y - y_min) / (y_max - y_min + 1e-8)
        
        return X, y
        
    except Exception as e:
        print(f"Warning: Could not load file - {e}")
        print("Generating synthetic data for demonstration...")
        
        np.random.seed(42)
        n_samples, n_features = 100, 3
        
        X = np.random.rand(n_samples, n_features)
        y = (np.sin(X[:, 0] * 2) + 
             np.cos(X[:, 1] * 3) + 
             X[:, 2]**2).reshape(-1, 1)
        y = (y - y.min()) / (y.max() - y.min())
        
        return X, y


def split_data(X, y, train_ratio=0.8):
    """Split data into training and test sets.
    
    Args:
        X: Feature matrix
        y: Target values
        train_ratio: Proportion of data for training (default: 0.8)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * train_ratio)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
