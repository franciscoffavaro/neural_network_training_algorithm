"""Data loading and preprocessing utilities."""
import numpy as np
import pandas as pd

# supports ODS format for data input

def load_data(file_path, mode='single_output'):
    """Load and normalize data from ODS file.
    
    Args:
        file_path: Path to the ODS file
        mode: 'single_output' (default) predicts last column only
              'multi_output' predicts all 15 positions
    
    Returns:
        For single_output: (X, y) where y is 1D
        For multi_output: (X, y) where y has 15 columns
    """
    try:
        df = pd.read_excel(file_path, engine='odf')
        print(f"Loaded {df.shape[0]} samples with {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        
        if mode == 'multi_output':
            # X = all 15 positions, y = all 15 positions (for multi-output)
            # Skip Processo column (column 0)
            positions = df.iloc[:, 1:].values  # columns 1-15 (all positions)
            X = positions
            y = positions  # Same as X for now, will be shifted in load_sequence_data
            
            # Min-max normalization
            X_min, X_max = X.min(axis=0), X.max(axis=0)
            X = (X - X_min) / (X_max - X_min + 1e-8)
            y = (y - X_min) / (X_max - X_min + 1e-8)  # Use same scaling
            
            return X, y, X_min, X_max
        else:
            # Original behavior: predict only last column
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


def load_sequence_data(file_path):
    """Load data for sequence prediction: use row N to predict row N+1.
    
    Returns:
        X_seq: Previous row's 15 positions (normalized)
        y_seq: Next row's 15 positions (normalized)
        pos_min, pos_max: Min/max values for denormalization
    """
    try:
        df = pd.read_excel(file_path, engine='odf')
        print(f"Loaded {df.shape[0]} samples with {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Extract only position columns (skip Processo)
        positions = df.iloc[:, 1:].values  # Shape: (n_samples, 15)
        
        # Create sequence pairs: row[i] -> row[i+1]
        X_seq = positions[:-1]  # All rows except last
        y_seq = positions[1:]   # All rows except first
        
        print(f"Created {len(X_seq)} sequence pairs (row N → row N+1)")
        
        # Min-max normalization (same scaling for X and y)
        pos_min = positions.min(axis=0)
        pos_max = positions.max(axis=0)
        
        X_seq = (X_seq - pos_min) / (pos_max - pos_min + 1e-8)
        y_seq = (y_seq - pos_min) / (pos_max - pos_min + 1e-8)
        
        return X_seq, y_seq, pos_min, pos_max
        
    except Exception as e:
        print(f"Error loading sequence data: {e}")
        raise


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
