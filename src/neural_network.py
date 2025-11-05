"""Simple feedforward neural network optimized by genetic algorithms."""
import numpy as np


class NeuralNetwork:
    """Simple feedforward neural network with one hidden layer."""
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    @staticmethod
    def _sigmoid(x):
        # clip to avoid overflow
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        """Forward pass through the network.
        
        Args:
            X: Input data of shape (n_samples, input_size)
            
        Returns:
            Network output of shape (n_samples, output_size)
        """
        self.z1 = X @ self.weights1 + self.bias1
        self.a1 = self._sigmoid(self.z1)
        
        self.z2 = self.a1 @ self.weights2 + self.bias2
        self.a2 = self._sigmoid(self.z2)
        
        return self.a2
    
    def predict(self, X):
        """Make predictions on input data."""
        return self.forward(X)
    
    def calculate_fitness(self, X, y):
        """Calculate fitness score using negative MSE.
        
        Higher fitness values indicate better performance.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Negative mean squared error
        """
        predictions = self.forward(X)
        mse = np.mean(np.square(predictions - y))
        return -mse
    
    def get_weights_as_array(self):
        """Flatten all network weights into a 1D array."""
        return np.concatenate([
            self.weights1.ravel(),
            self.bias1.ravel(),
            self.weights2.ravel(),
            self.bias2.ravel()
        ])
    
    def set_weights_from_array(self, weights_array):
        """Load weights from a 1D array.
        
        Args:
            weights_array: Flattened weight vector
        """
        idx = 0
        
        w1_size = self.input_size * self.hidden_size
        self.weights1 = weights_array[idx:idx+w1_size].reshape(
            self.input_size, self.hidden_size)
        idx += w1_size
        
        b1_size = self.hidden_size
        self.bias1 = weights_array[idx:idx+b1_size].reshape(1, self.hidden_size)
        idx += b1_size
        
        w2_size = self.hidden_size * self.output_size
        self.weights2 = weights_array[idx:idx+w2_size].reshape(
            self.hidden_size, self.output_size)
        idx += w2_size
        
        b2_size = self.output_size
        self.bias2 = weights_array[idx:idx+b2_size].reshape(1, self.output_size)
    
    def get_weights_count(self):
        """Get total number of trainable parameters."""
        return (self.input_size * self.hidden_size + self.hidden_size +
                self.hidden_size * self.output_size + self.output_size)
