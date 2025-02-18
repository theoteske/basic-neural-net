"""Loss functions for neural network training."""

import numpy as np

class MSE:
    """Mean Squared Error (L2) loss function."""
    
    def loss(self, predictions, y):
        """Compute MSE loss.
        
        Args:
            predictions: Model predictions.
            y: True values.
            
        Returns:
            Mean squared error averaged over samples.
        """
        l = np.mean((y - predictions)**2)
        return l
    
    def loss_gradient(self, predictions, y):
        """Compute gradient of MSE loss.
        
        Args:
            predictions: Model predictions.
            y: True values.
            
        Returns:
            Gradient with respect to predictions.
        """
        grad = -2*(y - predictions)/y.shape[1]
        return grad.T
    
class CrossEntropy:
    """Cross-entropy loss function with numerical stability."""
    
    def loss(self, predictions, y):
        """Compute cross-entropy loss.
        
        Args:
            predictions: Model predictions.
            y: True values.
            
        Returns:
            Cross-entropy loss averaged over samples.
        """
        l = -np.sum(y*np.log(predictions+1e-6))/y.shape[1] # add 1e-6 to the argument of the logarithm for numerical stability
        return l.T
    
    def loss_gradient(self, predictions, y):
        """Compute gradient of cross-entropy loss.
        
        Args:
            predictions: Model predictions.
            y: True values.
            
        Returns:
            Gradient with respect to predictions.
        """
        grad = (-1/y.shape[1]) * (y / (predictions + 1e-6))
        return grad.T
