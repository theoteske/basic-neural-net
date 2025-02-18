"""Common neural network activation functions with forward and backward pass implementations."""

import numpy as np
from scipy.special import softmax

class Sigmoid:
    """Sigmoid activation function σ(x) = 1/(1 + e^(-x))."""
    
    def forward(self, z):
        """Compute sigmoid activation.
        
        Args:
            z: Input array.
            
        Returns:
            Sigmoid activation output, clipped to prevent overflow.
        """
        self.input = z
        self.output = 1/(1+np.exp(-np.clip(z, -500, 500)))
        return self.output

    def backward(self):
        """Compute sigmoid derivative σ(x)(1 - σ(x)).
        
        Returns:
            Element-wise derivatives at input points.
        """
        derivative = self.output * (1-self.output)
        return derivative

class ReLu:
    """ReLU activation f(x) = max(0, x)."""
    
    def forward(self, z):
        """Compute ReLU activation.
        
        Args:
            z: Input array.
            
        Returns:
            ReLU activation output.
        """
        self.input = z
        return np.maximum(0, self.input)
    
    def backward(self):
        """Compute ReLU derivative (1 for x > 0, 0 otherwise).
        
        Returns:
            Element-wise derivatives at input points.
        """
        derivative = (self.input > 0)
        return derivative
    
class Softmax:
    """Softmax activation for converting inputs to probability distributions."""
    
    def forward(self, z):
        """Compute softmax probabilities column-wise.
        
        Args:
            z: Input array.
            
        Returns:
            Column-wise softmax probabilities.
        """
        self.input = z
        self.output = softmax(z, axis = 0) #softmax for each column
        return self.output

    def backward(self, d_output):
        """Compute softmax Jacobian.
        
        Args:
            d_output: Gradient of loss w.r.t. output.
            
        Returns:
            Gradient of softmax w.r.t. inputs.
        """
        p = self.output
        pp = np.einsum('ji, ki -> jki', p, p)
        diag = np.einsum('ji, kj -> jki',p, np.eye(p.shape[0]))
        return np.einsum('ij, jki -> ik', d_output, diag-pp)
    
class Identity:
    """Identity activation f(x) = x."""
    
    def forward(self, z):
        """Return input unchanged.
        
        Args:
            z: Input array.
            
        Returns:
            Input array unchanged.
        """
        self.input = z
        return z
    
    def backward(self):
        """Compute identity derivative (always 1).
        
        Returns:
            Array of ones matching input shape.
        """
        return np.ones_like(self.input)
