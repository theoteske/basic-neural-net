"""Sequential model for building and training neural networks."""

import numpy as np
from scipy.special import softmax
import losses

class Sequential:
    """Sequential container for neural network layers."""
    
    def __init__(self):
        """Initialize empty sequential model."""
        self.layers = []

    def add(self, layer):
        """Add layer to model."""
        self.layers.append(layer)

    def forward(self, inputs):
        """Compute forward pass through all layers.
        
        Args:
            inputs: Input data.
            
        Returns:
            Model output.
        """
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.output
        return inputs

    def backward(self, d_output, learning_rate):
        """Compute backward pass through all layers.
        
        Args:
            d_output: Gradient of loss.
            learning_rate: Learning rate for parameter updates.
        """
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output, learning_rate)

    def train(self, X, y, epochs, learning_rate, batch_size, loss=losses.MSE):
        """Train model using mini-batch gradient descent.
        
        Args:
            X: Training data.
            y: Target values.
            epochs: Number of training epochs.
            learning_rate: Learning rate for parameter updates.
            batch_size: Mini-batch size.
            loss: Loss function class (default: MSE).
        
        Returns:
            List of training losses.
        """
        N = X.shape[1]
        n_batches = N // batch_size
        train_losses = []
        
        for epoch in range(epochs):
            indices = np.random.permutation(N)
            X_shuffle = X[:,indices]
            y_shuffle = y[:,indices]
            
            for i in range(n_batches):
                X_batch = X[:,i * batch_size : (i+1) * batch_size]
                y_batch = y[:,i * batch_size : (i+1) * batch_size]
                predictions = self.forward(X_batch)

                d_loss = loss().loss_gradient(predictions, y_batch)
                self.backward(d_loss, learning_rate)
                
            predictions = self.forward(X)
            l = loss().loss(predictions, y)
            train_losses.append(l)
        
        return train_losses

    def predict(self, X):
        """Generate predictions for input data.
        
        Args:
            X: Input data.
            
        Returns:
            Model predictions.
        """
        return self.forward(X)