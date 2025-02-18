"""Implementations for the different kind of layers used to build neural networks."""

import numpy as np
from . import activations

class InputLayer:
    """Layer that handles network input with specified shape."""
    
    def __init__(self, input_shape):
        """Initialize input layer.
        
        Args:
            input_shape: Shape of input data.
        """
        self.input_shape = input_shape

    def forward(self, inputs):
        """Store input as output."""
        self.output = inputs
        
    def backward(self, d_output, learning_rate=0.1):
        """Pass gradient back unchanged."""
        return d_output

class Dense:
    """Fully connected layer computing Activation(Wx + b).
    
    For input x ∈ ℝⁿ, produces output in ℝʳ where:
    - W ∈ ℝʳˣⁿ (weights matrix)
    - b ∈ ℝʳ (bias vector)
    - Activation: ℝʳ → ℝʳ
    """
    def __init__(self, input_size, units, activation=activations.Identity, optimizer=None):
        """Initialize dense layer.
        
        Args:
            input_size: Input dimension n.
            units: Output dimension r.
            activation: Activation function class from activations.py.
            optimizer: Optional optimizer class from optimizers.py.
        """
        self.weights = np.random.randn(units, input_size) * np.sqrt(1.0 / input_size)
        self.biases = np.random.randn(units).reshape((units, 1))
        self.activation = activation()
        if optimizer is None:
            self.optimizer = None
        else:
            self.optimizer = optimizer()
        
    def forward(self, layer_input):
        """Compute layer output.
        
        Args:
            layer_input: Input tensor.
        """
        self.input = layer_input.reshape((-1, layer_input.shape[-1])) #reshaping in case
        self.output = self.activation.forward(self.weights @ self.input + self.biases)
        
    def backward(self, d_output, learning_rate = 0.1):
        """Compute gradients and update parameters.
        
        Args:
            d_output: Upstream gradient.
            learning_rate: Learning rate for parameter updates.
            
        Returns:
            Gradient with respect to layer input.
        """
        if isinstance(self.activation, activations.Softmax):
            dz = self.activation.backward(d_output)
        else:
            dz = (d_output.T * self.activation.backward()).T
            
        d_output = dz @ self.weights
        
        grad_W = dz.T @ self.input.T 
        grad_b = dz.sum(axis=0, keepdims=True).reshape(-1,1)
        
        if self.optimizer is None:
            self.weights = self.weights - learning_rate*grad_W
            self.biases = self.biases - learning_rate*grad_b
        else:
            self.optimizer.update(grad_W, grad_b)
            self.weights = self.weights - learning_rate*self.optimizer.param_W
            self.biases = self.biases - learning_rate*self.optimizer.param_b
            
        return d_output
        
