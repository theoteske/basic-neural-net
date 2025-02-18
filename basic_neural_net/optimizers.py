"""Gradient-based optimization algorithms for neural network training."""

import numpy as np
    
class SGDmomentum:
    """Stochastic gradient descent with momentum."""
    
    def __init__(self, beta = 0.9):
        """Initialize SGD with momentum.
        
        Args:
            beta: Momentum coefficient.
        """
        self.beta = beta
        self.param_W = None
        self.param_b = None

    def update(self, grad_W, grad_b):
        """Update parameters using momentum.
        
        Args:
            grad_W: Weight gradients.
            grad_b: Bias gradients.
        """
        if self.param_W is None:
            self.param_W = np.zeros_like(grad_W)
            self.param_b = np.zeros_like(grad_b)
            
        self.param_W = self.beta*self.param_W + (1-self.beta)*grad_W
        self.param_b = self.beta*self.param_b + (1-self.beta)*grad_b
        
class AGDmomentum:
    """Nesterov's accelerated gradient descent with momentum."""
    
    def __init__(self, beta = 0.9):
        """Initialize Nesterov AGD.
        
        Args:
            beta: Momentum coefficient.
        """
        self.beta = beta
        self.param_W = None
        self.param_b = None

    def update(self, grad_W, grad_b):
        """Update parameters using Nesterov momentum.
        
        Args:
            grad_W: Weight gradients.
            grad_b: Bias gradients.
        """
        if self.param_W is None:
            self.param_W = np.zeros_like(grad_W)
            self.param_b = np.zeros_like(grad_b)
            
        self.param_W = self.beta*self.param_W + (1-self.beta)*grad_W
        self.param_b = self.beta*self.param_b + (1-self.beta)*grad_b
    
class RMSprop:
    """RMSprop optimizer with gradient clipping."""
    
    def __init__(self, beta = 0.90, eps = 1e-7):
        """Initialize RMSprop.
        
        Args:
            beta: Decay rate for moving average.
            eps: Small constant for numerical stability.
        """
        self.eps = eps
        self.beta = beta
        self.param_W = None
        self.param_b = None
        self.cache_W = None
        self.cache_b = None
        
    def update(self, grad_W, grad_b):
        """Update parameters using RMSprop.
        
        Args:
            grad_W: Weight gradients.
            grad_b: Bias gradients.
        """
        if self.param_W is None:
            self.param_W = np.zeros_like(grad_W)
            self.param_b = np.zeros_like(grad_b)
            self.cache_W = np.zeros_like(grad_W)
            self.cache_b = np.zeros_like(grad_b)
        
        self.cache_W = self.beta * self.cache_W + (1 - self.beta) * (np.clip(grad_W, -5,5)**2)
        self.cache_b = self.beta * self.cache_b + (1 - self.beta) * (np.clip(grad_b, -5,5)**2)
        
        self.param_W = (np.clip(grad_W, -5,5)) / (np.sqrt(self.cache_W) + self.eps)
        self.param_b = (np.clip(grad_b, -5,5)) / (np.sqrt(self.cache_b) + self.eps)
        
        
class Adam:
    """Adam optimizer with gradient clipping."""
    
    def __init__(self, beta1 = 0.9, beta2 = 0.999, eps = 1e-7):
        """Initialize Adam.
        
        Args:
            beta1: Decay rate for first moment estimate.
            beta2: Decay rate for second moment estimate.
            eps: Small constant for numerical stability.
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.cache_W = None
        self.cache_b = None
        self.cache_WW = None
        self.cache_bb = None
        self.param_W = None
        self.param_b = None
        
    def update(self, grad_W, grad_b):
        """Update parameters using Adam.
        
        Args:
            grad_W: Weight gradients.
            grad_b: Bias gradients.
        """
        if self.cache_W is None:
            self.cache_W = np.zeros_like(grad_W)
            self.cache_WW = np.zeros_like(grad_W)
            self.cache_b = np.zeros_like(grad_b)
            self.cache_bb = np.zeros_like(grad_b)
            
        self.cache_W = self.beta1*self.cache_W + (1-self.beta1)*np.clip(grad_W, -5,5)
        self.cache_b = self.beta1*self.cache_b + (1-self.beta1)*np.clip(grad_b, -5,5)
                
        self.cache_WW = self.beta1*self.cache_WW + (1-self.beta1)*(np.clip(grad_W, -5,5)**2)
        self.cache_bb = self.beta1*self.cache_bb + (1-self.beta1)*(np.clip(grad_b, -5,5)**2)
        
        self.param_W = 1/(np.sqrt(self.cache_WW) + self.eps) * self.cache_W
        self.param_b = 1/(np.sqrt(self.cache_bb) + self.eps) * self.cache_b
        
        