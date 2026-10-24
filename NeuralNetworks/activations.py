from NeuralNetworks.layer import Layer
import numpy as np


class Tanh(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.out = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        Args:
            x: numpy array (n_samples, in_dim) of input features
        Returns:
            numpy array (n_samples, out_dim) of output features
        """
        self.out = np.tanh(x)
        return self.out
    
    def backprop(self, sensitivity: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of this layer.
        Args:
            sensitivity: numpy array (n_samples, out_dim) of gradient of the loss function with respect to the output of this layer
        """
        return sensitivity * (1 - self.out**2)


class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backprop(self, sensitivity: np.ndarray) -> np.ndarray:
        return sensitivity * self.mask