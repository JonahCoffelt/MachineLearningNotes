from NeuralNetworks.layer import Layer
from Helpers import np


class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Converts an array of shape (N, A, B, ..., C) to (N, A*B*...*C)
        """
        
        # Save the dimension to convert back during backprop
        self.shape = x.shape
        # Flatten all but sample dimension
        return x.reshape(x.shape[0], -1)

    def backprop(self, sensitivity: np.ndarray) -> np.ndarray:
        return sensitivity.reshape(self.shape)