from NeuralNetworks.layer import Layer
import numpy as np


class Dropout(Layer):
    def __init__(self, drop_probability: float=0.5) -> None:
        super().__init__()
        self.rng = np.random.default_rng(42)

        self.drop_probability = drop_probability
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Identity during prediction
        if not self.training or self.drop_probability == 0:
            return x

        # Generate new mask on forward
        self.mask = (self.rng.random(x.shape) >= self.drop_probability)

        # Divide by (1 - p) so that prediction stays consistent
        self.mask = self.mask / (1 - self.drop_probability)
        return x * self.mask

    def backprop(self, sensitivity: np.ndarray) -> np.ndarray:
        if not self.training or self.drop_probability == 0:
            return sensitivity
        return sensitivity * self.mask