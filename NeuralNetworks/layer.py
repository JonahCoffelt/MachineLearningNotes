import numpy as np


class Layer:
    def __init__(self) -> None:
        self.training = False
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backprop(self, sensitivity: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> list[np.ndarray]: 
        return []

    def grads(self) -> list[np.ndarray]: 
        return []