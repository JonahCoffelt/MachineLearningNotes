import numpy as np


def softmax(v: np.ndarray) -> np.ndarray:
    v = v - np.max(v, axis=-1, keepdims=True)  # Normalize for saftey (this is pretty standard)
    exp_v = np.exp(v)
    return exp_v / np.sum(exp_v, axis=-1, keepdims=True)


class Loss:
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    def backprop(self) -> np.ndarray:
        raise NotImplementedError


class CrossEntropy(Loss):
    def __init__(self) -> None:
        self.y = np.zeros(1)
        self.probs = np.zeros(1)

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Forward pass through the loss layer.
        Args:
            pred: numpy array (n_samples, out_dim) of logits
            target: numpy array (n_samples, out_dim) of y
        """

        self.probs = softmax(pred)
        self.y = target
        return -np.sum(target * np.log(np.clip(self.probs, 1e-12, 1.0))) / len(target)

    def backprop(self) -> np.ndarray:
        """
        Backward pass through the loss layer.
        Returns:
            numpy array (n_samples, out_dim) of gradient of the loss function with respect to the logits
        """
        return (self.probs - self.y) / len(self.y)