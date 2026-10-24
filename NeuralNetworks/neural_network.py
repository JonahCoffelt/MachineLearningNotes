import numpy as np
from NeuralNetworks import *


class NeuralNetwork:
    def __init__(self, layers: tuple[Layer, ...], loss_layer: Loss) -> None:
        self.layers = layers
        self.loss_layer = loss_layer

        self.rng = np.random.default_rng(42)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the neural network.
        Args:
            x: numpy array (n_samples, in_dim) of input features
        Returns:
            numpy array (n_samples, out_dim) of output features
        """
        
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Forward pass through the loss layer.
        Args:
            pred: numpy array (n_samples, out_dim) of predicted features
            target: numpy array (n_samples, out_dim) of target features
        Returns:
            float: loss
        """

        return self.loss_layer.forward(pred, target)

    def backprop(self) -> None:
        """
        Backward pass through the neural network.
        """

        sensitivity = self.loss_layer.backprop()
        for layer in reversed(self.layers):
            sensitivity = layer.backprop(sensitivity)

    def train(self, X, Y, learning_rate, batch_size, epochs) -> None:
        """
        Train the neural network.
        Args:
            X: numpy array (n_samples, in_dim) of input features
            Y: numpy array (n_samples, out_dim) of target features
            learning_rate: float: learning rate
            batch_size: int: batch size
            epochs: int: number of epochs
        """

        for layer in self.layers:
            layer.training = True
        
        n_samples = len(X)
        for epoch in range(epochs):
            order = self.rng.permutation(n_samples)
            loss_accumulation = 0.0

            for start in range(0, n_samples, batch_size):
                idx = order[start:start + batch_size]

                x = X[idx]
                y = Y[idx]

                logits = self.forward(x)
                loss = self.loss(logits, y)
                self.backprop()

                for layer in self.layers:
                    for param, grad in zip(layer.params(), layer.grads()):
                        param -= grad * learning_rate

                loss_accumulation += loss * len(idx)

            print(f"Epoch {epoch + 1} - Loss {loss_accumulation / n_samples:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class of the input data.
        Args:
            X: numpy array (n_samples, in_dim) of input features
        Returns:
            numpy array (n_samples, out_dim) of predicted features
        """
        for layer in self.layers:
            layer.training = False

        return np.argmax(self.forward(X), axis=1)