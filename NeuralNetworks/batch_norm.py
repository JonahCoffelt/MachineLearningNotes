from NeuralNetworks.layer import Layer
import numpy as np


class BatchNorm1d(Layer):
    def __init__(self, num_features: int, momentum: float = 0.1) -> None:
        super().__init__()

        self.num_features = num_features

        self.running_mean = 0.0
        self.running_variance = 1.0
        self.momentum = momentum

        self.batch_mean = 0.0
        self.batch_variance = 1.0

        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.epsilon = 0.0

        self.x = np.zeros(num_features)
        self.x_hat = np.zeros(num_features)
        self.dg = np.zeros(num_features)
        self.db = np.zeros(num_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Keep track of inputs
        self.x = x

        # Default to running stats
        mean = self.running_mean
        variance = self.running_variance

        # In training use batch mean and stdev
        if self.training:
            self.batch_mean = np.mean(x, axis=0)
            self.batch_variance = np.var(x, axis=0)
            mean, variance = self.batch_mean, self.batch_variance

            m = len(x)
            unbiased_var = variance * m / max(m - 1, 1)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_variance = (1 - self.momentum) * self.running_variance + self.momentum * unbiased_var


        # Normalize, scale, and shift
        self.x_hat = (x - mean) / np.sqrt(variance + self.epsilon)
        z = (self.x_hat * self.gamma) + self.beta

        return z

    def backprop(self, sensitivity: np.ndarray) -> np.ndarray:
        # TODO

        self.dg = np.sum(sensitivity * self.x, axis=0)
        self.db = np.sum(sensitivity, axis=0)

        m = len(self.x)
        dl_dx_hat = sensitivity * self.gamma
        dl_dvar = np.sum(dl_dx_hat * (self.x - self.batch_mean) * (-1 / 2) * np.power(self.batch_variance + self.epsilon, -3 / 2), axis=0)
        dl_dmean = np.sum(dl_dx_hat * (-1 / np.sqrt(self.batch_variance + self.epsilon)), axis=0) + dl_dvar * (np.sum(-2 * (self.x - self.batch_mean), axis=0) / m)
        dl_dx = dl_dx_hat * (1 / np.sqrt(self.batch_variance + self.epsilon)) + dl_dvar * (2 * (self.x - self.batch_mean) / m) + dl_dmean / m

        return dl_dx

    def params(self) -> list[np.ndarray]:
        return [self.gamma, self.beta]

    def grads(self) -> list[np.ndarray]:
        return [self.dg, self.db]