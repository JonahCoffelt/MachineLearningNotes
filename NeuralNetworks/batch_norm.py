from NeuralNetworks.layer import Layer
from Helpers import np


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
        self.epsilon = 1e-5

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

        self.dg = np.sum(sensitivity * self.x_hat, axis=0)
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

class BatchNorm2d(Layer):
    def __init__(self, num_features: int, momentum: float = 0.1) -> None:
        """
        Batch normalization over feature maps. Statistics are computed per channel
        over the batch and both spatial dimensions (N, H, W).
        Args:
            num_features: number of channels C
        """
        super().__init__()

        self.num_features = num_features

        # Per-channel statistics, shaped (1, C, 1, 1) to broadcast against (N, C, H, W)
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_variance = np.ones((1, num_features, 1, 1))
        self.momentum = momentum

        self.batch_mean = np.zeros((1, num_features, 1, 1))
        self.batch_variance = np.ones((1, num_features, 1, 1))

        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.epsilon = 1e-5

        self.x = None
        self.x_hat = None
        self.dg = np.zeros(num_features)
        self.db = np.zeros(num_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: numpy array (N, C, H, W) of input feature maps
        Returns:
            numpy array (N, C, H, W) of normalized feature maps
        """
        # Keep track of inputs
        self.x = x

        # Default to running stats
        mean = self.running_mean
        variance = self.running_variance

        # In training use batch mean and variance, computed per channel over (N, H, W)
        if self.training:
            self.batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            self.batch_variance = np.var(x, axis=(0, 2, 3), keepdims=True)
            mean, variance = self.batch_mean, self.batch_variance

            m = x.shape[0] * x.shape[2] * x.shape[3]
            unbiased_var = variance * m / max(m - 1, 1)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_variance = (1 - self.momentum) * self.running_variance + self.momentum * unbiased_var

        # Normalize, scale, and shift
        self.x_hat = (x - mean) / np.sqrt(variance + self.epsilon)
        gamma = self.gamma.reshape(1, -1, 1, 1)
        beta = self.beta.reshape(1, -1, 1, 1)
        return self.x_hat * gamma + beta

    def backprop(self, sensitivity: np.ndarray) -> np.ndarray:
        """
        Args:
            sensitivity: numpy array (N, C, H, W) of gradient of the loss with respect to the output
        Returns:
            numpy array (N, C, H, W) of gradient of the loss with respect to the input
        """
        axes = (0, 2, 3)
        self.dg = np.sum(sensitivity * self.x_hat, axis=axes)
        self.db = np.sum(sensitivity, axis=axes)

        # Number of values each channel's statistics are computed over
        m = self.x.shape[0] * self.x.shape[2] * self.x.shape[3]
        inv_std = 1 / np.sqrt(self.batch_variance + self.epsilon)
        x_centered = self.x - self.batch_mean

        dl_dx_hat = sensitivity * self.gamma.reshape(1, -1, 1, 1)
        dl_dvar = np.sum(dl_dx_hat * x_centered * (-1 / 2) * inv_std ** 3, axis=axes, keepdims=True)
        dl_dmean = np.sum(dl_dx_hat * -inv_std, axis=axes, keepdims=True) + dl_dvar * np.sum(-2 * x_centered, axis=axes, keepdims=True) / m
        dl_dx = dl_dx_hat * inv_std + dl_dvar * 2 * x_centered / m + dl_dmean / m

        return dl_dx

    def params(self) -> list[np.ndarray]:
        return [self.gamma, self.beta]

    def grads(self) -> list[np.ndarray]:
        return [self.dg, self.db]
