from NeuralNetworks.layer import Layer
from Helpers import np, sliding_window_view

class MaxPool2d(Layer):
    def __init__(self, kernel_size: int = 2, stride: int | None = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.x_shape = None
        self.argmax = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (N, C, H, W)
        Returns:
            (N, C, H_out, W_out)
        """
        k, s = self.kernel_size, self.stride
        N, C, H, W = x.shape
        self.x_shape = x.shape

        windows = sliding_window_view(x, (k, k), axis=(2, 3))[:, :, ::s, ::s]
        Ho, Wo = windows.shape[2], windows.shape[3]
        windows = windows.reshape(N, C, Ho, Wo, k * k)

        self.argmax = windows.argmax(axis=-1)
        return windows.max(axis=-1)

    def backprop(self, sensitivity: np.ndarray) -> np.ndarray:
        """
        Args:
            sensitivity: (N, C, H_out, W_out)
        Returns:
            (N, C, H, W)
        """
        k, s = self.kernel_size, self.stride
        _, _, Ho, Wo = sensitivity.shape
        dx = np.zeros(self.x_shape)

        for u in range(k):
            for v in range(k):
                mask = (self.argmax == u * k + v)
                dx[:, :, u:u + s * Ho:s, v:v + s * Wo:s] += sensitivity * mask

        return dx