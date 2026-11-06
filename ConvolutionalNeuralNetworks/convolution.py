from NeuralNetworks.layer import Layer
from Helpers import np, sliding_window_view

class Conv2d(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, padding: int=1) -> None:
        super().__init__()
        self.rng = np.random.default_rng(42)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernels = self.rng.standard_normal((out_channels, in_channels, kernel_size, kernel_size)) * np.sqrt(2 / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros((out_channels))

        self.x_pad = None
        self.cols = None
        self.dk = None
        self.db = None


    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        Args:
            x: numpy array (n_samples, in_channels, H, W) of input feature maps
        Returns:
            numpy array (n_samples, out_channels, H_out, W_out) of output feature maps
        """

        # Pad and save input
        x_pad = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        self.x_pad = x_pad

        # Compute output size
        N, C, H, W = x.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # im2col windows with numpy sliding_window_view
        windows = sliding_window_view(x_pad, (self.kernel_size, self.kernel_size), axis=(2, 3))
        # Slice to produce stride
        windows = windows[:,:,::self.stride,::self.stride]
        # Get cols
        cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(N * H_out * W_out, -1)
        self.cols = cols

        # GEMM for output
        k_flat = self.kernels.reshape(self.out_channels, -1)
        out = cols @ k_flat.T + self.b
        return out.reshape(N, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)

    def backprop(self, sensitivity: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of this layers kernels and bias.
        Args:
            sensitivity: numpy array (N, out_channels, H_out, W_out) of gradient of the loss function with respect to the output of this layer
        Returns:
            numpy array (n_samples, in_channels, H, W) of gradient of the loss function with respect to the input of this layer
        """

        # Compute output size
        N, C, H_pad, W_pad = self.x_pad.shape
        _, _, H_out, W_out = sensitivity.shape
        H = H_pad - 2 * self.padding
        W = W_pad - 2 * self.padding     

        dout = sensitivity.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        self.dk = (dout.T @ self.cols).reshape(self.kernels.shape)
        self.db = dout.sum(axis=0)
        k_flat = self.kernels.reshape(self.out_channels, -1)
        dcols = dout @ k_flat

        dc = dcols.reshape(N, H_out, W_out, C, self.kernel_size, self.kernel_size)
        dx_pad = np.zeros_like(self.x_pad)
        for u in range(self.kernel_size):
            for v in range(self.kernel_size):
                dx_pad[:, :, u:u + self.stride*H_out:self.stride, v:v + self.stride*W_out:self.stride] += dc[..., u, v].transpose(0, 3, 1, 2)

        return dx_pad[:, :, self.padding:self.padding+H, self.padding:self.padding+W]

    def params(self) -> list[np.ndarray]:
        return [self.kernels, self.b]

    def grads(self) -> list[np.ndarray]:
        return [self.dk, self.db]


if __name__ == "__main__":
    layer = Conv2d(3, 3)

    test_input = np.arange(2 * 3 * 5 * 5, dtype=float).reshape(2, 3, 5, 5)

    out = layer.forward(test_input)