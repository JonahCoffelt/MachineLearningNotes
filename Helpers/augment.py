from .numpy_cupy import np


class RandomFlipCrop:

    def __init__(self, pad: int = 4, flip: bool = True, seed: int = 42) -> None:
        """
        Args:
            pad: pixels of zero padding on each side before the random crop. 0 disables cropping.
            flip: whether to randomly flip horizontally. Not appropriate for e.g. digits.
            seed: seed for the random number generator
        """
        self.pad = pad
        self.flip = flip
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (N, C, H, W) batch of images
        Returns:
            (N, C, H, W) augmented batch. The input is not modified.
        """
        N, C, H, W = x.shape

        if self.flip:
            flipped = self.rng.random(N) < 0.5
            x = np.where(flipped[:, None, None, None], x[:, :, :, ::-1], x)

        if self.pad > 0:
            p = self.pad
            x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))

            # A random top-left corner for each image's crop window
            top = self.rng.integers(0, 2 * p + 1, size=N)
            left = self.rng.integers(0, 2 * p + 1, size=N)
            rows = top[:, None] + np.arange(H)[None, :]    # (N, H)
            cols = left[:, None] + np.arange(W)[None, :]   # (N, W)

            # Gather every image's own window at once
            x = x_pad[np.arange(N)[:, None, None, None],
                      np.arange(C)[None, :, None, None],
                      rows[:, None, :, None],
                      cols[:, None, None, :]]

        return x
