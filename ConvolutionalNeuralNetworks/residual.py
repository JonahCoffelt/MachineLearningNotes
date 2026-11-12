from NeuralNetworks.layer import Layer
from Helpers import np

class ResidualBlock(Layer):
    def __init__(self, layers: list[Layer], shortcut: Layer | None = None) -> None:
        """
        Args:
            layers: the inner layers f(x)
            shortcut: optional layer applied to the skip path, e.g. a 1x1 Conv2d (plus BatchNorm2d in a Sequential-style
                      wrapper) to match shape when f(x) changes the channels or spatial size. If None, the skip is the identity.
        """
        self.layers = layers 
        self.shortcut = shortcut
        super().__init__()

    def _all_layers(self) -> list[Layer]:
        return self.layers + ([self.shortcut] if self.shortcut is not None else [])

    @property
    def training(self) -> bool:
        return self._training

    @training.setter
    def training(self, value: bool) -> None:
        self._training = value
        for layer in self._all_layers():
            layer.training = value

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layers of the residual block and adds residual connection.
        No operation (including ReLU) is applied after the residual addition.
        Args:
            x: numpy array of input
        Returns:
            numpy array of output
        """

        x_save = x if self.shortcut is None else self.shortcut.forward(x)
        for layer in self.layers:
            x = layer.forward(x)
    
        return x + x_save

    def backprop(self, sensitivity: np.ndarray) -> np.ndarray:
        """
        Backpropagates through the inner layers and adds the gradient of the skip connection.
        Args:
            sensitivity: numpy array of gradient of the loss function with respect to the output of the last layer
        Returns:
            numpy array of gradient of the loss function with respect to the input of the first layer
        """

        grad = sensitivity
        for layer in reversed(self.layers):
            grad = layer.backprop(grad)

        skip = sensitivity if self.shortcut is None else self.shortcut.backprop(sensitivity)
        return grad + skip

    def params(self) -> list[np.ndarray]:
        layer_params = []
        for layer in self._all_layers():
            for param in layer.params():
                layer_params.append(param)
        return layer_params

    def grads(self) -> list[np.ndarray]:
        layer_grads = []
        for layer in self._all_layers():
            for grad in layer.grads():
                layer_grads.append(grad)
        return layer_grads