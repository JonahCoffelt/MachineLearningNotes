from ConvolutionalNeuralNetworks import *
from NeuralNetworks import *
from Helpers import accuracy, load_mnist

def get_linear_model(in_features: int, n_classes: int) -> NeuralNetwork:
    model = NeuralNetwork(
        (
            Flatten(),
            Linear(in_features, n_classes)
        ),
        CrossEntropy()
    )

    return model

def get_neural_network(in_features: int, n_classes: int, n_hidden_layers: int=2, hidden_size: int=128) -> NeuralNetwork:
    layers = [Flatten(), Linear(in_features, hidden_size), ReLU()]
    for i in range(n_hidden_layers - 1):
        layers += [Linear(hidden_size, hidden_size), ReLU()]
    layers.append(Linear(hidden_size, n_classes))

    return NeuralNetwork(tuple(layers), CrossEntropy())

def get_LeNet(in_channels: int = 3, n_classes: int = 10, conv1_padding: int = 0) -> NeuralNetwork:
    model = NeuralNetwork(
        (
            Conv2d(in_channels, 6, 5, padding=conv1_padding),  # 32x32 input needs 0, 28x28 (MNIST) needs 2
            BatchNorm2d(6),
            ReLU(),
            MaxPool2d(2, 2),

            Conv2d(6, 16, 5, padding=0),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(2, 2),

            Flatten(),
            Linear(5 * 5 * 16, 120),
            BatchNorm1d(120),
            ReLU(),

            Linear(120, 84),
            BatchNorm1d(84),
            ReLU(),
            Dropout(0.3),

            Linear(84, n_classes)
        ),
        CrossEntropy()
    )
    
    return model

def get_ResNet(n_classes: int = 10, in_channels: int = 3, depth: int = 20) -> NeuralNetwork:
    """
    Builds a CIFAR-style pre-activation ResNet (He et al. 2016, section 4.2 layout).
    The network has 3 stages of n residual blocks each, with 16, 32, and 64 channels. The second and third stages
    downsample by 2 in their first block. Each block has two 3x3 convs, so depth = 6n + 2 (conv stem + 6n convs + linear).
    Args:
        depth: number of weighted layers. Must be of the form 6n + 2, e.g. 20, 32, 44, 56, 110
        n_classes: number of output classes
        in_channels: number of input channels
    Returns:
        NeuralNetwork taking (N, in_channels, H, W) inputs and producing (N, n_classes) logits
    """

    if depth < 8 or (depth - 2) % 6 != 0:
        raise ValueError(f"depth must be of the form 6n + 2 (8, 14, 20, 32, 44, 56, ...), got {depth}.")
    n = (depth - 2) // 6

    layers: list[Layer] = [Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)]

    channels = 16
    for stage, out_channels in enumerate((16, 32, 64)):
        for block in range(n):
            # Downsample at the start of every stage but the first
            stride = 2 if stage > 0 and block == 0 else 1

            inner = [
                BatchNorm2d(channels), ReLU(),
                Conv2d(channels, out_channels, kernel_size=3, stride=stride, padding=1),
                BatchNorm2d(out_channels), ReLU(),
                Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ]

            # Identity skip when the shape is unchanged, otherwise a 1x1 conv projection
            shortcut = None
            if stride != 1 or channels != out_channels:
                shortcut = Conv2d(channels, out_channels, kernel_size=1, stride=stride, padding=0)

            layers.append(ResidualBlock(inner, shortcut))
            channels = out_channels

    # Pre-activation blocks leave the last output un-normalized, so finish with BN + ReLU before pooling
    layers += [BatchNorm2d(channels), ReLU(), GlobalAvgPool2d(), Linear(channels, n_classes)]

    return NeuralNetwork(tuple(layers), CrossEntropy())