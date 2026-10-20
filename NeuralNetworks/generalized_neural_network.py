import numpy as np
from numpy._core import test
from Helpers import load_iris, normalized_split, load_mnist, train_valid_test_split, accuracy

rng = np.random.default_rng(42)

def softmax(v: np.ndarray) -> np.ndarray:
    v = v - np.max(v, axis=-1, keepdims=True)  # Normalize for saftey (this is pretty standard)
    exp_v = np.exp(v)
    return exp_v / np.sum(exp_v, axis=-1, keepdims=True)

class Layer:
    def __init__(self) -> None:
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backprop(self, sensitivity: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> list[np.ndarray]: 
        return []

    def grads(self) -> list[np.ndarray]: 
        return []

class Linear(Layer):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        """
        Create a new Linear layer.
        Args:
            in_dim: number of input features
            out_dim: number of output features
        """
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Initialize the weights and biases
        # self.W = rng.standard_normal((in_dim, out_dim)) * np.sqrt(2 / in_dim)
        # self.W = rng.random((in_dim, out_dim)) * 2 - 1
        self.W = rng.standard_normal((in_dim, out_dim)) * np.sqrt(2 / (in_dim + out_dim))
        self.b = np.zeros(out_dim)
        
        # Initalize the input and gradient states
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        Args:
            x: numpy array (n_samples, in_dim) of input features
        Returns:
            numpy array (n_samples, out_dim) of output features
        """
        self.x = x
        return x @ self.W + self.b

    def backprop(self, sensitivity: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of this layers weights and bias.
        Args:
            sensitivity: numpy array (n_samples, out_dim) of gradient of the loss function with respect to the output of this layer
        Returns:
            numpy array (n_samples, in_dim) of gradient of the loss function with respect to the input of this layer
        """
        
        self.dW = self.x.T @ sensitivity
        self.db = sensitivity.sum(axis=0)
        return sensitivity @ self.W.T

    def params(self) -> list[np.ndarray]:
        return [self.W, self.b]

    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]

class Tanh(Layer):
    def __init__(self) -> None:
        self.out = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        Args:
            x: numpy array (n_samples, in_dim) of input features
        Returns:
            numpy array (n_samples, out_dim) of output features
        """
        self.out = np.tanh(x)
        return self.out
    
    def backprop(self, sensitivity: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of this layer.
        Args:
            sensitivity: numpy array (n_samples, out_dim) of gradient of the loss function with respect to the output of this layer
        """
        return sensitivity * (1 - self.out**2)

class ReLU(Layer):
    def __init__(self) -> None:
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backprop(self, sensitivity: np.ndarray) -> np.ndarray:
        return sensitivity * self.mask

class Loss:
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    def backprop(self) -> np.ndarray:
        raise NotImplementedError

class CrossEntropy(Loss):
    def __init__(self) -> None:
        self.y = np.zeros(1)
        self.probs = np.zeros(1)

    def forward(self, logits, y) -> float:
        """
        Forward pass through the loss layer.
        Args:
            logits: numpy array (n_samples, out_dim) of logits
            y: numpy array (n_samples, out_dim) of target
        """

        self.probs = softmax(logits)
        self.y = y
        return -np.sum(y * np.log(np.clip(self.probs, 1e-12, 1.0))) / len(y)

    def backprop(self) -> np.ndarray:
        """
        Backward pass through the loss layer.
        Returns:
            numpy array (n_samples, out_dim) of gradient of the loss function with respect to the logits
        """
        return (self.probs - self.y) / len(self.y)

class NeuralNetwork:
    def __init__(self, layers: tuple[Layer, ...], loss_layer: Loss) -> None:
        self.layers = layers
        self.loss_layer = loss_layer

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
        n_samples = len(X)
        for epoch in range(epochs):
            order = rng.permutation(n_samples)
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
        return np.argmax(self.forward(X), axis=1)

def test_iris():
    # Create model
    print("Creating an iris model...")
    model = NeuralNetwork(
        (
            Linear(4, 8), 
            Tanh(), 
            Linear(8, 3)
        ),
        CrossEntropy()
    )

    # Load data
    print("Loading data...")
    X, Y = load_iris()
    (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = normalized_split(X, Y)

    # Train
    print("Begin training:")
    model.train(X_train, Y_train, learning_rate=0.01, batch_size=32, epochs=500)

    # Test
    print("Begin testing:")
    print("Iris test accuracy:", accuracy(model, X_test, Y_test))


def test_mnist():
    # Create model
    print("Creating an mnist model...")
    model = NeuralNetwork(
        (
            Linear(784, 128), 
            ReLU(), 
            Linear(128, 10)
        ),
        CrossEntropy()
    )

    # Load data
    print("Loading data...")
    X, Y = load_mnist()
    (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = train_valid_test_split(X, Y)

    mu, sigma = X_train.mean(), X_train.std()
    X_train = (X_train - mu) / sigma
    X_valid = (X_valid - mu) / sigma
    X_test = (X_test - mu) / sigma

    # Train
    print("Begin training:")
    model.train(X_train, Y_train, learning_rate=0.1, batch_size=64, epochs=30)

    # Test
    print("Begin testing:")
    print("MNIST test accuracy:", accuracy(model, X_test, Y_test))

if __name__ == "__main__":
    test_mnist()