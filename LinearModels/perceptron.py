import numpy as np
from Helpers import load_breast_cancer


class Perceptron:
    """
    Implementation of a perceptron model using perceptron loss gradient descent training. 
    """

    def __init__(self, n_features: int) -> None:
        """
        Create a new Perceptron Model
        Args:
            n_features: The number of input features (dim of x), not including bias
        """
        
        self.rng = np.random.default_rng(42)
        self.weights = self.rng.random((n_features + 1))  # Add the bias term
        self.n_features = n_features

    def loss(self, z: float, y: float):
        """
        Returns the loss for a single sample
        Args:
            z: The prediction of the model
            y: The ground truth value given by the label
        """
        return max(0, -1 * y * z)

    def gradient(self, z: float, y: float, x_aug: np.ndarray) -> np.ndarray:
        """
        Returns the gradient with respect to the weights for a single sample
        Args:
            z: The prediction of the model
            y: The ground truth value given by the label
            x_aug: Numpy array (n_features + 1, ) of the input features with the bias term included
        """

        if y * z <= 0:
            return -y * x_aug
        return np.zeros((self.n_features + 1))

    def train(self, X: np.ndarray, Y: np.ndarray, learning_rate: float, batch_size: int, epochs: int) -> None:
        """
        Train the model on the given data. Uses -1/1 labels. 
        Args:
            X: numpy array (n_samples, n_features) of dataset features 
            Y: numpy array (n_samples, ) of dataset labels
            learning_rate: float value for the rate of gradient descent
            batch_size: number of samples per batch
            epochs: number of times to loop through entire dataset 
        """
        
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        for epoch in range(epochs):
            loss_accumulation = 0.0

            # Shuffle the samples each epoch
            order = self.rng.permutation(n_samples)

            for batch in range(n_batches):
                # Accumulate gradient for the batch
                gradient_accumulation = np.zeros((self.n_features + 1))
                for sample in range(batch_size):
                    sample_index = order[batch * batch_size + sample]
                    x = X[sample_index]
                    y = Y[sample_index]

                    # Get the prediction given current weights
                    z = self.forward(x)

                    # Add the loss to the accumulation
                    loss_accumulation += self.loss(z, y)

                    # Add the gradient to the accumulation
                    gradient_accumulation += self.gradient(z, y, x)

                # Get the mean gradient
                average_gradient = gradient_accumulation / batch_size

                # Add the gradient to the weights
                self.weights -= average_gradient * learning_rate

            # Print the current in-sample error
            print(f"Epoch {epoch + 1} - Loss {loss_accumulation / (n_batches * batch_size)}")

    def forward(self, x_aug: np.ndarray) -> int | np.ndarray:
        """
        Predict with input features that already include the bias term
        Args:
            x_aug: numpy array (n_features + 1, ) of the input features and the bias
        """

        return x_aug @ self.weights

    def predict(self, x: np.ndarray) -> int | np.ndarray:
        """
        Get the prediction for a single set of input features. 
        Input should not include the bias term
        Args:
            x: numpy array (n_features, ) of the input features, does not include bias
        """

        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :]
        x_aug = np.hstack([np.ones((x.shape[0], 1)), x])

        return np.where(self.forward(x_aug) >= 0, 1, -1)

X, Y = load_breast_cancer()
Y = np.where(Y > 0, 1, -1)
model = Perceptron(X.shape[1])
model.train(X, Y, learning_rate=0.1, batch_size=32, epochs=50)
preds = model.predict(X)
print("Breast cancer accuracy:", (preds == Y).mean())