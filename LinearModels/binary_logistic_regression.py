import numpy as np
from Helpers import load_breast_cancer

def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    """
    Returns the sigmoid value for the give value or numpy array of values. 
    If given as an array, applies element-wise. 
    """
    
    return 1 / (1 + np.exp(-1 * z))

class BinaryLogisticRegression:
    """
    Implementation of a binary logistic regression model using log loss gradient descent training. 
    """

    def __init__(self, n_features: int) -> None:
        """
        Create a new Binary Logistic Regression Model
        Args:
            n_features: The number of input features (dim of x), not including bias
        """
        
        self.rng = np.random.default_rng(42)
        self.weights = self.rng.random((n_features + 1))  # Add the bias term
        self.n_features = n_features

    def loss(self, y_hat: float, y: float):
        """
        Returns the loss for a single sample
        Args:
            y_hat: The prediction of the model
            y: The ground truth value given by the label
        """
        y_hat = np.clip(y_hat, 1e-12, 1 - 1e-12)
        return -1 * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def gradient(self, y_hat: float, y: float, x_aug: np.ndarray) -> np.ndarray:
        """
        Returns the gradient with respect to the weights for a single sample
        Args:
            y_hat: The prediction of the model
            y: The ground truth value given by the label
            x_aug: Numpy array (n_features + 1, ) of the input features with the bias term included
        """

        return x_aug * (y_hat - y)

    def train(self, X: np.ndarray, Y: np.ndarray, learning_rate: float, batch_size: int, epochs: int) -> None:
        """
        Train the model on the given data. Uses 0/1 labels. 
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
                    y_hat = self.forward(x)

                    # Add the loss to the accumulation
                    loss_accumulation += self.loss(y_hat, y)

                    # Add the gradient to the accumulation
                    gradient_accumulation += self.gradient(y_hat, y, x)

                # Get the mean gradient
                average_gradient = gradient_accumulation / batch_size

                # Add the gradient to the weights
                self.weights -= average_gradient * learning_rate

            # Print the current in-sample error
            print(f"Epoch {epoch + 1} - Loss {loss_accumulation / (n_batches * batch_size)}")

    def forward(self, x_aug: np.ndarray) -> float | np.ndarray:
        """
        Predict with input features that already include the bias term
        Args:
            x_aug: numpy array (n_features + 1, ) of the input features and the bias
        """

        return sigmoid(x_aug @ self.weights)

    def predict(self, x: np.ndarray) -> float | np.ndarray:
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

        return self.forward(x_aug)

X, Y = load_breast_cancer()
model = BinaryLogisticRegression(X.shape[1])
model.train(X, Y, learning_rate=0.1, batch_size=32, epochs=50)
preds = model.predict(X) > 0.5
print("Breast cancer accuracy:", (preds == Y).mean())