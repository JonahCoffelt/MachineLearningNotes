import numpy as np
from Helpers import load_iris


def softmax(v: np.ndarray) -> np.ndarray:
    v = v - np.max(v, axis=-1, keepdims=True)  # Normalize for saftey (this is pretty standard)
    exp_v = np.exp(v)
    return exp_v / np.sum(exp_v, axis=-1, keepdims=True)

class MulticlassLogisticRegression:
    """
    Implementation of a multiclass logistic regression model using cross entropy loss gradient descent training. 
    """

    def __init__(self, n_features: int, n_classes: int) -> None:
        """
        Create a new Multiclass Logistic Regression Model
        Args:
            n_features: The number of input features (dim of x), not including bias
            n_classes: The number of prediction classes
        """
        
        self.rng = np.random.default_rng(42)
        self.weights = self.rng.random((n_features + 1, n_classes))  # Add the bias term
        self.n_features = n_features
        self.n_classes = n_classes

    def loss(self, y_hat: np.ndarray, y: np.ndarray):
        """
        Returns the cross entropy loss for a single sample
        Args:
            y_hat: The prediction of the model
            y: The ground truth (one-hot vector) given by the label
        """

        y_hat = np.clip(y_hat, 1e-12, 1.0)
        return -np.sum(y * np.log(y_hat))

    def gradient(self, y_hat: np.ndarray, y: np.ndarray, x_aug: np.ndarray) -> np.ndarray:
        """
        Returns the gradient with respect to the weights for a single sample
        Args:
            y_hat: The prediction of the model
            y: The ground truth value given by the label
            x_aug: Numpy array (n_features + 1, ) of the input features with the bias term included
        """

        return np.outer(x_aug, y_hat - y)
        
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
                gradient_accumulation = np.zeros((self.n_features + 1, self.n_classes))
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

    def forward(self, x_aug: np.ndarray) -> np.ndarray:
        """
        Predict with input features that already include the bias term
        Args:
            x_aug: numpy array (n_features + 1, ) of the input features and the bias
        """

        return softmax(x_aug @ self.weights)

    def predict(self, x: np.ndarray) -> np.ndarray:
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

        return np.argmax(self.forward(x_aug), axis=1)

X, Y = load_iris()
model = MulticlassLogisticRegression(X.shape[1], Y.shape[1])
model.train(X, Y, learning_rate=0.01, batch_size=16, epochs=150)
preds = model.predict(X)
print("Iris accuracy:", (preds == np.argmax(Y, axis=1)).mean())