import numpy as np
from Helpers import load_diabetes, evaluate_mse

class LinearRegressionModel:
    """
    Implementation of a linear regression model using closed form training. 
    """
    
    def __init__(self, n_features: int) -> None:
        """
        Create a new Linear Regression Model
        Args:
            n_features: The number of input features (dim of x), not including bias
        """
        
        self.rng = np.random.default_rng(42)
        self.weights = self.rng.random((n_features + 1))  # Add the bias term
        self.n_features = n_features

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Trains the model with the given inputs and labels
        Args:
            X: numpy array (n_samples, n_features) of dataset features 
            Y: numpy array (n_samples, ) of dataset labels
        """
        
        # Add the bias term
        X = np.hstack([X, np.ones((X.shape[0], 1))])

        # Solve for weights
        X_transpose = X.T
        self.weights = np.linalg.inv(X_transpose @ X) @ (X_transpose @ Y)

    def forward(self, x_aug: np.ndarray) -> float:
        """
        Predict with input features that already include the bias term
        Args:
            x_aug: numpy array (n_features + 1, ) of the input features and the bias

        """

        return float(self.weights.T @ x_aug)

    def predict(self, x: np.ndarray) -> float:
        """
        Get the prediction for a single set of input features. 
        Input should not include the bias term
        Args:
            x: numpy array (n_features, ) of the input features, does not include bias
        """

        x_aug = np.append(x, 1)
        return self.forward(x_aug)

X, Y = load_diabetes()
n_features = X.shape[1]
model = LinearRegressionModel(n_features)
model.train(X, Y)
print("Diabetes MSE: ", evaluate_mse(model, X, Y))