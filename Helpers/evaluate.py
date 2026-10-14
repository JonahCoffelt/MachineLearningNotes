import numpy as np
from .data_loader import load_diabetes

def evaluate_accuracy(model, X: np.ndarray, Y: np.ndarray) -> float:
    n_samples = X.shape[0]
    correct = 0
    for i in range(n_samples):
        y_pred = model.predict(X[i])
        if np.abs(y_pred - Y[i]) < 1e-6:
            correct += 1
    return correct / n_samples


def evaluate_mse(model, X: np.ndarray, Y: np.ndarray) -> float:
    predictions = np.array([model.predict(x) for x in X])
    return float(np.mean((predictions - Y) ** 2))