import numpy as np
import sklearn.datasets as sk_datasets


def load_diabetes() -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper for the sklearn load_diabetes.
    Returns:
        X: (442, 10) array of features
        Y: (442,) array of disease progression targets
    """

    X, Y = sk_datasets.load_diabetes(return_X_y=True)
    return np.asarray(X), np.asarray(Y)

def load_breast_cancer() -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper for the sklearn load_breast_cancer.
    Returns:
        X: (569, 30) array of features
        Y: (569,) array of binary labels (0 = malignant, 1 = benign)
    """

    X, Y = sk_datasets.load_breast_cancer(return_X_y=True)
    return np.asarray(X), np.asarray(Y, dtype=float)