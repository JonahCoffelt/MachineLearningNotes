import numpy as np
import sklearn.datasets as sk_datasets

def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Convert integer class labels to one-hot vectors.
    Args:
        labels: (n_samples,) array of integer class indices in [0, n_classes)
        n_classes: total number of classes
    Returns:
        (n_samples, n_classes) array with a single 1.0 per row
    """

    out = np.zeros((len(labels), n_classes))
    out[np.arange(len(labels)), labels] = 1.0
    return out

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

def load_iris() -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper for the sklearn load_iris.
    Returns:
        X: (150, 4) array of features
        Y: (150, 3) array of one-hot labels (0 = setosa, 1 = versicolor, 2 = virginica)
    """

    X, labels = sk_datasets.load_iris(return_X_y=True)
    return np.asarray(X), one_hot(np.asarray(labels), 3)