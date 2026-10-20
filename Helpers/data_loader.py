import numpy as np
import sklearn.datasets as sk_datasets

def train_valid_test_split(X: np.ndarray, Y: np.ndarray, train: float=0.8, valid: float=0.1, test: float=0.1) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """
    Split features and labels into train, validation, and test sets.

    Args:
        X: Feature array of shape (n_samples, ...).
        Y: Label array of shape (n_samples, ...).
        train: Proportion of samples for training.
        valid: Proportion of samples for validation.
        test: Proportion of samples for testing.

    Returns:
        ((X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test))
    """
    
    if abs(1 - (train + valid + test)) > 0.0001:
        raise ValueError(f"Split proportions must sum to 1. The given proportions sum to {train + valid + test}.")
    
    if len(X) != len(Y):
        raise ValueError(f"X and Y must have the same length, got {len(X)} and {len(Y)}.")


    n_samples = len(Y)
    n_train = int(n_samples * train)
    n_valid = int(n_samples * valid)

    train_end = n_train
    valid_end = n_train + n_valid

    # Shuffle the data
    idx = np.random.default_rng(42).permutation(n_samples)
    X, Y = X[idx], Y[idx]

    return (
        (X[:train_end], Y[:train_end]),
        (X[train_end:valid_end], Y[train_end:valid_end]),
        (X[valid_end:], Y[valid_end:])
    )

def data_mean(X: np.ndarray) -> np.ndarray:
    """
    Per-feature mean over samples. Returns shape X.shape[1:]
    """
    return np.mean(X, axis=0)

def data_stdev(X: np.ndarray) -> np.ndarray:
    """
    Per-feature standard deviation over samples. Returns shape X.shape[1:]
    """
    return np.std(X, axis=0)

def normalize(X: np.ndarray, mean: np.ndarray, stdev: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize the given data using standardization.

    Args:
        X: Data array of shape (n_samples, ...).
        mean: Per-feature mean, computed from the training data only.
        stdev: Per-feature standard deviation, computed from the training data only.
        eps: Small constant to avoid division by zero for constant features.

    Returns:
        Array of the same shape as X.
    """
    return (X - mean) / (stdev + eps)

def normalized_split(X: np.ndarray, Y: np.ndarray, train: float=0.8, valid: float=0.1, test: float=0.1) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = train_valid_test_split(X, Y)

    mean = data_mean(X_train)
    stdev = data_stdev(X_train)

    X_train = normalize(X_train, mean, stdev)
    X_valid = normalize(X_valid, mean, stdev)
    X_test = normalize(X_test, mean, stdev)

    return (
        (X_train, Y_train), 
        (X_valid, Y_valid), 
        (X_test, Y_test)
    )

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