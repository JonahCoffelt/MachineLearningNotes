from .numpy_cupy import np
import numpy as _np
import sklearn.datasets as sk_datasets
import hashlib
import os
import pickle
import tarfile
import urllib.request
from pathlib import Path

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
    idx = _np.random.default_rng(42).permutation(n_samples)
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

def accuracy(model, X, Y):
    return (model.predict(X) == np.argmax(Y, axis=1)).mean()

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

def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper for the sklearn fetch_openml MNIST dataset.
    Returns:
        X: (70000, 784) array of flattened 28x28 images, scaled to [0, 1]
        Y: (70000, 10) array of one-hot labels (digits 0-9)
    """

    X, labels = sk_datasets.fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff"
    )
    X = _np.asarray(X, dtype=_np.float32) / 255.0
    labels = _np.asarray(labels).astype(int)
    return np.asarray(X), np.asarray(one_hot(labels, 10), dtype=np.float32)

_CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"

def load_cifar10(cache_dir: str | os.PathLike | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the CIFAR-10 dataset, downloading it on first use and caching it locally.
    The parsed arrays are cached as a compressed .npz so later calls skip the download
    and the archive parsing.
    Args:
        cache_dir: where to store the cache. Defaults to ~/.cache/machine_learning_notes/cifar10
    Returns:
        X: (60000, 3, 32, 32) array of images in (N, C, H, W) layout, scaled to [0, 1]
        Y: (60000, 10) array of one-hot labels (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
        The first 50000 samples are the official training set and the last 10000 are the official test set.
    """

    cache_dir = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "machine_learning_notes" / "cifar10"
    cache_file = cache_dir / "cifar10.npz"

    if cache_file.exists():
        with _np.load(cache_file) as data:
            images, labels = data["images"], data["labels"]
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive = cache_dir / "cifar-10-python.tar.gz"

        if not archive.exists():
            print(f"Downloading CIFAR-10 to {archive} ...")
            tmp = archive.with_suffix(".part")
            urllib.request.urlretrieve(_CIFAR10_URL, tmp)
            tmp.replace(archive)

        # Verify the download before unpickling anything from it
        md5 = hashlib.md5()
        with open(archive, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                md5.update(chunk)
        if md5.hexdigest() != _CIFAR10_MD5:
            archive.unlink()
            raise RuntimeError("CIFAR-10 download failed its checksum and was deleted. Please try again.")

        names = [f"cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)] + ["cifar-10-batches-py/test_batch"]
        image_batches, label_batches = [], []
        with tarfile.open(archive, "r:gz") as tar:
            for name in names:
                batch = pickle.load(tar.extractfile(name), encoding="bytes")
                # Each row is 3072 values: 1024 red, then 1024 green, then 1024 blue
                image_batches.append(_np.asarray(batch[b"data"], dtype=_np.uint8).reshape(-1, 3, 32, 32))
                label_batches.append(_np.asarray(batch[b"labels"], dtype=_np.int64))
        images = _np.concatenate(image_batches)
        labels = _np.concatenate(label_batches)

        tmp = cache_file.with_name("cifar10.tmp.npz")
        _np.savez_compressed(tmp, images=images, labels=labels)
        tmp.replace(cache_file)
        archive.unlink()  # the parsed cache replaces the raw archive

    X = images.astype(_np.float32) / 255.0
    return np.asarray(X), np.asarray(one_hot(labels, 10), dtype=np.float32)
