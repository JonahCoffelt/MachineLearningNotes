from Helpers import load_iris, normalized_split, load_mnist, train_valid_test_split, accuracy
from NeuralNetworks import *


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
            Linear(784, 512, 1e-5), BatchNorm1d(512), ReLU(), Dropout(0.3),
            Linear(512, 512, 1e-5), BatchNorm1d(512), ReLU(), Dropout(0.3),
            Linear(512, 10, 1e-5),
        ),
        CrossEntropy(),
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
    model.train(X_train, Y_train, learning_rate=0.1, batch_size=128, epochs=40)

    # Test
    print("Begin testing:")
    print("MNIST test accuracy:", accuracy(model, X_test, Y_test))


if __name__ == "__main__":
    # test_iris()
    test_mnist()