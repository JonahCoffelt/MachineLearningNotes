from Helpers import load_iris, normalized_split, load_mnist, train_valid_test_split, accuracy, np
from NeuralNetworks import *
from ConvolutionalNeuralNetworks import *

def test_mnist_cnn():
    print("Creating an mnist CNN model...")
    model = NeuralNetwork(
        (
            Conv2d(1, 16, kernel_size=3, stride=1, padding=1), ReLU(),
            MaxPool2d(2),
            Conv2d(16, 32, kernel_size=3, stride=1, padding=1), ReLU(),
            MaxPool2d(2),
            Flatten(),
            Linear(1568, 128, 1e-5), ReLU(), Dropout(0.3),
            Linear(128, 10, 1e-5),
        ),
        CrossEntropy(),
    )

    print("Loading data...")
    X, Y = load_mnist()
    (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = train_valid_test_split(X, Y)

    mu, sigma = X_train.mean(), X_train.std()
    X_train = (X_train - mu) / sigma
    X_valid = (X_valid - mu) / sigma
    X_test = (X_test - mu) / sigma

    X_train = X_train.reshape(-1, 1, 28, 28)
    X_valid = X_valid.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    print("Begin training:")
    model.train(X_train, Y_train, learning_rate=0.05, batch_size=128, epochs=15)

    print("Begin testing:")
    print("MNIST CNN test accuracy:", accuracy(model, X_test, Y_test))