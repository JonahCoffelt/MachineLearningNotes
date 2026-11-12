from pathlib import Path

from ConvolutionalNeuralNetworks import *
from NeuralNetworks import *
from Helpers import load_cifar10, RandomFlipCrop
from Experiments import get_linear_model, get_neural_network, get_LeNet, get_ResNet
from Experiments.mnist_comparison import EPOCHS, train_and_evaluate, write_results

RESULTS_PATH = Path(__file__).parent / "cifar10_comparison.md"
N_TRAIN = 50000

def compare_models():

    print("Loading data...")
    X, Y = load_cifar10()
    X_train, Y_train = X[:N_TRAIN], Y[:N_TRAIN]
    X_test, Y_test = X[N_TRAIN:], Y[N_TRAIN:]

    # Per-channel normalization using training statistics only, shape (1, 3, 1, 1)
    mu = X_train.mean(axis=(0, 2, 3), keepdims=True)
    sigma = X_train.std(axis=(0, 2, 3), keepdims=True)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    in_features = 3 * 32 * 32
    in_channels = 3
    n_classes = 10

    # Every model gets the same number of epochs
    schedule = [(0.05, EPOCHS)]
    # The ResNet decays its learning rate in three phases (2/3, 2/9, and the remaining 1/9 of the epochs)
    phase1, phase2 = round(EPOCHS * 2 / 3), round(EPOCHS * 2 / 9)
    resnet_schedule = [(0.1, phase1), (0.01, phase2), (0.001, EPOCHS - phase1 - phase2)]

    results = []

    # Linear Model
    print("Loading linear model...")
    model = get_linear_model(in_features, n_classes)
    results.append(train_and_evaluate("linear model", model, X_train, Y_train, X_test, Y_test, schedule))

    # NN
    print("\n\nLoading neural network model...")
    model = get_neural_network(in_features, n_classes)
    results.append(train_and_evaluate("neural network model", model, X_train, Y_train, X_test, Y_test, schedule))

    # LeNet
    print("\n\nLoading LeNet model...")
    model = get_LeNet(in_channels, n_classes)
    results.append(train_and_evaluate("LeNet model", model, X_train, Y_train, X_test, Y_test, schedule))

    # ResNet
    print("\n\nLoading ResNet-20 model...")
    model = get_ResNet(n_classes, in_channels)
    aug = RandomFlipCrop()  # random horizontal flips and 4-pixel shifts
    results.append(train_and_evaluate("ResNet-20 model", model, X_train, Y_train, X_test, Y_test, resnet_schedule, augment=aug))

    write_results(results, RESULTS_PATH, title="CIFAR-10 Model Comparison", augmentation="random flip and shift augmentation")

if __name__ == "__main__":
    compare_models()
