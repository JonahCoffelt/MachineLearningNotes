import time
from pathlib import Path

from ConvolutionalNeuralNetworks import *
from NeuralNetworks import *
from Helpers import accuracy, load_mnist, train_valid_test_split, RandomFlipCrop
from Experiments import get_linear_model, get_neural_network, get_LeNet, get_ResNet

EPOCHS = 45  # total training epochs for every model
RESULTS_PATH = Path(__file__).parent / "mnist_comparison.md"


def train_and_evaluate(name, model, X_train, Y_train, X_test, Y_test, schedule, augment=None):
    """
    Trains a model through a schedule of (learning_rate, epochs) phases, then prints its test accuracy,
    best training loss, and training time per epoch.
    Returns:
        dict of the results, for the results table
    """
    print(f"Begin training {name}:")
    losses = []
    start = time.perf_counter()
    for learning_rate, epochs in schedule:
        losses += model.train(X_train, Y_train, learning_rate=learning_rate, batch_size=128, epochs=epochs, augment=augment)
    elapsed = time.perf_counter() - start

    test_accuracy = float(accuracy(model, X_test, Y_test))
    best_loss = min(losses)
    seconds_per_epoch = elapsed / len(losses)

    print(f"Begin testing {name}:")
    print(" - Test accuracy:", test_accuracy)
    print(f" - Best training loss: {best_loss:.4f}")
    print(f" - Training time per epoch: {seconds_per_epoch:.2f}s")

    label = name.removesuffix(" model")
    return {"model": label[0].upper() + label[1:], "accuracy": test_accuracy, "best_loss": best_loss,
            "seconds_per_epoch": seconds_per_epoch, "epochs": len(losses)}

def write_results(results, path=RESULTS_PATH, title="MNIST Model Comparison", augmentation="random-shift augmentation"):
    """
    Writes the results of the comparison to a markdown file with a table.
    """
    lines = [
        f"# {title}",
        "",
        f"Every model was trained for {results[0]['epochs']} epochs with a batch size of 128. "
        "The linear model, neural network, and LeNet use a learning rate of 0.05. "
        f"The ResNet uses learning rates of 0.1, 0.01, and 0.001 in three phases, with {augmentation}.",
        "",
        "| Model | Test accuracy | Best training loss | Time per epoch |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['model']} | {r['accuracy'] * 100:.2f}% | {r['best_loss']:.4f} | {r['seconds_per_epoch']:.2f}s |")
    Path(path).write_text("\n".join(lines))
    print(f"\nSaved results to {path}")

def compare_models():

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

    in_features = 28 * 28
    in_channels = 1
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
    model = get_LeNet(in_channels, n_classes, conv1_padding=2)
    results.append(train_and_evaluate("LeNet model", model, X_train, Y_train, X_test, Y_test, schedule))

    # ResNet
    print("\n\nLoading ResNet-20 model...")
    model = get_ResNet(n_classes, in_channels)
    aug = RandomFlipCrop(flip=False, pad=2)  # no flips for digits
    results.append(train_and_evaluate("ResNet-20 model", model, X_train, Y_train, X_test, Y_test, resnet_schedule, augment=aug))

    write_results(results)

if __name__ == "__main__":
    compare_models()