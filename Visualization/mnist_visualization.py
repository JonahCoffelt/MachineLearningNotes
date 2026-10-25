from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from Helpers import load_mnist, train_valid_test_split, accuracy
from NeuralNetworks import NeuralNetwork, Linear, BatchNorm1d, ReLU, Dropout, CrossEntropy
from NeuralNetworks.loss import softmax


def save_figure(fig, name: str) -> None:
    """
    Save a figure as a png in the Visualization folder.
    """
    path = Path(__file__).parent / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved to", path)


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """
    Softmax probabilities of the model's outputs.
    """
    model.predict(X[:1])  # puts the layers in evaluation mode
    return softmax(model.forward(X))


def visualize_mnist_linear_outputs(model) -> None:
    """
    Visualize the weights going into each output neuron of a linear MNIST model.
    """
    weights = model.layers[0].W
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5), layout="constrained")
    for digit, ax in enumerate(axes.ravel()):
        coef = weights[:, digit]
        ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=coef.min(), vmax=coef.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(str(digit))
    fig.suptitle("Weights into each output neuron of a linear MNIST model")

    save_figure(fig, "mnist_linear_outputs.png")


def visualize_confusion_matrix(model, X: np.ndarray, Y: np.ndarray) -> None:
    """
    Visualize how often each true digit is predicted as each digit.
    """
    true = np.argmax(Y, axis=1)
    pred = model.predict(X)
    counts = np.zeros((10, 10), dtype=int)
    np.add.at(counts, (true, pred), 1)

    # color by fraction of each true digit so rows are comparable
    fractions = counts / counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    image = ax.imshow(fractions, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, counts[i, j], ha="center", va="center", fontsize=8,
                    color="white" if fractions[i, j] > 0.5 else "black")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    fig.colorbar(image, ax=ax, label="Fraction of true digit")
    ax.set_title("MNIST confusion matrix (test set)")

    save_figure(fig, "mnist_confusion_matrix.png")


def visualize_softmax_outputs(model, X: np.ndarray, Y: np.ndarray, n: int = 5) -> None:
    """
    Visualize the softmax probabilities for the n test images the model is least confident about.
    """
    probs = predict_proba(model, X)
    true = np.argmax(Y, axis=1)
    least_confident = np.argsort(probs.max(axis=1))[:n]

    fig, axes = plt.subplots(n, 2, figsize=(7, 1.8 * n), layout="constrained", width_ratios=(1, 3))
    for row, i in enumerate(least_confident):
        pred = np.argmax(probs[i])

        image_ax, bar_ax = axes[row]
        image_ax.matshow(X[i].reshape(28, 28), cmap=plt.cm.gray_r)
        image_ax.set_xticks(())
        image_ax.set_yticks(())
        image_ax.set_title(f"True {true[i]}, predicted {pred}", fontsize=9)

        colors = ["tab:gray"] * 10
        colors[pred] = "tab:red"
        colors[true[i]] = "tab:green"
        bar_ax.bar(range(10), probs[i], color=colors)
        bar_ax.set_xticks(range(10))
        bar_ax.set_ylim(0, 1)
        bar_ax.set_ylabel("Probability")
    fig.suptitle("Least confident test predictions (green = true, red = wrong prediction)")

    save_figure(fig, "mnist_softmax_outputs.png")


def visualize_training_curves(history: dict[str, list[float]]) -> None:
    """
    Visualize the loss and accuracy on the training and validation sets after each epoch.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    loss_ax.plot(epochs, history["train_loss"], label="Train")
    loss_ax.plot(epochs, history["valid_loss"], label="Validation")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Cross-entropy loss")
    loss_ax.legend()

    accuracy_ax.plot(epochs, history["train_accuracy"], label="Train")
    accuracy_ax.plot(epochs, history["valid_accuracy"], label="Validation")
    accuracy_ax.set_xlabel("Epoch")
    accuracy_ax.set_ylabel("Accuracy")
    accuracy_ax.legend()
    for ax in (loss_ax, accuracy_ax):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle("MNIST training curves")

    save_figure(fig, "mnist_training_curves.png")


def cross_entropy(model, X: np.ndarray, Y: np.ndarray) -> float:
    probs = predict_proba(model, X)
    return -np.sum(Y * np.log(np.clip(probs, 1e-12, 1.0))) / len(Y)


def main() -> None:
    X, Y = load_mnist()
    (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = train_valid_test_split(X, Y)

    # Linear model: pixels feed straight into the output neurons, so its weights can be shown as images
    linear_model = NeuralNetwork(
        (
            Linear(784, 10, 1e-4),
        ),
        CrossEntropy(),
    )
    linear_model.train(X_train, Y_train, learning_rate=0.1, batch_size=128, epochs=100)
    print("MNIST linear test accuracy:", accuracy(linear_model, X_test, Y_test))
    visualize_mnist_linear_outputs(linear_model)

    # Deep model
    deep_model = NeuralNetwork(
        (
            Linear(784, 512, 1e-5), BatchNorm1d(512), ReLU(), Dropout(0.3),
            Linear(512, 512, 1e-5), BatchNorm1d(512), ReLU(), Dropout(0.3),
            Linear(512, 10, 1e-5),
        ),
        CrossEntropy(),
    )

    mu, sigma = X_train.mean(), X_train.std()
    X_train = (X_train - mu) / sigma
    X_valid = (X_valid - mu) / sigma
    X_test = (X_test - mu) / sigma

    # train one epoch at a time so the loss and accuracy can be recorded after each epoch
    history = {"train_loss": [], "valid_loss": [], "train_accuracy": [], "valid_accuracy": []}
    for epoch in range(20):
        deep_model.train(X_train, Y_train, learning_rate=0.1, batch_size=128, epochs=1)
        history["train_loss"].append(cross_entropy(deep_model, X_train, Y_train))
        history["valid_loss"].append(cross_entropy(deep_model, X_valid, Y_valid))
        history["train_accuracy"].append(accuracy(deep_model, X_train, Y_train))
        history["valid_accuracy"].append(accuracy(deep_model, X_valid, Y_valid))
    print("MNIST deep test accuracy:", accuracy(deep_model, X_test, Y_test))

    visualize_confusion_matrix(deep_model, X_test, Y_test)
    visualize_softmax_outputs(deep_model, X_test, Y_test)
    visualize_training_curves(history)


if __name__ == "__main__":
    main()
