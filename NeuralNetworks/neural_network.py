import numpy as np
from Helpers import load_iris

def softmax(v: np.ndarray) -> np.ndarray:
    v = v - np.max(v, axis=-1, keepdims=True)  # Normalize for saftey (this is pretty standard)
    exp_v = np.exp(v)
    return exp_v / np.sum(exp_v, axis=-1, keepdims=True)

def tanh_prime(s: np.ndarray) -> np.ndarray:
    return 1 - np.power(np.tanh(s), 2)

class NeuralNetwork:
    def __init__(self, dims: tuple) -> None:
        """
        
        """

        self.rng = np.random.default_rng(42)
        self.dims = dims
        self.weights = [self.rng.random((dims[i] + 1, dims[i + 1])) * 2 - 1 for i in range(len(dims) - 1)]
        self.n_layers = len(self.weights)

    def loss(self, q: np.ndarray, p: np.ndarray) -> float:
        q = np.clip(q, 1e-12, 1.0)
        return -np.sum(p * np.log(q))

    def gradient(self, y: np.ndarray, x_aug: np.ndarray) -> list[np.ndarray]:
        """
        
        """
        
        signals = []
        outputs = [x_aug]
        sensitivities = [np.array([0]) for i in range(self.n_layers)]
        gradients = [np.array([0]) for i in range(self.n_layers)]

        for layer in range(self.n_layers):
            W = self.weights[layer]
            signal = W.T @ outputs[-1]
            signals.append(signal)
            
            if layer != self.n_layers - 1:
                output = np.insert(np.tanh(signal), 0, 1)
                outputs.append(output)
            else:
                outputs.append(softmax(signal))

        sensitivities[-1] = outputs[-1] - y

        for layer in range(self.n_layers - 2, -1, -1):
            W_next = self.weights[layer + 1]
            nabla_next = sensitivities[layer + 1]
            s = signals[layer]
            sensitivities[layer] = tanh_prime(s) * (W_next @ nabla_next)[1:]

        for layer in range(self.n_layers):
            gradients[layer] = np.outer(outputs[layer], sensitivities[layer])

        return gradients

    def train(self, X: np.ndarray, Y: np.ndarray, learning_rate: float, batch_size: int, epochs: int) -> None:
        """
        Train the model on the given data. Uses 0/1 labels. 
        Args:
            X: numpy array (n_samples, n_features) of dataset features 
            Y: numpy array (n_samples, ) of dataset labels
            learning_rate: float value for the rate of gradient descent
            batch_size: number of samples per batch
            epochs: number of times to loop through entire dataset 
        """

        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        for epoch in range(epochs):
            loss_accumulation = 0.0

            # Shuffle the samples each epoch
            order = self.rng.permutation(n_samples)

            for batch in range(n_batches):
                # Accumulate gradient for the batch
                gradient_accumulation = [np.zeros_like(W) for W in self.weights]

                for sample in range(batch_size):
                    sample_index = order[batch * batch_size + sample]
                    x = X[sample_index]
                    y = Y[sample_index]

                    # Get the probability and add loss
                    probabilities = self.forward(x)
                    loss_accumulation += self.loss(probabilities, y)

                    # Get the gradient on each layer
                    gradients = self.gradient(y, x)

                    # Add the gradient to the accumulation
                    for layer in range(self.n_layers):
                        gradient_accumulation[layer] += gradients[layer]

                # Add the gradients to the weight layers
                for layer in range(self.n_layers):
                    self.weights[layer] -= (gradient_accumulation[layer] / batch_size) * learning_rate
           
            # Print the current in-sample error
            print(f"Epoch {epoch + 1} - Loss {loss_accumulation / (n_batches * batch_size)}")

    def forward(self, x_aug: np.ndarray) -> np.ndarray:
        """
        
        """
        
        z = x_aug
        for layer in range(self.n_layers):
            W = self.weights[layer]
            signal = W.T @ z

            if layer == self.n_layers - 1:
                return softmax(signal) 

            z = np.insert(np.tanh(signal), 0, 1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :]
        x_aug = np.hstack([np.ones((x.shape[0], 1)), x])
        return np.array([np.argmax(self.forward(row)) for row in x_aug])

        
X, Y = load_iris()
model = NeuralNetwork((X.shape[1], 8, Y.shape[1]))
model.train(X, Y, learning_rate=0.05, batch_size=32, epochs=500)
preds = model.predict(X)
print("Iris accuracy:", (preds == np.argmax(Y, axis=1)).mean())