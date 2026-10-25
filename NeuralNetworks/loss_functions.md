# Loss Functions
Loss functions are an important consideration in training a neural network, as they were for linear models. The loss function is used to propagate error back through the network, so it determines all of the gradients we get during training.

## Cross Entropy
For classification, it is common to use cross entropy, which is the same loss function used in multi-class (softmax) logistic regression. **Minimizing the cross entropy loss maximizes the log probability of the correct class**.

The network outputs raw scores (logits) $z$, which are converted to probabilities with softmax:

$$Q_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

Given a one-hot label $P$, predicted probabilities $Q$, and $K$ classes, the loss for a single sample is:

$$H(P, Q) = -\sum_{i=1}^K P_i \log Q_i$$

Note that $P$ is a one-hot vector, so exactly one term, the one for the correct class $c$, contributes to the loss and all others are multiplied by 0. The loss is therefore simply the negative log of the predicted probability of the correct class:

$$H(P, Q) = -\log Q_c$$

When softmax and cross entropy are combined, the gradient of the loss with respect to the logits is:

$$\frac{\partial H}{\partial z_i} = Q_i - P_i$$

## Use in Backpropagation
The loss is used to send update signals through the whole network. To do this, we initialize the sensitivity at the output to be the gradient of the loss with respect to the logits, $Q - P$, and each layer then passes it backward using the chain rule.