# Perceptron 
Perceptrons are linear models that are used for classification problems. 


## Prediction
Perceptrons make predictions by taking the sign of the [linear signal](./linear_models.md), which will produce either a -1 or a 1:

$$\hat{y} = sign(w^T x)$$

Note that $x$ will have the bias set with $x_0 = 1$.


## Example
**TODO**


## Loss
If the prediction was correct, there is no loss, otherwise, the loss is simply the magnitude of the linear signal. This can be represented as such:

$$E(w) = max(0, -y(w^T x))$$


## Training
Perceptrons are trained with gradient decent. Training a perceptron updates the weights of a model only when it makes an incorrect prediction. If the model only makes correct predictions, it is never updated. If the classes are linearly separable, then the model will eventually converge into a solution weight, otherwise, will never converge. 

Given an incorrect prediction $\hat{y}$ and a ground truth prediction $y$, the update signal is written as such:

$$w_{t+1} = w_{t}\ +\ \eta\  y\  x$$

Where $\eta$ is the learning rate. Intuitively, a missed positive point pulls the weights toward $x$, and a missed negative point pushes them away from $x$, moving the decision boundary toward classifying that point correctly.