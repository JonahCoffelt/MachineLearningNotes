# Binary Logistic Regression
Logistic regression models are linear models used for probability and classification problems.

## Prediction
Logistic Regression models make predictions by applying some non-linear activation function $\theta (s)$ to the [linear signal](./linear_models.md). For the binary case, this function will be sigmoid. 

$$\hat{y} = \theta(w^T x)$$
$$\theta(z) = \frac{1}{1 + e^{-z}}$$

Because this output is between $0$ and $1$, it can be interpreted as a probability. Additionally, by using some threshold (e.g. $\tau = 0.5$), we can use the probability prediction to do classification. Note that $x$ will have the bias set with $x_0 = 1$.


## Example
**TODO**


## Loss Function
For binary logistic regression, we use log loss (also known as logistic regression loss):

$$E(w) = \frac{1}{N} \sum_{n=1}^N ln(1 + e^{-y_n w^T x})$$

This is preferable to the square loss because, as we will see, it is easy to minimize. Furthermore, it can be shown that minimizing this loss is equivalent to maximizing the likelihood, which is exactly our goal. 


## Training
Logistic regression models are trained using gradient descent. For this section, we assume we are given binary labels $y$, where y is either $-1$ or $1$. Given the loss function above for a binary model, the gradient is:

$$\nabla_w E(w) = -\frac{1}{N} \sum_{n=1}^N \frac{y_nx_n}{1 + e^{y_n w^T x_n}}$$

This  formula can be used to update the weights as such:

$$w_{t+1} = w_{t} - \eta \nabla_w E(w)$$

### 0/1 labels
If our labels are given as $0$ or $1$ rather than $-1$ or $1$, the loss is given by:

$$E(w) = -\frac{1}{N} \sum_{n=1}^N [y_n ln(\hat{y}_n) + (1 - y_n)ln(1 - \hat{y}_n)]$$

Doing this gives us the gradient in a remarkably simple form:

$$\nabla_w E(w) = \frac{1}{N} \sum_{n=1}^N(\hat{y}_n - y_n)x_n$$

This is often what is used in practice, but is not as easy to compare to the perceptron model.
