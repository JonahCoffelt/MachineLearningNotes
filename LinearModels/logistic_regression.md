# Logistic Regression
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


## Multiclass Logistic Regression
Binary linear regression can be extended to work for multiple classes rather than just two. To do this, we first need to restructure how we calculate the linear signal. Rather than having just one weight vector, we can make a weight vector for each class and horizontally stack them into a matrix like so:

$$
W = \begin{bmatrix}
| & | & & | \\
w_1 & w_2 & \cdots & w_K \\
| & | & & |
\end{bmatrix}
= \begin{bmatrix}
w_{10} & w_{20} & \cdots & w_{K0} \\
w_{11} & w_{21} & \cdots & w_{K1} \\
\vdots & \vdots & \ddots & \vdots \\
w_{1d} & w_{2d} & \cdots & w_{Kd}
\end{bmatrix}
\in \mathbb{R}^{(d+1) \times K}
$$

we can multiply this $W$ matrix with the input $x$ just like we have been doing. Intuitivly, this givens us a vector of the linear signals we had before:

$$
W^T x =
\begin{bmatrix}
w_1^T x \\
w_2^T x \\
\vdots \\
w_K^T x
\end{bmatrix}
$$

At this point in the binary case, we applied the sigmoid function. For the multiclass case, we apply a similar function known as *softmax*:

$$
\operatorname{softmax}(\mathbf{\vec{v}}) = \frac{1}{\sum_{k=1}^K e^{v_k}}
\begin{bmatrix}
e^{v_1} \\
e^{v_2} \\
\vdots \\
e^{v_K}
\end{bmatrix}
$$

This results in a vector whose elements are in the range $[0, 1]$ and sum up to one. Thus softmax can be interpreted as producing a vector of probabilities corresponding to each class. 