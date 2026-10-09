# Linear Regression
Linear regression is a class of linear models that are used to solve problems of quantity. These are the simplest model because the output of the model is just the [linear signal](./linear_models.md). 

## Prediction
Prediction is simple, given a weight vector $w$ and an input vector of features $x$, the output of the model $y$ is simply the linear signal $s$:

$$\hat{y} = s = w^Tx$$

Note that $x$ will have the bias set with $x_0 = 1$.

## Example
**TODO**

## Training Data
Given $N$ training examples, we stack the input vectors as the rows of a matrix $X$ and the correct outputs into a vector $y$:

$$
X = \begin{bmatrix}
\text{---} \; x_1^T \; \text{---} \\
\text{---} \; x_2^T \; \text{---} \\
\vdots \\
\text{---} \; x_N^T \; \text{---}
\end{bmatrix}
= \begin{bmatrix}
x_{10} & x_{11} & \cdots & x_{1d} \\
x_{20} & x_{21} & \cdots & x_{2d} \\
\vdots & \vdots & \ddots & \vdots \\
x_{N0} & x_{N1} & \cdots & x_{Nd}
\end{bmatrix},
\qquad
y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{bmatrix}
$$

Row $n$ of $X$ is the input vector $x_n^T$ (with $x_{n0} = 1$ for the bias), and entry $n$ of $y$ is the correct output for that input. So $X$ is $N \times (d+1)$ and $y$ is $N \times 1$.


## Loss Function
Loss is a measure of error in a machine learning model. In linear regression, we use a mean square loss function, given as:

$$E(w) = \frac{1}{N} \sum_{n=1}^N (\hat{y}_n - y_n)^2$$

This loss function simply takes the square of the difference between the predicted value $\hat{y}_n$ and the actual value $y_n$. We sum all given training samples and divide by $N$, the number of samples, to get the mean loss. 

Note that the loss is a function of the weights $w$, so if we want to change the loss, we need to change the weights. 

## Training
### Closed Form
Unlike most machine learning models, there is a closed for solution for training a linear regression model. To do this, we need a dataset $X$ and labels $y$. The optimal weights is given by:

$$w = (X^T X)^{-1} X^T y$$

Note that this only works when $X^T X$ is invertable. 

#### Derivation
**TODO**

### Gradient Decent
**TODO**