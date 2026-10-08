# Linear Models
Linear models are a fundamental model in machine learning, suitable for many problems. They function by taking a weighted sum of features. 


## The Linear Signal
All linear models are built on the following equation:

$$s = w^Tx$$

Here, $s$ is the weighted sum of features, called the linear signal. $x$ is an input vector for all the features considered in the model. $w$ is a vector of weights corresponding to each input feature.

Once we have this value, we apply some function to the linear signal $s$ to obtain an output $y$. The function used depends on the type of model (covered in "Turning the Signal into an Output" below).

### Bias Term
In general, we will use a bias term in the model to allow it to shift the results independent of input. This is done by adding an extra feature and weight where $x_0 = 1$ and $w_0$ is the amount to shift (usually learned).


## Example
Let's say we want to calculate credit score from some given features:

| Feature | Value |
| --- | ---|
| age | 32 |
| salary | 40,000 |
| debt | 26,000 |
| years in job | 1 |
| years in home | 3 |

Then we can calculate the signal by taking the weighted sum of the features:

$$s = w_0 + w_1(\text{age}) + w_2(\text{salary}) + w_3(\text{debt}) + w_4(\text{years in job}) + w_5(\text{years in home})$$

Using the convention $x_0 = 1$, this is the same as:

$$s = \sum_{i=0}^{5} w_i x_i = w^T x$$

where

$$
x = \begin{bmatrix} 1 \\ 32 \\ 40{,}000 \\ 26{,}000 \\ 1 \\ 3 \end{bmatrix}, \quad
w = \begin{bmatrix} w_0 \\ w_1 \\ w_2 \\ w_3 \\ w_4 \\ w_5 \end{bmatrix}
$$

For instance, with made-up weights $w = [500,\ 0.5,\ 0.004,\ -0.003,\ 5,\ 2]^T$:

$$s = 500 + 0.5(32) + 0.004(40{,}000) - 0.003(26{,}000) + 5(1) + 2(3) = 609$$

Note that the weight on debt is negative, since more debt should lower the score, while the weight on salary is positive.


## Weight Intuition
Generally, we can think of the weight as determining the importance and benefit of a feature. If the magnitude of a weight is close to zero, it is not very important, whereas a large magnitude indicates great importance. If the sign of the weight is positive, then the feature is beneficial. If the sign of the weight is negative, the feature is detrimental.

- $|w_i|$ is large $\rightarrow$ $x_i$ is important
- $w_i > 0$ $\rightarrow$ $x_i$ is beneficial
- $w_i < 0$ $\rightarrow$ $x_i$ is detrimental

Note that weight magnitude only reflects importance when the features are on comparable scales. In the example above, the weight on salary ($0.004$) is much smaller than the weight on years in job ($5$), yet salary contributes $160$ to the sum while years in job contributes only $5$. This is why features are often normalized first.


## Turning the Signal into an Output
Linear models differ in the function applied to the signal $s$ to produce the output $y$.

### Classification
Linear models can be used for classification, which are yes/no problems. For example approving or denying a credit card is a classification problem. Typically, we will use a perceptron (a thresholded weighted sum) for these problems, where the output is $\pm 1$. The answer is simply the sign of the signal:

$$y = \text{sign}(s)$$

### Regression
Linear models can be used for regression, which are problems of quantity. For example, determining a credit line. Typically, we can use linear regression for these problems, where the output is a real number. We just take the raw signal:

$$y = s$$

### Logistic Regression
Logistic regression is a linear model used for problems of probability. For example, determining the probability of defaulting on a loan. Typically we use a logistic regression model for these problems, where the output is a probability on $[0, 1]$. We typically use a non-linear squashing function ($\theta(s)$) to ensure the output is a probability between $0$ and $1$:

$$y = \theta(s)$$

Thresholding this probability (e.g. at $0.5$) gives a yes/no decision, so logistic regression can also be used for classification.


## Training
So far we have treated the weights as given. In practice, the weights $w$ are learned from labeled examples by minimizing an error measure. All three models share the same linear signal, but they differ in the output function, the error measure, and the learning algorithm.

## Why Are These Models Considered Linear?
TODO (linear combination and hyperplane threshold is the answer)