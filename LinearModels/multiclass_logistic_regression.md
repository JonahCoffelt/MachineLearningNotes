# Multiclass Logistic Regression
Binary logistic regression can be extended to work for multiple classes rather than just two. This is useful for problems that are not easily divided into two classes, such as image recognition where we may have classes dog, cat, train, and so on.

## Prediction
To do multiclass prediction, we first need to restructure how we calculate the linear signal. Rather than having just one weight vector, we can make a weight vector for each class and horizontally stack them into a matrix like so:

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

Here $w_{kj}$ denotes the weight for class $k$ on feature $j$, with $j=0$ the bias term (so $x_0 = 1$). We can multiply this $W$ matrix with the input $x$ just like we have been doing. Intuitively, this gives us a vector of the linear signals we had before:

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
softmax(\mathbf{\vec{v}}) = \frac{1}{\sum_{k=1}^K e^{v_k}}
\begin{bmatrix}
e^{v_1} \\
e^{v_2} \\
\vdots \\
e^{v_K}
\end{bmatrix}
$$

This results in a vector whose elements are in the range $(0, 1)$ and sum up to one. Thus softmax can be interpreted as producing a vector of probabilities corresponding to each class. The prediction of the model is the class with the highest probability. For example:

$$
Q = 
\begin{bmatrix}
0.1 \\
0.2 \\
0.6 \\
0.1 \\
\end{bmatrix}
$$

In the above prediction, the third class is chosen because it has the highest probability. Note also that all entries sum to one, which is the expected output from softmax. 


## Training Data
Labels for multiclass regression are given as **one-hot vectors**, which are vectors with exactly one 1 entry, with the remaining entries all being 0. The entry that is 1 corresponds to the correct class. For example:

$$
P = 
\begin{bmatrix}
0 \\
1 \\
0 \\
\vdots \\
0
\end{bmatrix}
$$

In the above one-hot vector, the second entry is set to one, thus class two is correct for this sample. 

## Cross Entropy Loss
Cross entropy loss is used for multiclass logistic regression. **Minimizing the cross entropy loss maximizes the log probability of the correct class**. Given a sample one-hot label $P$, a prediction $Q$, and a number of classes K, the loss for a single sample is:

$$H(P, Q) =-\sum_{i=1}^K P_i\ log(Q_i)$$

Note that $P$ is a one-hot vector, so exactly one $Q_i$, which corresponds to the correct class, will contribute to the loss, all others will be multiplied by 0. Because of this, it is equivalent to say that the loss for a single sample is simply the negative log of the predicted probability for the correct class, $Q_c$:

$$H(P, Q) = -log(Q_c)$$

For the purposes of minimization, it is useful to write the loss for the whole dataset in terms of X (stack of inputs), Y (stack of label one-hot vectors), and W:

$$E(W)= -\sum_{n=1}^N \sum_{i=1}^K Y_{n, i}\ log(\frac{e^{W_i^T X_n}}{\sum_{k=1}^K e^{W_k^T X_n}})$$

This is what we will use when training. 


## Training
The gradient for cross entropy loss is given by:

$$\nabla_W E(W) = \sum_{n = 1}^N (Q_{n, i} - P_{n, i})X_n$$