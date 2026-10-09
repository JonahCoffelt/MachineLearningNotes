# Multiclass Logistic Regression
Binary linear regression can be extended to work for multiple classes rather than just two. 

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

This results in a vector whose elements are in the range $[0, 1]$ and sum up to one. Thus softmax can be interpreted as producing a vector of probabilities corresponding to each class. The prediction of the model is the class with the highest probability. 


## Cross Entropy Loss
Cross entropy loss is used for multiclass logistic regression. 