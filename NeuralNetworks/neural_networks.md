# Neural Networks
Neural networks are the fundamental deep learning model, used to build many more complex models.


## Non-Linearity
Neural networks are not linear, allowing them to model more functions than linear models. This added capability this the primary motivation behind neural networks. Take XOR for example:

![Xor](./../Images/xor.png)

It is imposible to model XOR with a linear model because the two classes, true and false, are not linearly seperable. However, XOR can be decomposed into two linear functions:

![xor_decomp](./../Images/xor_decomp.png)

This means we can express XOR using multiple layers linear models. The perceptron can easily be used to model other logic functions, namely OR, NOT, and AND. 

![perceptron_or_and](./../Images/perceptron_or_and.png)

Conveniently, we can express XOR in terms of OR, NOT, and AND:

$$XOR(x_1, x_2) = x_1 \overline{x_2} + \overline{x_1} x_2$$

Thus, to implement XOR we can use multiple perceptron layers. The intermediate layers are known as **hidden layers**:

![mlp_xor](./../Images/mlp_xor.png)

This is what is historically know as a **multi-layer perceptron (MLP)**. However, the use of the non-differentiable $sign()$ function makes this model difficult. To fix this issue, modern neural networks replace $sign()$ with $tanh()$. This function is similar to the perceptron model and allows us to use gradient descent.


## Notation
For further discussion, it is crucial to define some notation. Typically, notation is given in terms of the layer $\ell$:

![Neural Network Notation](./../Images/nn_notation.png)

|             | Symbol          | Dimensions                                              |
|-------------|-----------------|---------------------------------------------------------|
| signals in  | $s^{(\ell)}$    | $d^{(\ell)}$ dimensional input vector                   |
| outputs     | $x^{(\ell)}$    | $d^{(\ell)} + 1$ dimensional output vector              |
| weights in  | $W^{(\ell)}$    | $(d^{(\ell-1)} + 1) \times d^{(\ell)}$ dimensional matrix |
| weights out | $W^{(\ell+1)}$  | $(d^{(\ell)} + 1) \times d^{(\ell+1)}$ dimensional matrix |

It is important to note that the bias is encoded in the weights in the same way we did for the linear models. That is, $x_0$, for all layers, is always set to $1$, and $w_0$ encodes the bias. This is why the weight matricies have dimensions of the form $d + 1$, the bias is the additional term. Importantly, we never send any signal into the bias, as we know it will be set to $1$.


## Number of Tunable Parameters
A useful excersize for building understanding the structure of a neural network is calculating the number of tunable parameters (weights and biases) in the network.  We can get the weight/bias matrix for each layer with:

$$W^{(\ell)} \in (d^{(\ell-1)}+1) \times d^{(\ell)}$$

Take as an example, a network with 10 inputs, 2 hidden layers of 20 neurons, and 5 outputs.

$$W^{(1)} \in (11 \times 20)$$
$$W^{(2)} \in (21 \times 20)$$
$$W^{(2)} \in (21 \times 5)$$

$$(11 \cdot 20) + (21 \cdot 20) + (21 \cdot 5) = \boxed{745}$$

## Forward pass
The forward pass takes in an input and passes it through the neural network layers sequentially, producing output logits. Each layer itself is quite simple, and is functionally very similar to a linear model. At a given layer $\ell$, we first do a linear weight operation on the input:

$$s^{(\ell)} = (W^{(\ell)})^T x^{(\ell-1)}$$

Then, we pass this linear signal through a non-linear function element-wise:

$$x^{(\ell)} = \begin{bmatrix} 1 \\ \theta(s^{(\ell)}) \end{bmatrix}$$

Since we are encoding the bias in the weight matrix, we then need to augment with a unit value to be used by the bias in the next layer. 

## Prediction
After forward pass, we have logits from the last layer $s^{(L)}$. These will then be passed through softmax to generate a probability vector:

$$Q = softmax(s^{(L)})$$

The class with the greatest probability is chosen as the predicted class.

## Backpropagation
Backpropagation is an algorithm to calculate the gradients of a neural network's weights at each layer. These gradients can be used to train the model. It works by passing gradient backward through the model, much like the forward pass worked by passing signal forward through the model. To do backproagation, we will first need to do a forward pass and keep track of the signals at each layer. 

The sensitivity of a layer $\delta^{(\ell)} = \frac{\partial E}{\partial s^{(\ell)}}$ represents how sensitive the loss function is to changes at that layer. When doing backpropagation, we are passing the sensitivity back through the layers like so:

$$\delta^{(\ell)} = \theta'(s^{(\ell)}) \otimes \left[ W^{(\ell+1)} \delta^{(\ell+1)} \right]_1^{d^{(\ell)}}$$

The slice $[\cdot]_1^{d^{(\ell)}}$ drops the bias component, since no signal flows back into the bias. These values can then be used to get the gradient with respect to the weights at each layer:

$$\frac{\partial E}{\partial W^{(\ell)}} = x^{(\ell-1)} (\delta^{(\ell)})^T$$

To start backpropagation, we need the sensitivity of the last layer. When using softmax with cross entropy loss, this is:

$$\delta^{(L)} = Q - y, \quad Q = \text{softmax}(s^{(L)})$$