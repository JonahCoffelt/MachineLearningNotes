# Gradient Descent
Gradient descent is a method for minimizing the loss of a model. It is generally applied to models for which no analytical solution is possible. It functions by taking incremental steps against the gradient, which, over time, reaches a local minimum. 

## Batches
Gradient descent may be performed on a whole dataset, only one sample, or partitions of the dataset. These are known as batch, stochastic, and mini-batch gradient descent respectively. In practice mini-batch gradient descent is used, as datasets are generally too large for batch gradient descent and the updates from stochastic gradient descent are very noisy. 


| Method        | Updates Per Iteration | Convergence Speed | Stability       | Computation Cost |
|---------------|-----------------------|-------------------|-----------------|------------------|
| Batch GD      | After full dataset    | Slow              | Stable          | High             |
| Stochastic GD | After each sample     | Fast              | Noisy (unstable)| Low              |
| Mini-Batch GD | After each mini-batch | Moderate          | Balanced        | Moderate         |

## Epoch
An epoch is one full pass over the dataset. 

## Limiting Behavior
Note that batch gradient descent and stochastic gradient descent are both special cases of mini-batch gradient descent. Batch gradient descent is mini-batch gradient descent where the batch size is equal to the total number of samples. Stochastic gradient descent is mini-batch gradient descent where the batch size is equal to 1.