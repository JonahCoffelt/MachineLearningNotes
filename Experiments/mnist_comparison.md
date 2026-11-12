# MNIST Model Comparison

Every model was trained for 45 epochs with a batch size of 128. The linear model, neural network, and LeNet use a learning rate of 0.05. The ResNet uses learning rates of 0.1, 0.01, and 0.001 in three phases, with random-shift augmentation.

| Model | Test accuracy | Best training loss | Time per epoch |
|---|---|---|---|
| Linear | 91.63% | 0.2408 | 0.11s |
| Neural network | 97.66% | 0.0018 | 0.20s |
| LeNet | 99.03% | 0.0074 | 1.51s |
| ResNet-20 | 99.54% | 0.0049 | 22.77s |