# CIFAR-10 Model Comparison

Every model was trained for 45 epochs with a batch size of 128. The linear model, neural network, and LeNet use a learning rate of 0.05. The ResNet uses learning rates of 0.1, 0.01, and 0.001 in three phases, with random flip and shift augmentation.

| Model | Test accuracy | Best training loss | Time per epoch |
|---|---|---|---|
| Linear | 29.15% | 3.0195 | 0.10s |
| Neural network | 49.31% | 0.4013 | 0.19s |
| LeNet | 63.72% | 0.6328 | 1.50s |
| ResNet-20 | 85.08% | 0.3506 | 28.43s |