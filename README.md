# Machine Learning Notes
This is a collection of my personal machine learning notes and code implementations. Largely based on the following resources:

- Lectures and slides by Dr. Ji
- Learning From Data textbook. 


# Contents

## Linear Models
- [Overview of Linear Models](LinearModels/linear_models.md)
- [Linear Regression](LinearModels/linear_regression.md)
    - [Numpy Implementation](LinearModels/linear_regression.py)
- [Perceptron](LinearModels/perceptron.md)
    - [Numpy Implementation](LinearModels/perceptron.py)
- [Binary Logistic Regression](LinearModels/binary_logistic_regression.md)
    - [Numpy Implementation](LinearModels/binary_logistic_regression.py)
- [Multiclass Logistic Regression](LinearModels/multiclass_logistic_regression.md)
    - [Numpy Implementation](LinearModels/multiclass_logistic_regression.py)

## General Techniques
- [Gradient Descent](GeneralTechniques/gradient_descent.md)
- [Overfitting](GeneralTechniques/overfitting.md)
- [Regularization](GeneralTechniques/regularization.md)

## Neural Networks
- [Neural Networks](NeuralNetworks/neural_networks.md)

# Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run any file directly, for example: 
```bash
python LinearModels/linear_regression.py
```
