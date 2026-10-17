import numpy as np

class Loss:
    def __init__(self) -> None:
        raise NotImplementedError("Loss class is abstract. Use an implemented loss function")

    def value(self, prediction, label):
        raise NotImplementedError("Loss class is abstract. Use an implemented loss function")

    def gradient(self, prediction, label):
        raise NotImplementedError("Loss class is abstract. Use an implemented loss function")


class ZeroOneLoss(Loss):
    def __init__(self) -> None:
        pass

    def value(self, prediction: int | float, label: int | float) -> int:
        """
        Returns the Zero One loss of a single prediction compared to the label. Gives 0 if they are equal and 1 otherwise.
        """
        
        return 1 - (prediction == label)

    def gradient(self, prediction, label) -> float:
        raise NotImplementedError("Zero One Loss is non-differentiable")


class MeanSquaredError(Loss):
    def __init__(self) -> None:
        pass

    def value(self, prediction: float, label: float) -> float:
        """
        Returns the square loss for a single prediction 
        """
        
        return np.square(label - prediction)

    def gradient(self, prediction: float, label: float) -> float:
        """
        Returns the gradient for a 
        """
        
        return 2 * (label - prediction)


class LogLoss(Loss):
    def __init__(self) -> None:
        pass

    def value(self, prediction: float, label: float) -> float:
        return -1 * (label * np.log(prediction) + (1 - label) * np.log(1 - prediction))