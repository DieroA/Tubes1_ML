import numpy as np

class Neuron:
    def __init__(self, weights: np.ndarray, bias: float):
        self.weights = weights
        self.bias = bias
        self.weights_gradients = np.zeros_like(weights)
        self.bias_gradients = 0.0
        self.value_matrice = np.zeros((0, 0))