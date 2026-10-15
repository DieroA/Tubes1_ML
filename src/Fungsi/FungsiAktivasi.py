import numpy as np
from typing import Dict, Callable, Tuple

class FungsiAktivasi:
    def __init__(self, name: str):
        activation_functions: Dict[str, Tuple[Callable, Callable]] = {
            "linear": (self.linear, self.linear_derivative),
            "relu": (self.relu, self.relu_derivative),
            "sigmoid": (self.sigmoid, self.sigmoid_derivative),
            "tanh": (self.tanh, self.tanh_derivative),
            "softmax": (self.softmax, self.softmax_derivative)
        }

        if name not in activation_functions:
            raise ValueError(f"{name} bukan merupakan nama fungsi aktivasi yang valid.")

        self.name = name
        self.func, self.derivative = activation_functions[name]

    @staticmethod
    def linear(x: np.array) -> np.array:
        return x

    @staticmethod
    def linear_derivative(x: np.array) -> np.array:
        return np.ones_like(x)

    @staticmethod
    def relu(x: np.array) -> np.array:
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.array) -> np.array:
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    @staticmethod
    def sigmoid_derivative(a: np.array) -> np.array:
        # a = sigmoid(z), the layer's activated output
        return a * (1 - a)

    @staticmethod
    def tanh(x: np.array) -> np.array:
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(a: np.array) -> np.array:
        # a = tanh(z), the layer's activated output
        return 1 - a**2

    @staticmethod
    def softmax(x: np.array) -> np.array:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def softmax_derivative(x: np.array) -> np.array:
        s = x.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
    
RELU = FungsiAktivasi("relu")
LINEAR = FungsiAktivasi("linear")
SIGMOID = FungsiAktivasi("sigmoid")
TANH = FungsiAktivasi("tanh")
SOFTMAX = FungsiAktivasi("softmax")