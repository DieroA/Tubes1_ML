import numpy as np
from typing import Dict, Callable

class FungsiAktivasi:
    def __init__(self, name: str):
        activation_functions: Dict[str, Callable] = {
            "linear": self.linear,
            "relu": self.relu,
            "sigmoid": self.sigmoid,
            "tanh": self.tanh,
            "softmax": self.softmax
        }

        if name not in activation_functions:
            raise ValueError(f"{name} bukan merupakan nama fungsi aktivasi yang valid.")

        self.func: Callable = activation_functions[name]


    @staticmethod
    def linear(x: np.array) -> np.array:
        return np.array(x, dtype=np.float64)
    
    @staticmethod
    def relu(x: np.array) -> np.array:
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x: np.array) -> np.array:
        return np.tanh(x)
    
    @staticmethod
    def softmax(x: np.array) -> np.array:
        exp_x: np.array = np.exp(x)
        return exp_x / np.sum(exp_x)