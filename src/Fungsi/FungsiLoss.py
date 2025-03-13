import numpy as np
from typing import Dict, Callable

class FungsiLoss:
    def __init__(self, name: str):
        """
            mse -> Mean Squared Error
            bce -> Binary Cross-Entropy
            cce -> Categorical Cross-Entropy
        """

        loss_functions: Dict[str, Callable] = {
            "mse": self.mse,
            "bce": self.bce,
            "cce": self.cce
        }

        if name not in loss_functions:
            raise ValueError(f"{name} bukan merupakan nama fungsi loss yang valid.")
        
        self.func: Callable = loss_functions[name]
    
    @staticmethod
    def mse(y_true: np.array, y_pred: np.array) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def bce(y_true: np.array, y_pred: np.array) -> float:
        # Cek apakah y_true hanya mengandung 0 atau 1
        if not np.all(np.logical_or(y_true == 0, y_true == 1)):
            raise ValueError("y_true hanya dapat berbentuk 0 atau 1.")
        
        # Cek apakah y_pred mengandung 0 atau 1
        if np.any(y_pred == 0) or np.any(y_pred == 1):
            raise ValueError("y_pred tidak boleh mengadung 0 atau 1.")

        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * (np.log(1 - y_pred)))
    
    @staticmethod
    def cce(y_true: np.array, y_pred: np.array) -> float:
        pass