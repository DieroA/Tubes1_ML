import numpy as np
from typing import Dict, Callable, Tuple

class FungsiLoss:
    def __init__(self, name: str):
        """
            mse -> Mean Squared Error
            bce -> Binary Cross-Entropy
            cce -> Categorical Cross-Entropy
        """
        loss_functions: Dict[str, Tuple[Callable, Callable]] = {
            "mse": (self.mse, self.mse_derivative),
            "bce": (self.bce, self.bce_derivative),
            "cce": (self.cce, self.cce_derivative)
        }

        if name not in loss_functions:
            raise ValueError(f"{name} bukan merupakan nama fungsi loss yang valid.")
        
        self.name: str = name
        self.func, self.derivative_func = loss_functions[name]
    
    @staticmethod
    def mse(y_true: np.array, y_pred: np.array) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_derivative(y_true: np.array, y_pred: np.array) -> np.array:
        return 2 * (y_pred - y_true) / y_true.size
    
    @staticmethod
    def bce(y_true: np.array, y_pred: np.array) -> float:
        # Cek apakah y_true hanya mengandung 0 atau 1
        if not np.all(np.logical_or(y_true == 0, y_true == 1)):
            raise ValueError("y_true hanya dapat berbentuk 0 atau 1.")
        
        # Cek apakah y_pred mengandung 0 atau 1
        if np.any(y_pred == 0) or np.any(y_pred == 1):
            raise ValueError("y_pred tidak boleh mengadung 0 atau 1.")

        # Klip prediksi untuk menghindari kasus log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def bce_derivative(y_true: np.array, y_pred: np.array) -> np.array:
        # Klip Prediksi untuk menghindari kasus pembagian oleh nol
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.size
    
    @staticmethod
    def cce(y_true: np.array, y_pred: np.array) -> float:
        # Klip Prediksi untuk menghindari kasus log(0)
        y_pred = np.clip(y_pred, 1e-15, 1)
        # Menormalisasi prediksi agar jumlahnya 1 (Seperti Softmax)
        y_pred = y_pred / np.sum(y_pred, axis=-1, keepdims=True)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))
    
    @staticmethod
    def cce_derivative(y_true: np.array, y_pred: np.array) -> np.array:
        """
        General CCE derivative without softmax assumption.
        The derivative is -y_true/(y_pred * n_samples)
        """
        # Klip dan normalisasi prediksi
        y_pred = np.clip(y_pred, 1e-15, 1)
        y_pred = y_pred / np.sum(y_pred, axis=-1, keepdims=True)
        return -y_true / (y_pred * y_true.shape[0])
    
    def derivative(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
        Computes the derivative of the loss function
        """
        return self.derivative_func(y_true, y_pred)