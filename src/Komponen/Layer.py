from typing import List
from Komponen.Neuron import Neuron
from Fungsi.FungsiAktivasi import FungsiAktivasi
import numpy as np

class Layer:
    def __init__(self, n_neurons: int, 
                 n_input: int, weight_init_method: str, activation_func: FungsiAktivasi, 
                 lower_bound: float = None, upper_bound: float = None, 
                 mean: float = None, variance: float = None, seed: int = None, values: List[int] = None):
        # Inisialisasi layer.
        self.n_neurons = n_neurons
        self.activation_func = activation_func

        # Inisialisasi Neuron
        self.neurons: List[Neuron] = [
            Neuron(n_input, weight_init_method, lower_bound, upper_bound, mean, variance, seed) 
            for _ in range(n_neurons)
        ]

        self.weight_matrice = np.array([neuron.weights for neuron in self.neurons])
        self.bias_matrice = np.array([[neuron.bias] for neuron in self.neurons])

        self.value_matrice: np.array = np.empty((0, 0))
        self.gradients_matrice: np.array = np.empty((0, 0))