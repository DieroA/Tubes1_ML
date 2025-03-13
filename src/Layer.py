from Neuron import Neuron
from typing import List
from FungsiAktivasi import FungsiAktivasi

class Layer:
    def __init__(self, n_neurons: int, 
                 n_input: int, weight_init_method: str, activation_func: FungsiAktivasi, 
                 lower_bound: float = None, upper_bound: float = None, 
                 mean: float = None, variance: float = None, seed: int = None):
        # Inisialisasi layer.

        self.neurons: List[Neuron] = [Neuron(n_input, weight_init_method, lower_bound, upper_bound, mean, variance, seed) for _ in range(n_neurons)]
        self.activation_func = activation_func