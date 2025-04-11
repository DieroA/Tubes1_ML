from typing import List
from Model.Komponen.Neuron import Neuron
from Fungsi.FungsiAktivasi import FungsiAktivasi
import numpy as np

class Layer:
    def __init__(self, n_neurons: int, 
                 n_input: int, weight_init_method: str, activation_func: FungsiAktivasi, 
                 lower_bound: float = None, upper_bound: float = None, 
                 mean: float = None, variance: float = None, seed: int = None):
        self.n_neurons = n_neurons
        self.activation_func = activation_func

        fan_in = n_input
        fan_out = n_neurons

        if seed is not None:
            np.random.seed(seed)

        def generate_weights():
            if weight_init_method == "zero":
                return np.zeros(fan_in), 0.0
            elif weight_init_method == "uniform":
                if lower_bound is None or upper_bound is None:
                    raise ValueError("Uniform init needs lower_bound and upper_bound.")
                return np.random.uniform(lower_bound, upper_bound, fan_in), np.random.uniform(lower_bound, upper_bound)
            elif weight_init_method == "normal":
                if mean is None or variance is None:
                    raise ValueError("Normal init needs mean and variance.")
                return np.random.normal(mean, np.sqrt(variance), fan_in), np.random.normal(mean, np.sqrt(variance))
            elif weight_init_method == "xavier":
                if fan_in != 0:
                    limit = np.sqrt(6 / (fan_in + fan_out))
                    return np.random.uniform(-limit, limit, fan_in), np.random.uniform(-limit, limit)
                else:
                    return np.zeros(fan_in), 0.0
            elif weight_init_method == "he":
                if fan_in != 0:
                    std = np.sqrt(2 / fan_in)
                    return np.random.normal(0, std, fan_in), np.random.normal(0, std)
                else:
                    return np.zeros(fan_in), 0.0
            else:
                raise ValueError(f"Invalid weight initialization method: {weight_init_method}")

        self.neurons: List[Neuron] = []
        for _ in range(n_neurons):
            w, b = generate_weights()
            self.neurons.append(Neuron(w, b))

        self.weight_matrice = np.array([neuron.weights for neuron in self.neurons])
        self.bias_matrice = np.array([[neuron.bias] for neuron in self.neurons])
        self.weight_gradients = np.array([neuron.weights_gradients for neuron in self.neurons])
        self.bias_gradients = np.array([[neuron.bias_gradients] for neuron in self.neurons])
        self.value_matrice: np.array = np.empty((0, 0))
