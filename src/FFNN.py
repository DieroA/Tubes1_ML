from Fungsi.FungsiAktivasi import FungsiAktivasi
from Fungsi.FungsiLoss import FungsiLoss
from Komponen.Layer import Layer
from typing import List

import numpy as np

class FFNN:
    def __init__(self, input: List[float], output: List[float], input_size: int, hidden_size: int, output_size: int, n_hidden: int,
                 activation_func: List[FungsiAktivasi],  loss_func: FungsiLoss = FungsiLoss("mse"), weight_init_method: str = "zero", 
                 lower_bound: float = None, upper_bound: float = None, mean: float = None, 
                 variance: float = None, seed: int = 42):
        # Inisialisasi model.
        # input_size: Jumlah input
        # hidden_size: Jumlah neuron per layer
        # output_size: Jumlah output / kelas
        # n_hidden: Jumlah hidden layer
        
        self.layers: List[Layer] = []
        self.activation_func: List[FungsiAktivasi] = activation_func

        # Input layer
        self.layers.append(Layer(input_size, 0, weight_init_method, activation_func[0], lower_bound, upper_bound, mean, variance, seed, input))
        
        # Hidden layer(s)
        prev_size: int = input_size
        for i in range(n_hidden):
            self.layers.append(Layer(hidden_size, prev_size, weight_init_method, activation_func[i + 1], lower_bound, upper_bound, mean, variance, seed))
            prev_size = hidden_size

        # Output layer
        self.layers.append(Layer(output_size, hidden_size, weight_init_method, activation_func[-1], lower_bound, upper_bound, mean, variance, seed, output))
        
        # Loss Function
        self.lost_function:FungsiLoss = loss_func

        # Target Value
        self.target : List[float] = output

        self.generate_matrices()

    def generate_matrices(self):
        # Menghasilkan matriks bobot & bias untuk setiap layer
        for idx, layer in enumerate(self.layers):
            if (idx != 0):
                weight_matrice: np.array = np.array([neuron.weights for neuron in layer.neurons])
                bias_matrice: np.array = np.array([neuron.bias for neuron in layer.neurons]) 

                layer.weight_matrice = weight_matrice
                layer.bias_matrice = bias_matrice

            value_matrice: np.array = np.array([neuron.value for neuron in layer.neurons])
            
            layer.value_matrice = value_matrice   

    def update_neurons_from_matrices(self):
        # Memperbarui bobot & bias neuron dari matriks yang tersimpan di layer
        for layer in self.layers[1:]:
            weight_matrice: np.array = layer.weight_matrice
            bias_matrice: np.array = layer.bias_matrice
            value_matrice: np.array = layer.value_matrice

            for i, neuron in enumerate(layer.neurons):
                neuron.weights = weight_matrice[i]
                neuron.bias = bias_matrice[i]
                neuron.value = value_matrice[i]

    def forward_propagation(self):
        for idx, layer in enumerate(self.layers) :
            if (idx != 0):
                # Mencari net
                hidden_value = np.dot(layer.weight_matrice, last_value) + layer.bias_matrice
                # Fungsi Aktivasi
                hidden_value = self.activation_func[idx].func(hidden_value)
                #Update nilai matriks
                layer.value_matrice = hidden_value
            # Nilai input untuk layer berikutnya
            last_value = layer.value_matrice
        # Fungsi Loss untuk nilai error
        error = self.lost_function.func(self.target, last_value)
        # Update value tiap neuron setelah 
        self.update_neurons_from_matrices()
        return error

    def backward_propagation(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

# Testing
from visualize import visualize_ffnn

ffnn = FFNN([1, 1], [1, 1], 2, 4, 2, 2, [FungsiAktivasi("relu") for _ in range(4)], weight_init_method = "uniform", lower_bound = 0, upper_bound = 1, seed = 42)
ffnn.forward_propagation()
visualize_ffnn(ffnn)