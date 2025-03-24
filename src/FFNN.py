from Fungsi.FungsiAktivasi import FungsiAktivasi
from Fungsi.FungsiLoss import FungsiLoss
from Komponen.Layer import Layer
from typing import List

import numpy as np

class FFNN:
    def __init__(self, input_data: np.array, output_data: np.array, input_size: int, hidden_size: int, output_size: int, n_hidden: int,
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
        self.batch_size: int = input_data.shape[0]

        # Input layer
        self.layers.append(Layer(input_size, 0, weight_init_method, activation_func[0], lower_bound, upper_bound, mean, variance, seed))
        
        # Hidden layer(s)
        prev_size: int = input_size
        for i in range(n_hidden):
            self.layers.append(Layer(hidden_size, prev_size, weight_init_method, activation_func[i + 1], lower_bound, upper_bound, mean, variance, seed))
            prev_size = hidden_size

        # Output layer
        self.layers.append(Layer(output_size, hidden_size, weight_init_method, activation_func[-1], lower_bound, upper_bound, mean, variance, seed))
        
        # Loss Function
        self.lost_function: FungsiLoss = loss_func

        # Target Value
        self.target: np.array = output_data

        # Init matrix
        self.generate_matrices(input_data)

    def generate_matrices(self, input_data):
        # Menghasilkan matriks bobot & bias untuk setiap layer
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                layer.value_matrice = input_data
            else:
                layer.weight_matrice = np.array([neuron.weights for neuron in layer.neurons])
                layer.bias_matrice = np.array([neuron.bias for neuron in layer.neurons]).reshape(-1, 1)
                layer.value_matrice = np.zeros((self.batch_size, layer.n_neurons))

    def update_neurons_from_matrices(self):
        # Memperbarui bobot & bias neuron dari matriks yang tersimpan di layer
        for layer in self.layers[1:]:
            weight_matrice: np.array = layer.weight_matrice 
            bias_matrice: np.array = layer.bias_matrice
            value_matrice: np.array = layer.value_matrice

            for i, neuron in enumerate(layer.neurons):
                neuron.weights = weight_matrice[i]
                neuron.bias = bias_matrice[i, 0]
                neuron.value_matrice = value_matrice[:, i].reshape(-1, 1)

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

""" -------------------------------------------------------------------------------------------------------------------- """

# Testing
from visualize import visualize_ffnn

def display_matrices(model: FFNN):
    print("=== Neural Network Matrices ===")
    for idx, layer in enumerate(model.layers):
        print(f"\nLayer {idx}:")
        if idx == 0:
            print("  Input Values:\n", layer.value_matrice)
        else:
            print("  Weights:\n", layer.weight_matrice)
            print("  Biases:\n", layer.bias_matrice)
            print("  Values:\n", layer.value_matrice)

input_data = np.array([[1, 1], [1, 1]])
output_target = np.array([1, 1])

# Jumlah neuron per layer
input_size = 2
hidden_size = 4
output_size = 2

# Jumlah hidden layer
n_hidden = 2

# Fungsi Aktivasi
fungsi_aktivasi = [FungsiAktivasi("relu") for _ in range(2 + n_hidden)]

ffnn = FFNN(
    input_data, output_target, 
    input_size, hidden_size, output_size, n_hidden, 
    fungsi_aktivasi, weight_init_method = "uniform", 
    lower_bound = 0, upper_bound = 1, seed = 42
)

display_matrices(ffnn)