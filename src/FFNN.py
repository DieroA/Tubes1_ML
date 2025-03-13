from FungsiAktivasi import FungsiAktivasi
from FungsiLoss import FungsiLoss
from Layer import Layer
from typing import Any, List

class FFNN:
    def __init__(self, input: List[float], output: List[float], input_size: int, hidden_size: int, output_size: int, n_hidden: int,
                 activation_func: List[FungsiAktivasi],  loss_func: FungsiLoss = FungsiLoss("mse"), weight_init_method: str = "uniform", 
                 lower_bound: float = 10, upper_bound: float = 100, mean: float = None, 
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
        
    def forward_propagation(self):
        pass

    def backward_propagation(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

f: FungsiAktivasi = FungsiAktivasi("relu")
FFNN([1, 1], [1, 1], 2, 2, 2, 1, [f for _ in range(3)])

from visualize import visualize_ffnn
visualize_ffnn(FFNN([1, 1], [1, 1], 2, 4, 2, 2, [f for _ in range(4)]))