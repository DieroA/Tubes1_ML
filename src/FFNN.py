from FungsiAktivasi import FungsiAktivasi
from FungsiLoss import FungsiLoss
from Layer import Layer
from typing import Any, List

class FFNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, n_hidden: int,
                 activation_func: List[FungsiAktivasi] = FungsiAktivasi("relu"), 
                 loss_func: FungsiLoss = FungsiLoss("mse"), weight_init_method: str = "zero", 
                 lower_bound: float = None, upper_bound: float = None, mean: float = None, 
                 variance: float = None, seed: int = None):
        # Inisialisasi model.
        # input_size: Jumlah input
        # hidden_size: Jumlah neuron per layer
        # output_size: Jumlah output / kelas
        # n_hidden: Jumlah hidden layer

        self.layers: List[Layer] = []

        # Input layer
        self.layers.append(Layer(input_size, 0, weight_init_method, activation_func, lower_bound, upper_bound, mean, variance, seed))
        
        # Hidden layer(s)
        self.layers.append(Layer(hidden_size, input_size, weight_init_method, activation_func, lower_bound, upper_bound, mean, variance, seed))
        for _ in range(n_hidden - 1):
            self.layers.append(Layer(hidden_size, hidden_size, weight_init_method, activation_func, lower_bound, upper_bound, mean, variance, seed))
        
        # Output layer
        self.layers.append(Layer(output_size, hidden_size, weight_init_method, activation_func, lower_bound, upper_bound, mean, variance, seed))
        
    def forward_propagation(self, batch: Any) -> Any:
        pass

    def backward_propagation(self):
        pass

    def save(self):
        pass

    def load(self):
        pass