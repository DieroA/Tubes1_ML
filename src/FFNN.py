from FungsiAktivasi import FungsiAktivasi
from FungsiLoss import FungsiLoss
from Layer import Layer
from typing import Any, List

class FFNN:
    def __init__(self, n_neuron: int, activation_func: List[FungsiAktivasi], loss_func: FungsiLoss, 
                 weight_init_method: str = "zero"):

        pass

    def forward_propagation(self, batch: Any) -> Any:
        pass

    def backward_propagation(self):
        pass

    def save(self):
        pass

    def load(self):
        pass