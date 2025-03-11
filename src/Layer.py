from Neuron import Neuron
from typing import List
from FungsiAktivasi import FungsiAktivasi

class Layer:
    def __init__(self, neurons: List[Neuron], activation_func: FungsiAktivasi):
        self.neurons = neurons
        self.activation_func = activation_func  