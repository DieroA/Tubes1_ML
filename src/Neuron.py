class Neuron:
    def __init__(self, weight_init_method: str, lower_bound: int = 0, upper_bound: int = 0):
        def random_uniform() -> float:
            # Inisialisasi bobot dengan distribusi uniform
            pass

        def random_normal() -> float:
            # Inisialisasi bobot dengan distribusi normal
            pass

        self.weight: float = 0 
        if (weight_init_method == "uniform"):
            self.weight = random_uniform()
        elif (weight_init_method == "normal"):
            self.weight = random_normal()
        else:
            raise ValueError("Metode inisialisasi bobot tidak valid!")

        pass

class Bias(Neuron):
    def __init__(self):
        super().__init__()