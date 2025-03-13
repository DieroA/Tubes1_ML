import numpy as np

class Neuron:
    def __init__(self, n_input: int, weight_init_method: str,
                 lower_bound: float = None, upper_bound: float = None,
                 mean: float = None, variance: float = None,
                 seed: int = None, value: float = 0.0):
        # Inisialisasi neuron.

        if seed is not None:
            np.random.seed(seed)

        def generateWeight() -> float:
            # Inisialisasi bobot neuron menggunakan metode yang dimasukkan."
            
            if weight_init_method == "zero":
                return 0
            elif (weight_init_method == "uniform"):
                if lower_bound is None or upper_bound is None:
                    raise ValueError("Inisialisasi uniform memperlukan lower_bound dan upper_bound yang terdefinisi.")
                return np.random.uniform(lower_bound, upper_bound)
            elif (weight_init_method == "normal"):
                if mean is None or variance is None:
                    raise ValueError("Inisialisasi uniform memperlukan mean dan variance yang terdefinisi.")
                return np.random.normal(loc = mean, scale = np.sqrt(variance))
            raise ValueError(f"Metode inisialisasi bobot {weight_init_method} tidak valid!")
        
        self.weights: np.array = np.array([generateWeight() for _ in range(n_input)])
        self.bias: float = generateWeight()

        self.value: float = value
