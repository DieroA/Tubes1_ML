from Fungsi.FungsiAktivasi import FungsiAktivasi
from Fungsi.FungsiLoss import FungsiLoss
# from Scaler.StandardScaler import StandardScalers
from Komponen.Layer import Layer
from typing import List
from tqdm import tqdm

import numpy as np

class FFNN:
    def __init__(self, input_data: np.array, output_data: np.array, input_size: int, hidden_size: int, output_size: int, n_hidden: int, batch_size: int, learning_rate: float, epoch: int,
                 activation_func: List[FungsiAktivasi],  loss_func: FungsiLoss = FungsiLoss("mse"), weight_init_method: List[str] = [], 
                 lower_bound: float = None, upper_bound: float = None, mean: float = None, 
                 variance: float = None, seed: int = 42):
        """ 
            Inisialisasi model.
            
            input_size: Jumlah input
            hidden_size: Jumlah neuron per layer
            output_size: Jumlah output / kelas
            n_hidden: Jumlah hidden layer
        """
        
        self.layers: List[Layer] = []
        self.activation_func: List[FungsiAktivasi] = activation_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch = epoch

        if len(weight_init_method) == 0:
            weight_init_method = ["zero" for _ in range(2 + n_hidden)]

        # Input layer
        self.layers.append(Layer(input_size, 0, weight_init_method[0], activation_func[0], lower_bound, upper_bound, mean, variance, seed))
        
        # Hidden layer(s)
        prev_size: int = input_size
        for i in range(n_hidden):
            self.layers.append(Layer(hidden_size, prev_size, weight_init_method[i + 1], activation_func[i + 1], lower_bound, upper_bound, mean, variance, seed))
            prev_size = hidden_size

        # Output layer
        self.layers.append(Layer(output_size, hidden_size, weight_init_method[-1], activation_func[-1], lower_bound, upper_bound, mean, variance, seed))
        
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

    def forward_propagation(self, batch_input, batch_output):
        for idx, layer in enumerate(self.layers) :
            if (idx != 0):
                # Mencari net
                hidden_value = np.dot(layer.weight_matrice, last_value.transpose()) + layer.bias_matrice
                # Fungsi Aktivasi
                # print(idx)
                # print(hidden_value)
                hidden_value = self.activation_func[idx].func(hidden_value)
                #Update nilai matriks
                layer.value_matrice = hidden_value.transpose()
            else:
                layer.value_matrice = batch_input
            # Nilai input untuk layer berikutnya
            last_value = layer.value_matrice
        # Fungsi Loss untuk nilai error
        error = self.lost_function.func(batch_output, last_value)
        # Update value tiap neuron setelah 
        self.update_neurons_from_matrices()
        return error

    def backward_propagation(self, batch_output):

        # Inisialisasi List Gradien
        gradients = []
        
        # Mulai dari output layer
        output_layer = self.layers[-1]
        
        # Hitung loss gradient dan activation_derivative
        loss_gradient = self.lost_function.derivative(batch_output, output_layer.value_matrice)

        # Calculate output gradient
        if self.activation_func[-1].name == "softmax":
            batch_gradients = []
            for i in range(output_layer.value_matrice.shape[0]):  # Loop over batch
                jacobian = self.activation_func[-1].derivative(output_layer.value_matrice[i])
                batch_gradients.append(np.dot(loss_gradient[i], jacobian))
            output_gradient = np.array(batch_gradients)
        else:
            activation_derivative = self.activation_func[-1].derivative(output_layer.value_matrice)
            output_gradient = loss_gradient * activation_derivative
        
        gradients.insert(0, output_gradient)

        # Gradien mengalir dari layer belakang ke depan
        for i in range(len(self.layers)-2, 0, -1):
            current_layer = self.layers[i]
            next_layer = self.layers[i+1]
            
            error = np.dot(gradients[0], next_layer.weight_matrice)
        
            if self.activation_func[i].name == "softmax":
                batch_gradients = []
                for sample_idx in range(current_layer.value_matrice.shape[0]):
                    jacobian = self.activation_func[i].derivative(current_layer.value_matrice[sample_idx])
                    batch_gradients.append(np.dot(error[sample_idx], jacobian))
                layer_gradient = np.array(batch_gradients)
            else:
                activation_derivative = self.activation_func[i].derivative(current_layer.value_matrice)
                layer_gradient = error * activation_derivative
                
            gradients.insert(0, layer_gradient)

        # Update bobot dan bias dengan gradien
        for i in range(1, len(self.layers)):
            current_layer = self.layers[i]
            prev_layer = self.layers[i-1]
            
            # Hitung Gradients
            current_layer.weight_gradients = np.dot(gradients[i-1].T, prev_layer.value_matrice) / self.batch_size
            current_layer.bias_gradients = np.mean(gradients[i-1], axis=0, keepdims=True).T
            
            # Update parameter
            current_layer.weight_matrice -= self.learning_rate * current_layer.weight_gradients
            current_layer.bias_matrice -= self.learning_rate * current_layer.bias_gradients
        
        # Update neuron
        self.update_neurons_from_matrices()
    
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
    
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.epoch):
            epoch_train_loss = 0
            epoch_val_loss = 0
            
            # Fase Training
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i+self.batch_size]
                batch_y = y_train[i:i+self.batch_size]
                
                self.layers[0].value_matrice = batch_X
                self.target = batch_y
                
                batch_loss = self.forward_propagation(batch_X, batch_y)
                self.backward_propagation(batch_y)
                
                epoch_train_loss += batch_loss * len(batch_X)
            
            epoch_train_loss /= len(X_train)
            history['train_loss'].append(epoch_train_loss)
            
            # Fase validasi
            if X_val is not None and y_val is not None:
                for i in range(0, len(X_val), self.batch_size):
                    batch_X = X_val[i:i+self.batch_size]
                    batch_y = y_val[i:i+self.batch_size]
                    
                    self.layers[0].value_matrice = batch_X
                    self.target = batch_y
                    
                    val_loss = self.forward_propagation(batch_X, batch_y)
                    epoch_val_loss += val_loss * len(batch_X)
                
                epoch_val_loss /= len(X_val)
                history['val_loss'].append(epoch_val_loss)
            
            # Tampilkan progress
            if verbose == 1:
                desc = f"Epoch {epoch+1}/{self.epoch}"
                if X_val is not None and y_val is not None:
                    desc += f" - loss: {epoch_train_loss:.4f} - val_loss: {epoch_val_loss:.4f}"
                else:
                    desc += f" - loss: {epoch_train_loss:.4f}"
                
                # tqdm.write(desc)
        
        return history

    def predict(self, input):
        for idx, layer in enumerate(self.layers) :
            if (idx != 0):
                # Mencari net
                hidden_value = np.dot(layer.weight_matrice, last_value.transpose()) + layer.bias_matrice
                # Fungsi Aktivasi
                # print(idx)
                # print(hidden_value)
                hidden_value = self.activation_func[idx].func(hidden_value)
                #Update nilai matriks
                last_value = hidden_value.transpose()
            else:
                last_value = input
        
        # Return output layer values
        return last_value

    def save(self):
        pass

    def load(self):
        pass

""" -------------------------------------------------------------------------------------------------------------------- """

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def display_matrices(model: FFNN):
    print("=== Neural Network Matrices ===")
    for idx, layer in enumerate(model.layers):
        print(f"\nLayer {idx}:")
        if idx == 0:
            print("  Input Values:\n", layer.value_matrice)
        else:
            print("  Weights:\n", layer.weight_matrice)
            print("  Weights Gradients:\n", layer.weight_gradients)
            print("  Biases:\n", layer.bias_matrice)
            print("  Biases Gradients:\n", layer.bias_gradients)
            print("  Values:\n", layer.value_matrice)


# Test data (simple linear relationship)
X_train = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
y_train = np.array([[30, 30], [60, 60], [90, 90], [120, 120]])

# Your FFNN implementation
input_size = 2
hidden_size = 4
output_size = 2
n_hidden = 2

fungsi_aktivasi = [FungsiAktivasi("relu")] + [FungsiAktivasi("relu") for _ in range(n_hidden)] + [FungsiAktivasi("linear")]
weight_init_method = ["uniform" for _ in range(2 + n_hidden)]

custom_nn = FFNN(
    X_train, y_train,
    input_size, hidden_size, output_size, n_hidden,
    batch_size=3,
    activation_func=fungsi_aktivasi,
    learning_rate=0.0001,
    epoch=2000,
    loss_func=FungsiLoss("mse"),
    weight_init_method=weight_init_method,
    lower_bound=0, upper_bound=1,
    seed=42
)

# Train your model
print("Training Custom FFNN...")
custom_history = custom_nn.train(
    X_train, y_train,
    verbose=1
)

# Scikit-learn's MLP
print("\nTraining scikit-learn MLP...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

sklearn_nn = MLPRegressor(
    hidden_layer_sizes=(4, 4),
    activation='relu',
    solver='sgd',
    learning_rate_init=0.01,
    batch_size=2,
    max_iter=1000,
    random_state=42
)
sklearn_nn.fit(X_scaled, y_train)

# Compare predictions
test_input = np.array([[2.5, 2.5]])
test_input_scaled = scaler.transform(test_input)

custom_pred = custom_nn.predict(test_input)
sklearn_pred = sklearn_nn.predict(test_input_scaled)

display_matrices(custom_nn)

print("\n=== Prediction Comparison ===")
print(f"Input: {test_input[0]}")
print(f"Custom FFNN prediction: {custom_pred[0]}")
print(f"Scikit-learn prediction: {sklearn_pred[0]}")
print(f"Expected output: [75, 75]")

# Compare final losses
final_custom_loss = custom_history['train_loss'][-1]
final_sklearn_loss = np.mean((sklearn_nn.predict(X_scaled) - y_train) ** 2)

print("\n=== Final Loss Comparison ===")
print(f"Custom FFNN final loss: {final_custom_loss:.4f}")
print(f"Scikit-learn final loss: {final_sklearn_loss:.4f}")

from visualize import visualize_network

visualize_network(custom_nn.layers)


# # Testing
# from visualize import visualize_ffnn


# # Testing
# input_data = np.array([[1, 1]])
# output_target = np.array([[30, 30]])

# # Jumlah neuron per layer
# input_size = 2
# hidden_size = 4
# output_size = 2

# # Jumlah hidden layer
# n_hidden = 2

# # Fungsi Aktivasi - Now properly initialized with derivative support
# fungsi_aktivasi = [FungsiAktivasi("linear")] + [FungsiAktivasi("relu") for _ in range(n_hidden)] + [FungsiAktivasi("linear")]

# ffnn = FFNN(
#     input_data, output_target, 
#     input_size, hidden_size, output_size, n_hidden, 
#     fungsi_aktivasi, 
#     loss_func=FungsiLoss("mse"),
#     weight_init_method="uniform", 
#     lower_bound=0, upper_bound=1, 
#     seed=42
# )

# # Test forward and backward
# print("=== Initial Pass ===")
# loss_value = ffnn.forward_propagation()
# display_matrices(ffnn)
# print(f"Initial Loss: {loss_value}")

# print("\n=== After Training ===")
# ffnn.train()
# display_matrices(ffnn)
# print(f"After Loss: {loss_value}")
