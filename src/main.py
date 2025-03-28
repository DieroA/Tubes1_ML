import numpy as np

from typing import List
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer

from FFNN import FFNN
from Fungsi.FungsiAktivasi import FungsiAktivasi, RELU, SIGMOID, LINEAR, TANH, SOFTMAX
from Fungsi.FungsiLoss import FungsiLoss, CCE, MSE, BCE
from visualize import visualize_network, plot_gradient_dist, plot_weight_dist

""" 
    Load and Process data 
"""

# Load MNIST data
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=5000, test_size=10000
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert labels to one-hot encoding
lb = LabelBinarizer()
y_train_onehot = lb.fit_transform(y_train)
y_test_onehot = lb.transform(y_test)

""" 
    --------------------------------------------------------------------------
"""

# Constants
INPUT_SIZE: int = 784
OUTPUT_SIZE: int = 10

# Variables
depth: int = 2
width: int = 1
activation_func: List[FungsiAktivasi] = [LINEAR] + [TANH for _ in range(depth)] + [SOFTMAX] 
loss_func: List[FungsiLoss] = CCE
weight_init_method: List[str] = ["uniform" for _ in range(2 + depth)]

if __name__ == "__main__":
    # Input, asumsi data yang dimasukkan valid.
    batch_size: int = int(input("Batch size: "))
    learning_rate: float = float(input("Learning rate: "))
    jumlah_epoch: int = int(input("Jumlah epoch: "))
    verbose: int = int(input("Verbose: "))

    # Init model
    model: FFNN = FFNN(
        X_train, y_train_onehot,
        INPUT_SIZE, width, OUTPUT_SIZE,
        depth, batch_size, learning_rate, jumlah_epoch,
        activation_func, loss_func, weight_init_method,
        lower_bound = -0.1, upper_bound = 0.1, seed = 42
    )

    # Train model
    history = model.train(
        X_train, y_train_onehot,
        verbose = verbose
    )
    
    # Evaluation
    def accuracy(y_true, y_pred):
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print(f"Train accuracy: {accuracy(y_train_onehot, train_pred):.4f}")
    print(f"Test accuracy: {accuracy(y_test_onehot, test_pred):.4f}")

    # Visualize
    visualize_network(model.layers)
    plot_gradient_dist([0, 1], model.layers)
    plot_weight_dist([0, 1], model.layers)
    


