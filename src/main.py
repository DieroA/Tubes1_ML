import numpy as np
import os

from typing import List, Optional

from Model.FFNN import FFNN
from Fungsi.FungsiAktivasi import FungsiAktivasi, RELU, SIGMOID, LINEAR, TANH, SOFTMAX
from Fungsi.FungsiLoss import FungsiLoss, CCE, MSE, BCE
from Fungsi.visualize import visualize_network, plot_gradient_dist, plot_weight_dist, grafik_loss

from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer

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

""" --------------------------------------------------------- """

# Constants
INPUT_SIZE: int = 784
OUTPUT_SIZE: int = 10

# Variables
depth: int = 1
width: int = 128
activation_func: List[FungsiAktivasi] = [LINEAR] + [TANH for _ in range(depth)] + [SOFTMAX] 
loss_func: List[FungsiLoss] = CCE
weight_init_method: List[str] = ["uniform" for _ in range(2 + depth)]

if __name__ == "__main__":
    print("""
          --------------------------------------------------------

            ░▒▓████████▓▒░▒▓████████▓▒░▒▓███████▓▒░░▒▓███████▓▒░  
            ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
            ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
            ░▒▓██████▓▒░ ░▒▓██████▓▒░ ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
            ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
            ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
            ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░
          
          --------------------------------------------------------

          Rici Trisna Putra (13522026)
          Imam Hanif Mulyarahman (13522030)
          Diero Arga Purnama (13522056)
    """)

    model: Optional[FFNN] = None

    # Opsi untuk load
    load_save: str = str(input("Apakah anda ingin memuat simpanan model? [Ya/Tidak]: ")).strip().lower()
    if load_save == "ya":
        load_path: str = str(input("Masukkan path file .pkl: ")).strip()
        
        if os.path.exists(load_path + ".pkl"):
            model = FFNN.load(load_path)
            print("Model berhasil dimuat.")
        else:
            print("Model gagal dimuat, file tidak ditemukan.")

    # Init model (Bukan load  / load gagal)
    if model is None:
        # Input, asumsi data yang dimasukkan valid.
        batch_size: int = int(input("Batch size: "))
        learning_rate: float = float(input("Learning rate: "))
        jumlah_epoch: int = int(input("Jumlah epoch: "))
        verbose: int = int(input("Verbose [0/1]: "))

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
    visualize: str = str(input("Apakah anda ingin menampilkan visualisasi model dan distribusi bobot? [Ya/Tidak]: ")).strip().lower()
    if visualize == "ya":
        visualize_network(model.layers)
        plot_gradient_dist([0, 1], model.layers)
        plot_weight_dist([0, 1], model.layers)
        grafik_loss(history)


    # Save
    save_model: str = str(input("Apakah anda ingin menyimpan model ini? [Ya/Tidak]: ")).strip().lower()
    if save_model == "ya":
        save_path: str = str(input("Masukkan path untuk menyimpan model: ")).strip()

        FFNN.save(save_path, model)
    
    """ Bandingkan dengan scikit """

    compare: str = str(input("Apakah anda ingin melakukan perbandingan dengan model scikitlearn? [Ya/Tidak]: ")).strip().lower()
    if compare != "ya":
        exit()

    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    sklearn_model = MLPRegressor(
        hidden_layer_sizes=(width,) * depth,
        activation='tanh',
        solver='adam',
        alpha=0.0001,
        batch_size=batch_size,
        learning_rate_init=learning_rate,
        max_iter=jumlah_epoch,
        random_state=42,
        verbose=verbose
    )

    sklearn_model.fit(X_train, np.argmax(y_train_onehot, axis=1))
    
    sklearn_train_pred = sklearn_model.predict(X_train)
    sklearn_test_pred = sklearn_model.predict(X_test)
    
    def to_onehot(predictions, num_classes=10):
        predictions = predictions.reshape(-1, 1) if predictions.ndim == 1 else predictions

        if predictions.ndim == 2:
            predictions = np.argmax(predictions, axis=1)

        onehot = np.zeros((predictions.shape[0], num_classes))
        onehot[np.arange(predictions.shape[0]), predictions.astype(int)] = 1
        return onehot

    sklearn_train_pred_onehot = to_onehot(sklearn_train_pred)
    sklearn_test_pred_onehot = to_onehot(sklearn_test_pred)
    
    # Accuracy Comparison
    print("\nPerbandingan dengan FFNN scikit\n")
    print(f"Model Train Accuracy: {accuracy(y_train_onehot, train_pred):.4f}")
    print(f"Model Test Accuracy: {accuracy(y_test_onehot, test_pred):.4f}")
    print(f"Scikit-learn Train Accuracy: {accuracy(y_train_onehot, sklearn_train_pred_onehot):.4f}")
    print(f"Scikit-learn Test Accuracy: {accuracy(y_test_onehot, sklearn_test_pred_onehot):.4f}")


