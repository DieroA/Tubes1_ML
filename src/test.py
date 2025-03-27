import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from Fungsi.FungsiAktivasi import FungsiAktivasi
from Fungsi.FungsiLoss import FungsiLoss
from FFNN import FFNN
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.utils import check_random_state

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

# Initialize FFNN
input_size = 784  # 28x28 images
hidden_size = 128
output_size = 10   # Digits 0-9
n_hidden = 1

# Use ReLU for hidden, softmax for output
fungsi_aktivasi = [FungsiAktivasi("linear")] + [FungsiAktivasi("relu") for _ in range(n_hidden)] + [FungsiAktivasi("relu")]

ffnn = FFNN(
    X_train, y_train_onehot,
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    n_hidden=n_hidden,
    batch_size = 10,
    learning_rate = 0.001,
    epoch = 20,
    activation_func=fungsi_aktivasi,
    loss_func=FungsiLoss("cce"),  # Categorical cross-entropy
    weight_init_method="uniform",
    lower_bound=-0.1,
    upper_bound=0.1,
    seed=42
)


# Train the network
history = ffnn.train(
    X_train, y_train_onehot,
    verbose=1
)

plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.legend()
plt.show()

# Evaluation
def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

train_pred = ffnn.predict(X_train)
test_pred = ffnn.predict(X_test)

print(f"Train accuracy: {accuracy(y_train_onehot, train_pred):.4f}")
print(f"Test accuracy: {accuracy(y_test_onehot, test_pred):.4f}")

# # Visualize learned weights (like the original example)
# plt.figure(figsize=(10, 5))
# scale = np.abs(ffnn.layers[1].weight_matrice).max()  # First hidden layer weights
# for i in range(10):
#     l1_plot = plt.subplot(2, 5, i + 1)
#     # For visualization, take weights going to one output neuron
#     if i < output_size:
#         weights = ffnn.layers[-1].weight_matrice[i].reshape(28, 28)
#     else:
#         weights = ffnn.layers[1].weight_matrice[:,i].reshape(28, 28)
#     l1_plot.imshow(
#         weights,
#         interpolation="nearest",
#         cmap=plt.cm.RdBu,
#         vmin=-scale,
#         vmax=scale,
#     )
#     l1_plot.set_xticks(())
#     l1_plot.set_yticks(())
#     l1_plot.set_xlabel(f"Neuron {i}")
# plt.suptitle("Learned weights visualization")
# plt.show()