import networkx as nx
import matplotlib.pyplot as plt
from typing import List
from Model.Komponen.Layer import Layer
import numpy as np

def visualize_network(layers: List[Layer]):
    """
        Visualisasi FFNN dalam bentuk graf
        
        iᵢ  -> Input ke-i
        bᵢ  -> Bias di layer i
        hᵢⱼ -> Neuron ke-j di layer i 
        oᵢ  -> Output ke-i
    """

    def subscript(num: int) -> str:
        subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

        return str(num).translate(subscript_map)

    def get_label_neuron(n_layers: int, layer: int, n: int) -> str:
        if layer == 0:
            node_label = f"i{subscript(n)}"                    # Input layer
        elif layer == n_layers - 1:
            node_label = f"o{subscript(n)}"                    # Output layer
        else:
            node_label = f"h{subscript(layer)}{subscript(n)}"  # Hidden layer

        return node_label
    
    def get_label_bias(n_layers: int, layer: int, n: int):
        if layer == n_layers - 1:
            bias_label = f"o{subscript(n)}"
        else:
            bias_label = f"h{subscript(layer)}{subscript(n)}"

        return bias_label

    G = nx.DiGraph()

    pos = {}                            # Posisi neuron

    edge_labels_weight = {}             # Label bobot
    edge_labels_gradient = {}           # Label gradien bobot

    edge_labels_bias_weight = {}        # Label bobot bias
    edge_labels_bias_gradient = {}      # Store gradien bobot bias

    layer_offset = 3                    # Jarak antar layer
    neuron_spacing_factor = 0.01        # Jarak antar neuron dalam layer yang sama
    bias_spacing_factor = 0.01 * 0.25   # Jarak antara bias dengan neuron

    max_neurons = max(layer.n_neurons for layer in layers)
    y_positions = {}

    bias_y_pos = (max_neurons + 1) * bias_spacing_factor

    for i, layer in enumerate(layers):
        n_neurons = layer.n_neurons
        y_spacing = max_neurons / max(n_neurons, 1) * neuron_spacing_factor
        y_positions[i] = [-j * y_spacing for j in range(n_neurons)]

        # Neuron
        for j, neuron in enumerate(layer.neurons):
            neuron_label = get_label_neuron(len(layers), i, j)

            pos[neuron_label] = (i * layer_offset, y_positions[i][j])

            G.add_node(neuron_label)

        # Bias (kecuali layer input)
        if i > 0:
            bias_node = f"b{subscript(i)}"

            bias_x_pos = (i - 0.5) * layer_offset
            pos[bias_node] = (bias_x_pos, bias_y_pos)

            G.add_node(bias_node, color = "red")

            for j, neuron in enumerate(layer.neurons):
                bias_label  = get_label_bias(len(layers), i, j)

                G.add_edge(bias_node, bias_label)

                edge_labels_bias_weight[(bias_node, bias_label)] = f"B: {neuron.bias:.2f}"
                edge_labels_bias_gradient[(bias_node, bias_label)] = f"∇B: {neuron.bias_gradients:.2f}"

        # Hubungkan neuron di layer yang berbeda
        if i > 0:
            prev_layer = layers[i - 1]
            for j, _ in enumerate(prev_layer.neurons):
                for k, neuron in enumerate(layer.neurons):
                    weight = neuron.weights[j]
                    gradient = neuron.weights_gradients[j]

                    prev_label = f"i{subscript(j)}" if i == 1 else f"h{subscript(i-1)}{subscript(j)}"
                    current_label = f"o{subscript(k)}" if i == len(layers) - 1 else f"h{subscript(i)}{subscript(k)}"

                    edge_key = (prev_label, current_label)
                    
                    G.add_edge(*edge_key)

                    edge_labels_weight[edge_key] = f"W: {weight:.2f}"
                    edge_labels_gradient[edge_key] = f"∇W: {gradient:.2f}"

    # Plot network
    plt.figure(figsize = (12, 8))
    nx.draw(G, pos, with_labels = True, node_color = "skyblue", edge_color = "gray", node_size = 1000, font_size = 8)

    # Display W
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels_weight, font_size = 6, label_pos = 0.2, 
                                 bbox = dict(facecolor = "yellow", alpha = 0.5, edgecolor = "none"))

    # Display ∇W
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels_gradient, font_size = 6, label_pos = 0.83, 
                                 bbox = dict(facecolor = "lightyellow", alpha = 0.5, edgecolor = "none"))

    # Display B
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels_bias_weight, font_size = 6, label_pos = 0.2, 
                                 bbox = dict(facecolor = "lightcoral", alpha = 0.5, edgecolor = "none"))

    # Display ∇B
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels_bias_gradient, font_size = 6, label_pos = 0.83, 
                                 bbox = dict(facecolor = "lightpink", alpha = 0.5, edgecolor = "none"))

    # Display plot
    plt.title("FFNN", fontsize = 12)
    plt.show()

def plot_weight_dist(layer_idxs: List[int], layers: List[Layer]):
    # 0 -> input
    # len(layers) - 1 -> output

    displayed_layers: List[Layer] = []
    for idx in layer_idxs:
        if idx < 0 or idx > len(layers) - 2:
            raise ValueError(f"Layer {idx} tidak valid.")

        displayed_layers.append(layers[idx + 1])

    # Membuat plot histogram
    plt.figure(figsize=(10, 5))

    for layer in displayed_layers:
        weights = np.concatenate([neuron.weights for neuron in layer.neurons])

        plt.hist(weights, bins=30, alpha=0.6, label=f"Layer {layer_idxs[displayed_layers.index(layer)]}")

    plt.xlabel("Nilai Bobot")
    plt.ylabel("Frekuensi")
    plt.title("Distribusi Bobot di Layer yang Dipilih")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_gradient_dist(layer_idxs: List[int], layers: List[Layer]):
    # 0 -> input
    # len(layers) - 1 -> output

    displayed_layers: List[Layer] = []
    for idx in layer_idxs:
        if idx < 0 or idx > len(layers) - 2:
            raise ValueError(f"Layer {idx} tidak valid..")

        displayed_layers.append(layers[idx + 1])

    # Membuat plot histogram
    plt.figure(figsize=(10, 5))

    for layer in displayed_layers:
        gradients = np.concatenate([neuron.weights_gradients for neuron in layer.neurons])

        plt.hist(gradients, bins=30, alpha=0.6, label=f"Layer {layer_idxs[displayed_layers.index(layer)]}")

    plt.xlabel("Nilai Gradien Bobot")
    plt.ylabel("Frekuensi")
    plt.title("Distribusi Gradien Bobot di Layer yang Dipilih")
    plt.legend()
    plt.grid(True)
    plt.show()