import matplotlib.pyplot as plt
import networkx as nx

# Testing doang, tar hapus

def visualize_ffnn(ffnn):
    """Visualizes a Feedforward Neural Network (FFNN) with neuron values and weights on the edges."""
    
    G = nx.DiGraph()
    layer_sizes = [len(layer.neurons) for layer in ffnn.layers]
    
    pos = {}  # Store positions for plotting
    node_labels = {}  # Store neuron values as labels
    edge_labels = {}  # Store weights as edge labels
    
    # Assign positions to neurons layer by layer
    x_offset = 0  # Horizontal position (Layer index)
    for layer_idx, layer in enumerate(ffnn.layers):
        size = len(layer.neurons)
        y_offset = (size - 1) / 2  # Center neurons vertically
        
        for neuron_idx, neuron in enumerate(layer.neurons):
            node_id = f"L{layer_idx}_N{neuron_idx}"
            pos[node_id] = (x_offset, y_offset - neuron_idx)
            node_labels[node_id] = f"{neuron.value:.2f}"  # Show neuron value rounded to 2 decimal places
        
        x_offset += 1  # Move to next layer

    # Add neurons as nodes
    for node in pos:
        G.add_node(node)

    # Add edges (connections between neurons)
    for layer_idx in range(1, len(layer_sizes)):  # Start from 1 (input layer has no weights)
        prev_layer = ffnn.layers[layer_idx - 1]
        curr_layer = ffnn.layers[layer_idx]

        for j, neuron in enumerate(curr_layer.neurons):
            for i, weight in enumerate(neuron.weights):  # Each neuron has weights from all neurons in the previous layer
                G.add_edge(f"L{layer_idx-1}_N{i}", f"L{layer_idx}_N{j}")
                edge_labels[(f"L{layer_idx-1}_N{i}", f"L{layer_idx}_N{j}")] = f"{weight:.2f}"  # Show weight rounded to 2 decimal places

    # Draw the network
    plt.figure(figsize=(12, 7))
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=2000, node_color="lightblue", font_size=10, edge_color="gray", font_weight="bold")
    
    # Draw edge labels (weights)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="red")

    # Add title
    plt.title("Feedforward Neural Network Visualization with Neuron Values & Weights", fontsize=14)
    plt.show()
