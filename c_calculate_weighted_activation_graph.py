#!/usr/bin/env python
# python c_calculate_weighted_activation_graph.py --input_dir parcellation_datasets/test_1 --output_path parcellation_datasets/test_1/gpt2_weighted_graph.gpickle
"""
Calculates node and edge weights for a GELU neuron graph based on activation data.

This script takes the output directory from b_generate_activation_dataset.py, 
loads the neuron attributes, base edges, activation frequencies, and activation matrix.
It then constructs a networkx graph where:
- Nodes represent GELU neurons with attributes: layer, dimension, activation_count.
- Edges represent connections between neurons in adjacent layers with attribute: co_activation_count.

Example usage:
python c_calculate_node_weighted_activation_graph.py --input_dir parcellation_datasets/test_1 --output_path parcellation_datasets/test_1/gpt2_weighted_graph.gpickle
"""

import os
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from glob import glob

def find_model_type(input_dir):
    """Infer model type from filenames in the input directory."""
    attribute_files = glob(os.path.join(input_dir, '*_neuron_attributes.csv'))
    if not attribute_files:
        raise FileNotFoundError(f"No '*_neuron_attributes.csv' file found in {input_dir}")
    
    # Extract model type from the first found file
    filename = os.path.basename(attribute_files[0])
    model_type = filename.replace('_neuron_attributes.csv', '')
    print(f"Inferred model type: {model_type}")
    return model_type

def load_base_edges(edge_file_path):
    """Load base edges from the .el file."""
    edges = []
    with open(edge_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    edges.append((u, v))
                except ValueError:
                    print(f"Warning: Skipping invalid line in edge file: {line.strip()}")
    return edges

def main():
    parser = argparse.ArgumentParser(description="Calculate weighted activation graph")
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing output from b_generate_activation_dataset.py')
    parser.add_argument('--output_path', required=True,
                        help='Path to save the output weighted graph (.gpickle)')
    args = parser.parse_args()

    # Ensure input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return

    try:
        model_type = find_model_type(args.input_dir)
    except FileNotFoundError as e:
        print(e)
        return
        
    # Define file paths
    attr_file = os.path.join(args.input_dir, f"{model_type}_neuron_attributes.csv")
    edge_file = os.path.join(args.input_dir, f"{model_type}_neuron_edges.el")
    freq_file = os.path.join(args.input_dir, f"{model_type}_neuron_frequency.csv")
    matrix_file = os.path.join(args.input_dir, f"{model_type}_activation_matrix.npz")

    # Check if required files exist
    required_files = [attr_file, edge_file, freq_file, matrix_file]
    for f_path in required_files:
        if not os.path.exists(f_path):
            print(f"Error: Required file not found: {f_path}")
            return

    print("Loading data...")
    # Load node attributes
    node_attributes_df = pd.read_csv(attr_file)

    # Load node activation frequencies (for node weights)
    node_frequency_df = pd.read_csv(freq_file)
    node_weights = node_frequency_df.set_index('neuron_id')['activation_count'].to_dict()

    # Load base edges
    base_edges = load_base_edges(edge_file)

    # Load activation matrix and neuron map (for edge weights)
    try:
        npz_data = np.load(matrix_file, allow_pickle=True)
        activation_matrix = npz_data['matrix']
        # Ensure neuron_map is loaded correctly if saved as an object array
        neuron_map = npz_data['neuron_map'].item() if npz_data['neuron_map'].ndim == 0 else npz_data['neuron_map']
        if isinstance(neuron_map, np.ndarray) and neuron_map.ndim == 0: # Handle case where item() wasn't called during save
             neuron_map = neuron_map.item()
        if not isinstance(neuron_map, dict):
             raise TypeError(f"neuron_map is not a dictionary. Type: {type(neuron_map)}")

    except Exception as e:
        print(f"Error loading activation matrix file {matrix_file}: {e}")
        return

    print("Building weighted graph...")
    G = nx.Graph()

    # Add nodes with attributes and weights
    for _, row in node_attributes_df.iterrows():
        node_id = row['node_id']
        activation_count = node_weights.get(node_id, 0) # Default to 0 if neuron never activated
        G.add_node(
            node_id, 
            layer=row['layer'], 
            dimension=row['dimension'],
            activation_count=activation_count
        )

    # Add edges with co-activation weights
    num_edges = len(base_edges)
    processed_edges = 0
    print(f"Calculating weights for {num_edges} base edges...")
    
    for u, v in base_edges:
        co_activation_count = 0
        # Check if both neurons were ever active (i.e., present in neuron_map)
        if u in neuron_map and v in neuron_map:
            try:
                idx_u = neuron_map[u]
                idx_v = neuron_map[v]
                # Calculate dot product of the corresponding columns in the activation matrix
                # Ensure columns exist before slicing
                if idx_u < activation_matrix.shape[1] and idx_v < activation_matrix.shape[1]:
                     co_activation_count = int(np.dot(activation_matrix[:, idx_u], activation_matrix[:, idx_v]))
                else:
                     print(f"Warning: Index out of bounds for neuron {u} ({idx_u}) or {v} ({idx_v}). Matrix shape: {activation_matrix.shape}")

            except KeyError as e:
                 print(f"Warning: Neuron ID {e} not found in neuron_map. Skipping co-activation calculation for edge ({u}, {v}).")
            except IndexError as e:
                 print(f"Warning: Index error accessing activation matrix for edge ({u}, {v}): {e}. Matrix shape: {activation_matrix.shape}, Indices: {idx_u}, {idx_v}")
        
        # Add edge with its weight (0 if co-activation couldn't be calculated or was zero)
        G.add_edge(u, v, co_activation_count=co_activation_count)
        
        processed_edges += 1
        if processed_edges % 100000 == 0: # Print progress periodically
             print(f"Processed {processed_edges}/{num_edges} edges...")

    print(f"Finished processing {processed_edges} edges.")
    print(f"Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Save the graph
    print(f"Saving weighted graph to {args.output_path}...")
    with open(args.output_path, 'wb') as f:
        pickle.dump(G, f)

    print("Weighted graph saved successfully.")

if __name__ == "__main__":
    main() 