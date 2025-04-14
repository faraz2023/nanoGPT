#!/usr/bin/env python
"""
Generate a graph representation of GELU activation neurons in the GPT model.
The script creates:
1. An edge list (.el) file representing connectivity between neurons
2. A CSV file with node attributes (layer, neuron dimension)
"""

import os
import argparse
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Generate GELU neuron graph for GPT model")
    parser.add_argument('--model_type', default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                       help='GPT-2 model size to use')
    parser.add_argument('--output_dir', default='gelu_graph', help='Directory to store the output files')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get model config based on the model type
    if args.model_type == 'gpt2':
        n_layer, n_head, n_embd = 12, 12, 768
    elif args.model_type == 'gpt2-medium':
        n_layer, n_head, n_embd = 24, 16, 1024
    elif args.model_type == 'gpt2-large':
        n_layer, n_head, n_embd = 36, 20, 1280
    elif args.model_type == 'gpt2-xl':
        n_layer, n_head, n_embd = 48, 25, 1600
    
    # Calculate the dimensions of GELU activations
    # In the MLP class, GELU is applied after c_fc which expands embedding to 4x
    gelu_dim = 4 * n_embd
    
    # Total number of GELU neurons
    total_neurons = n_layer * gelu_dim
    print(f"Model: {args.model_type}")
    print(f"Layers: {n_layer}, Embedding dimension: {n_embd}, GELU dimension: {gelu_dim}")
    print(f"Total neurons: {total_neurons}")
    
    # Generate node attributes
    node_attrs = []
    for layer_idx in range(n_layer):
        for neuron_idx in range(gelu_dim):
            node_id = layer_idx * gelu_dim + neuron_idx
            # Node attributes: ID, Layer, Dimension
            node_attrs.append((node_id, layer_idx, neuron_idx))
    
    # Generate edges between GELU neurons in adjacent layers
    edges = []
    for layer_idx in range(n_layer - 1):
        # Source nodes: all neurons in current layer
        src_start = layer_idx * gelu_dim
        src_end = src_start + gelu_dim
        
        # Target nodes: all neurons in next layer
        tgt_start = (layer_idx + 1) * gelu_dim
        tgt_end = tgt_start + gelu_dim
        
        # Each neuron in the current layer connects to all neurons in the next layer
        for src in range(src_start, src_end):
            for tgt in range(tgt_start, tgt_end):
                edges.append((src, tgt))
    
    # Write node attributes to CSV
    attr_file = os.path.join(args.output_dir, f"{args.model_type}_neuron_attributes.csv")
    with open(attr_file, 'w') as f:
        f.write("node_id,layer,dimension\n")
        for node_id, layer, dim in node_attrs:
            f.write(f"{node_id},{layer},{dim}\n")
    
    # Write edges to edge list file
    edge_file = os.path.join(args.output_dir, f"{args.model_type}_neuron_edges.el")
    with open(edge_file, 'w') as f:
        for src, tgt in edges:
            f.write(f"{src} {tgt}\n")
    
    print(f"Node attributes written to {attr_file}")
    print(f"Edge list written to {edge_file}")
    print(f"Total nodes: {len(node_attrs)}")
    print(f"Total edges: {len(edges)}")


if __name__ == "__main__":
    main() 