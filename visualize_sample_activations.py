#!/usr/bin/env python
"""
Visualize GELU neuron activations for specific sample IDs.

This script generates rectangular heatmaps showing active neurons for specified samples.
Each layer is represented as a pillar-like heatmap with active neurons shown in red.
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def load_dataset(dataset_dir, model_type):
    """Load all necessary data files"""
    # Load activation dataset
    dataset_path = os.path.join(dataset_dir, f"{model_type}_activation_dataset.csv")
    dataset_df = pd.read_csv(dataset_path)
    
    # Load neuron activations
    activations_path = os.path.join(dataset_dir, f"{model_type}_neuron_activations.pickle")
    if os.path.exists(activations_path):
        with open(activations_path, 'rb') as f:
            activations = pickle.load(f)
    else:
        # Try JSON version if pickle not available
        activations_path = os.path.join(dataset_dir, f"{model_type}_neuron_activations.json")
        with open(activations_path, 'r') as f:
            activations = json.load(f)
    
    # Convert string keys to integers if needed
    if isinstance(next(iter(activations.keys())), str):
        activations = {int(k): v for k, v in activations.items()}
    
    # Load neuron attributes
    attr_path = os.path.join(dataset_dir, f"{model_type}_neuron_attributes.csv")
    attr_df = pd.read_csv(attr_path)
    
    return dataset_df, activations, attr_df


def get_model_dimensions(attr_df):
    """Extract model dimensions from neuron attributes"""
    n_layers = attr_df['layer'].max() + 1
    dim_per_layer = attr_df.groupby('layer').size().iloc[0]
    return n_layers, dim_per_layer


def visualize_sample_activation(sample_id, dataset_df, activations, attr_df, export_dir, model_type):
    """Create heatmap visualization for a specific sample ID"""
    if str(sample_id) in activations:
        sample_activations = activations[str(sample_id)]
    elif sample_id in activations:
        sample_activations = activations[sample_id]
    else:
        print(f"Sample ID {sample_id} not found in activations data.")
        return None
    
    # Get sample metadata
    sample_data = dataset_df[dataset_df['sample_id'] == sample_id]
    if len(sample_data) == 0:
        print(f"Sample ID {sample_id} not found in dataset.")
        return None
    
    input_text = sample_data['input_text'].iloc[0]
    output_token = sample_data['output'].iloc[0]
    
    # Get model dimensions
    n_layers, dim_per_layer = get_model_dimensions(attr_df)
    
    # Create a binary activation matrix for this sample
    activation_matrix = np.zeros((n_layers, dim_per_layer), dtype=np.int8)
    
    # Fill the matrix with activations
    for neuron_id in sample_activations:
        # Use neuron attributes to get layer and dimension
        neuron_attrs = attr_df[attr_df['node_id'] == neuron_id]
        if len(neuron_attrs) > 0:
            layer = neuron_attrs['layer'].iloc[0]
            dim = neuron_attrs['dimension'].iloc[0]
            activation_matrix[layer, dim] = 1
    
    # Create figure with subplots for each layer
    fig, axes = plt.subplots(1, n_layers, figsize=(20, 8), gridspec_kw={'width_ratios': [1] * n_layers})
    
    # Create a colormap with blue for 0 and red for 1
    colors = ['#0000FF', '#FF0000']  # Blue, Red
    cmap = ListedColormap(colors)
    
    # Plot each layer as a vertical heatmap
    for layer in range(n_layers):
        ax = axes[layer] if n_layers > 1 else axes
        
        # Reshape the layer data to make it more visible (optional)
        # We reshape a 1D array into a 2D grid to make individual neurons more visible
        width = int(np.sqrt(dim_per_layer))
        height = dim_per_layer // width
        if width * height < dim_per_layer:
            height += 1
        
        # Create a rectangular grid for better visualization
        layer_grid = np.zeros((height, width), dtype=np.int8)
        for i in range(min(dim_per_layer, width * height)):
            row = i // width
            col = i % width
            layer_grid[row, col] = activation_matrix[layer, i]
        
        # Plot the heatmap
        im = ax.imshow(layer_grid, cmap=cmap, aspect='auto', interpolation='none')
        
        # Add layer number as title
        ax.set_title(f"Layer {layer}")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add count of active neurons
        active_count = np.sum(activation_matrix[layer])
        ax.text(0.5, -0.1, f"Active: {active_count}", transform=ax.transAxes, 
                ha='center', fontsize=8)
    
    # Set overall title with input and output text
    plt.suptitle(f"Sample ID: {sample_id}\nInput: '{input_text}'\nOutput Token: '{output_token}'", 
                fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.85])  # Make room for suptitle
    
    # Save figure
    output_path = os.path.join(export_dir, f"{model_type}_sample_{sample_id}_activation.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path


def main():
    # python visualize_sample_activations.py --dataset_dir parcellation_datasets/test_1 --sample_ids 0,50,200 --export_dir parcellation_datasets/test_1/visualizations
    parser = argparse.ArgumentParser(description="Visualize GELU neuron activations for specific samples")
    parser.add_argument('--dataset_dir', required=True,
                        help='Directory containing the dataset and activation files')
    parser.add_argument('--sample_ids', required=True,
                        help='Comma-separated list of sample IDs to visualize')
    parser.add_argument('--export_dir', required=True,
                        help='Directory to save the visualizations')
    parser.add_argument('--model_type', default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='GPT-2 model size')
    args = parser.parse_args()
    
    # Parse sample IDs
    sample_ids = [int(id.strip()) for id in args.sample_ids.split(',')]
    
    # Create export directory if it doesn't exist
    os.makedirs(args.export_dir, exist_ok=True)
    
    # Load dataset and activations
    dataset_df, activations, attr_df = load_dataset(args.dataset_dir, args.model_type)
    
    # Generate visualizations for each sample ID
    for sample_id in sample_ids:
        print(f"Generating visualization for sample ID {sample_id}...")
        output_path = visualize_sample_activation(
            sample_id, dataset_df, activations, attr_df, args.export_dir, args.model_type
        )
        if output_path:
            print(f"Visualization saved to {output_path}")
    
    print("Visualization complete.")


if __name__ == "__main__":
    main() 