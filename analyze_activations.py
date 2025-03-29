#!/usr/bin/env python
"""
Analyze GELU neuron activations across different prompts.
This script provides tools to compare activations, identify common patterns,
and visualize activation differences between prompts and between samples.
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def load_dataset(csv_path, activations_path):
    """Load the dataset and activations"""
    # Load CSV dataset
    df = pd.read_csv(csv_path)
    
    # Load activations
    if activations_path.endswith('.pickle'):
        with open(activations_path, 'rb') as f:
            activations = pickle.load(f)
    elif activations_path.endswith('.json'):
        with open(activations_path, 'r') as f:
            activations = json.load(f)
    else:
        raise ValueError("Activations file must be .pickle or .json")
    
    # Convert string keys to integers if needed
    if isinstance(next(iter(activations.keys())), str):
        activations = {int(k): v for k, v in activations.items()}
    
    return df, activations


def create_activation_matrix(df, activations, max_neurons=None):
    """Create a binary matrix of sample_id x neuron_id"""
    # Get the maximum neuron ID
    if max_neurons is None:
        max_neurons = max(max(neurons) for neurons in activations.values() if neurons) + 1
    
    # Create empty matrix
    n_samples = len(df)
    matrix = np.zeros((n_samples, max_neurons), dtype=np.int8)
    
    # Fill matrix with activations
    for i, sample_id in enumerate(df['sample_id']):
        if sample_id in activations:
            for neuron_id in activations[sample_id]:
                if neuron_id < max_neurons:
                    matrix[i, neuron_id] = 1
    
    return matrix


def get_most_common_neurons(activation_matrix, n=100):
    """Get the most commonly activated neurons"""
    # Sum activations across samples
    neuron_counts = np.sum(activation_matrix, axis=0)
    
    # Get indices of top N neurons
    top_neurons = np.argsort(neuron_counts)[-n:][::-1]
    
    return top_neurons, neuron_counts[top_neurons]


def get_neuron_co_activations(activation_matrix, threshold=0.5):
    """Get neurons that tend to activate together"""
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(activation_matrix.T)
    
    # Find highly correlated pairs
    n_neurons = corr_matrix.shape[0]
    co_activations = []
    
    for i in range(n_neurons):
        for j in range(i+1, n_neurons):
            if corr_matrix[i, j] > threshold:
                co_activations.append((i, j, corr_matrix[i, j]))
    
    # Sort by correlation strength
    co_activations.sort(key=lambda x: x[2], reverse=True)
    
    return co_activations


def visualize_token_clusters(df, activation_matrix, output_dir, model_type, n_clusters=5):
    """Visualize clusters of tokens based on neuron activations"""
    # Apply dimensionality reduction
    if activation_matrix.shape[1] > 50:
        # First reduce with PCA if there are many dimensions
        pca = PCA(n_components=50)
        reduced_data = pca.fit_transform(activation_matrix)
    else:
        reduced_data = activation_matrix
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(reduced_data)
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_data)
    
    # Add results to dataframe
    plot_df = pd.DataFrame({
        'x': tsne_result[:, 0],
        'y': tsne_result[:, 1],
        'cluster': cluster_labels,
        'token': df['output'],
        'sample_id': df['sample_id'],
        'prompt_sample_idx': df['prompt_sample_idx'] if 'prompt_sample_idx' in df.columns else 0,
        'token_position': df['token_position'] if 'token_position' in df.columns else 0,
        'input_text': df['input_text']
    })
    
    # Plot
    plt.figure(figsize=(12, 10))
    scatter = sns.scatterplot(
        x='x', y='y', 
        hue='cluster', 
        data=plot_df,
        palette='viridis',
        s=100,
        alpha=0.7
    )
    
    # Add token labels
    for _, row in plot_df.iterrows():
        plt.annotate(
            row['token'],
            (row['x'], row['y']),
            fontsize=8,
            alpha=0.8
        )
    
    plt.title(f"{model_type} Token Clusters by Neuron Activation", fontsize=15)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{model_type}_token_clusters.png")
    plt.savefig(output_path)
    plt.close()
    
    # Create a new plot colored by prompt_sample_idx if available
    if 'prompt_sample_idx' in plot_df.columns:
        plt.figure(figsize=(12, 10))
        scatter = sns.scatterplot(
            x='x', y='y', 
            hue='prompt_sample_idx', 
            data=plot_df,
            palette='Set2',
            s=100,
            alpha=0.7
        )
        
        # Add token labels
        for _, row in plot_df.iterrows():
            plt.annotate(
                row['token'],
                (row['x'], row['y']),
                fontsize=8,
                alpha=0.8
            )
        
        plt.title(f"{model_type} Tokens by Sample Index", fontsize=15)
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, f"{model_type}_token_clusters_by_sample.png")
        plt.savefig(output_path)
        plt.close()
    
    # Save cluster data
    cluster_data_path = os.path.join(output_dir, f"{model_type}_token_clusters.csv")
    plot_df.to_csv(cluster_data_path, index=False)
    
    return plot_df


def visualize_neuron_importance(df, activations, output_dir, model_type, top_n=100):
    """Visualize the most important neurons across different tokens"""
    # Count how many times each neuron is activated
    neuron_counter = Counter()
    for sample_id in df['sample_id']:
        if sample_id in activations:
            for neuron_id in activations[sample_id]:
                neuron_counter[neuron_id] += 1
    
    # Get the top neurons
    top_neurons = neuron_counter.most_common(top_n)
    
    # Create dataframe for visualization
    top_df = pd.DataFrame(top_neurons, columns=['neuron_id', 'activation_count'])
    
    # Extract layer information
    top_df['layer'] = top_df['neuron_id'].apply(lambda x: calculate_layer(x, model_type))
    
    # Group by layer
    layer_counts = top_df.groupby('layer')['activation_count'].sum().reset_index()
    
    # Plot neuron importance by layer
    plt.figure(figsize=(12, 8))
    sns.barplot(x='layer', y='activation_count', data=layer_counts)
    plt.title(f"{model_type} Neuron Activations by Layer", fontsize=15)
    plt.xlabel("Layer")
    plt.ylabel("Total Activations")
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{model_type}_neuron_importance_by_layer.png")
    plt.savefig(output_path)
    plt.close()
    
    # Plot top neurons
    plt.figure(figsize=(15, 8))
    top_n_plot = min(50, len(top_df))  # Limit to top 50 for readability
    plot_data = top_df.head(top_n_plot)
    
    # Create a color map based on layer
    cmap = plt.cm.viridis
    unique_layers = sorted(plot_data['layer'].unique())
    layer_colors = {layer: cmap(i/len(unique_layers)) for i, layer in enumerate(unique_layers)}
    colors = [layer_colors[layer] for layer in plot_data['layer']]
    
    # Plot
    plt.bar(range(top_n_plot), plot_data['activation_count'], color=colors)
    plt.xticks(range(top_n_plot), plot_data['neuron_id'], rotation=90)
    plt.title(f"Top {top_n_plot} Most Activated Neurons", fontsize=15)
    plt.xlabel("Neuron ID")
    plt.ylabel("Activation Count")
    
    # Add layer legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=layer_colors[layer], label=f'Layer {layer}') 
                      for layer in unique_layers]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{model_type}_top_neurons.png")
    plt.savefig(output_path)
    plt.close()
    
    return top_df


def calculate_layer(neuron_id, model_type):
    """Calculate which layer a neuron belongs to based on its ID"""
    # Get model configuration
    if model_type == 'gpt2':
        n_layer, n_embd = 12, 768
    elif model_type == 'gpt2-medium':
        n_layer, n_embd = 24, 1024
    elif model_type == 'gpt2-large':
        n_layer, n_embd = 36, 1280
    elif model_type == 'gpt2-xl':
        n_layer, n_embd = 48, 1600
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Calculate GELU dimension
    gelu_dim = 4 * n_embd
    
    # Calculate layer
    layer = neuron_id // gelu_dim
    
    return layer


def analyze_token_specific_neurons(df, activations, output_dir, model_type):
    """Find neurons that activate specifically for certain tokens"""
    # Group by output token
    token_groups = df.groupby('output')
    
    # Collect activations by token
    token_activations = {}
    for token, group in token_groups:
        # Skip tokens that appear too rarely
        if len(group) < 3:
            continue
            
        # Collect all neuron activations for this token
        token_neurons = []
        for sample_id in group['sample_id']:
            if sample_id in activations:
                token_neurons.extend(activations[sample_id])
        
        # Count activations
        neuron_counts = Counter(token_neurons)
        # Keep only neurons that activate at least 50% of the time for this token
        threshold = len(group) * 0.5
        specific_neurons = {n: c for n, c in neuron_counts.items() if c >= threshold}
        
        token_activations[token] = specific_neurons
    
    # Create dataframe for visualization
    rows = []
    for token, specific_neurons in token_activations.items():
        if not specific_neurons:
            continue
            
        for neuron_id, count in specific_neurons.items():
            rows.append({
                'token': token,
                'neuron_id': neuron_id,
                'activation_count': count,
                'layer': calculate_layer(neuron_id, model_type)
            })
    
    if not rows:
        print("No token-specific neurons found with current threshold.")
        return None
        
    token_neuron_df = pd.DataFrame(rows)
    
    # Save dataframe
    output_path = os.path.join(output_dir, f"{model_type}_token_specific_neurons.csv")
    token_neuron_df.to_csv(output_path, index=False)
    
    # Visualize top tokens by number of specific neurons
    token_counts = token_neuron_df.groupby('token').size().reset_index(name='neuron_count')
    token_counts = token_counts.sort_values('neuron_count', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='token', y='neuron_count', data=token_counts)
    plt.title(f"{model_type} Number of Specific Neurons by Token", fontsize=15)
    plt.xlabel("Token")
    plt.ylabel("Number of Specific Neurons")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{model_type}_token_specific_neurons.png")
    plt.savefig(output_path)
    plt.close()
    
    return token_neuron_df


def compare_prompt_samples(df, activations, output_dir, model_type):
    """Compare activation patterns across different samples of the same prompts"""
    # Check if we have the required columns
    if 'prompt_sample_idx' not in df.columns or 'input_text' not in df.columns:
        print("Dataset doesn't contain sample index information. Skipping prompt sample comparison.")
        return None
    
    # Group by prompt and sample index
    prompt_samples = df.groupby(['input_text', 'prompt_sample_idx'])
    
    # Store Jaccard similarities between samples
    sample_similarities = []
    
    # Collect neurons by prompt and sample
    prompt_neurons = defaultdict(lambda: defaultdict(set))
    
    for (prompt, sample_idx), group in prompt_samples:
        # Get all activated neurons for this prompt sample
        sample_neurons = set()
        for sample_id in group['sample_id']:
            if sample_id in activations:
                sample_neurons.update(activations[sample_id])
        
        prompt_neurons[prompt][sample_idx] = sample_neurons
    
    # Calculate Jaccard similarities between samples of the same prompt
    for prompt, samples in prompt_neurons.items():
        sample_indices = sorted(samples.keys())
        
        for i, idx1 in enumerate(sample_indices):
            for j, idx2 in enumerate(sample_indices):
                if i >= j:  # Only calculate for unique pairs
                    continue
                
                set1 = samples[idx1]
                set2 = samples[idx2]
                
                # Calculate Jaccard similarity
                if not set1 or not set2:
                    continue
                    
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                similarity = intersection / union if union > 0 else 0
                
                sample_similarities.append({
                    'prompt': prompt,
                    'sample1': idx1,
                    'sample2': idx2,
                    'similarity': similarity,
                    'intersection': intersection,
                    'union': union
                })
    
    if not sample_similarities:
        print("No sample similarities calculated. Check if dataset has multiple samples per prompt.")
        return None
    
    # Create dataframe
    similarity_df = pd.DataFrame(sample_similarities)
    
    # Calculate average similarity per prompt
    avg_similarities = similarity_df.groupby('prompt')['similarity'].mean().reset_index()
    avg_similarities = avg_similarities.sort_values('similarity', ascending=False)
    
    # Visualize average similarities
    plt.figure(figsize=(12, 6))
    sns.barplot(x='similarity', y='prompt', data=avg_similarities)
    plt.title(f"{model_type} Average Neuron Activation Similarity Between Samples", fontsize=15)
    plt.xlabel("Jaccard Similarity")
    plt.ylabel("Prompt")
    plt.xlim(0, 1)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{model_type}_sample_similarities.png")
    plt.savefig(output_path)
    plt.close()
    
    # Save similarity data
    similarity_path = os.path.join(output_dir, f"{model_type}_sample_similarities.csv")
    similarity_df.to_csv(similarity_path, index=False)
    
    # Find consistent neurons that activate across all samples of a prompt
    consistent_neurons = []
    
    for prompt, samples in prompt_neurons.items():
        if len(samples) <= 1:
            continue
            
        # Get intersection of neurons that activate in all samples
        all_samples = list(samples.values())
        if not all_samples:
            continue
            
        common_neurons = set.intersection(*all_samples)
        
        for neuron_id in common_neurons:
            consistent_neurons.append({
                'prompt': prompt,
                'neuron_id': neuron_id,
                'layer': calculate_layer(neuron_id, model_type),
                'num_samples': len(samples)
            })
    
    if consistent_neurons:
        consistent_df = pd.DataFrame(consistent_neurons)
        consistent_path = os.path.join(output_dir, f"{model_type}_consistent_neurons.csv")
        consistent_df.to_csv(consistent_path, index=False)
        
        # Visualize consistent neurons by layer
        plt.figure(figsize=(12, 6))
        sns.countplot(x='layer', data=consistent_df)
        plt.title(f"{model_type} Consistently Activated Neurons by Layer", fontsize=15)
        plt.xlabel("Layer")
        plt.ylabel("Count")
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, f"{model_type}_consistent_neurons_by_layer.png")
        plt.savefig(output_path)
        plt.close()
    
    return similarity_df


def analyze_position_specific_neurons(df, activations, output_dir, model_type):
    """Analyze neurons that activate at specific positions in the sequence"""
    # Check if we have position information
    if 'token_position' not in df.columns:
        print("Dataset doesn't contain token position information. Skipping position analysis.")
        return None
    
    # Group by token position
    position_groups = df.groupby('token_position')
    
    # Collect neurons that activate at each position
    position_neurons = {}
    for position, group in position_groups:
        # Skip positions with too few samples
        if len(group) < 5:
            continue
            
        # Collect all neuron activations for this position
        position_neuron_list = []
        for sample_id in group['sample_id']:
            if sample_id in activations:
                position_neuron_list.extend(activations[sample_id])
        
        # Count activations
        neuron_counts = Counter(position_neuron_list)
        
        # Keep only neurons that activate at least 30% of the time at this position
        threshold = len(group) * 0.3
        specific_neurons = {n: c for n, c in neuron_counts.items() if c >= threshold}
        
        position_neurons[position] = specific_neurons
    
    # Create dataframe for visualization
    rows = []
    for position, specific_neurons in position_neurons.items():
        if not specific_neurons:
            continue
            
        for neuron_id, count in specific_neurons.items():
            rows.append({
                'position': position,
                'neuron_id': neuron_id,
                'activation_count': count,
                'layer': calculate_layer(neuron_id, model_type)
            })
    
    if not rows:
        print("No position-specific neurons found with current threshold.")
        return None
        
    position_df = pd.DataFrame(rows)
    
    # Save dataframe
    output_path = os.path.join(output_dir, f"{model_type}_position_specific_neurons.csv")
    position_df.to_csv(output_path, index=False)
    
    # Plot neuron count by position
    position_counts = position_df.groupby('position').size().reset_index(name='neuron_count')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='position', y='neuron_count', data=position_counts)
    plt.title(f"{model_type} Number of Position-Specific Neurons", fontsize=15)
    plt.xlabel("Token Position")
    plt.ylabel("Number of Specific Neurons")
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{model_type}_position_specific_neurons.png")
    plt.savefig(output_path)
    plt.close()
    
    # Plot heatmap of position-specific neurons by layer
    position_layer = position_df.groupby(['position', 'layer']).size().reset_index(name='count')
    position_layer_matrix = position_layer.pivot(index='position', columns='layer', values='count').fillna(0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(position_layer_matrix, cmap='viridis', annot=True, fmt='g')
    plt.title(f"{model_type} Position-Specific Neurons by Layer", fontsize=15)
    plt.xlabel("Layer")
    plt.ylabel("Token Position")
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{model_type}_position_layer_heatmap.png")
    plt.savefig(output_path)
    plt.close()
    
    return position_df


def main():
    # python analyze_activations.py --dataset_csv parcellation_datasets/test_1/gpt2_activation_dataset.csv --activations_file parcellation_datasets/test_1/gpt2_neuron_activations.json --output_dir parcellation_datasets/test_1/analysis
    parser = argparse.ArgumentParser(description="Analyze GELU neuron activations")
    parser.add_argument('--dataset_csv', required=True,
                        help='Path to the activation dataset CSV')
    parser.add_argument('--activations_file', required=True,
                        help='Path to the activations file (.pickle or .json)')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save analysis outputs')
    parser.add_argument('--model_type', default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='GPT-2 model size')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset and activations
    print(f"Loading dataset from {args.dataset_csv}")
    df, activations = load_dataset(args.dataset_csv, args.activations_file)
    
    print(f"Loaded {len(df)} samples with {len(activations)} activation records")
    
    # Create activation matrix
    print("Creating activation matrix...")
    activation_matrix = create_activation_matrix(df, activations)
    
    # Run analyses
    print("Running analyses...")
    
    # 1. Visualize token clusters
    print("Visualizing token clusters...")
    visualize_token_clusters(df, activation_matrix, args.output_dir, args.model_type)
    
    # 2. Analyze neuron importance
    print("Analyzing neuron importance...")
    visualize_neuron_importance(df, activations, args.output_dir, args.model_type)
    
    # 3. Find token-specific neurons
    print("Finding token-specific neurons...")
    analyze_token_specific_neurons(df, activations, args.output_dir, args.model_type)
    
    # 4. Compare samples of the same prompt (if applicable)
    print("Comparing samples of the same prompt...")
    compare_prompt_samples(df, activations, args.output_dir, args.model_type)
    
    # 5. Analyze position-specific neurons (if applicable)
    print("Analyzing position-specific neurons...")
    analyze_position_specific_neurons(df, activations, args.output_dir, args.model_type)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 