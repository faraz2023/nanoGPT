#!/usr/bin/env python

"""
Selects a subset of influential neurons from an activation graph based on degree centrality 
after applying an edge weight threshold.

Loads a graph (ideally the '_analyzed.gpickle' output with normalized weights),
filters edges based on a threshold applied to 'co_activation_count_normalized',
calculates the degree of nodes in the resulting unweighted graph, and saves the IDs
of the top k nodes (determined by a ratio 'r') to a NumPy file.

Example usage:

# works
python d_calculate_perturbation_subset.py \
    --input_graph_path parcellation_datasets/test_2/weighted_graph_analysis/gpt2_weighted_graph_analyzed.gpickle \
    --threshold 0.1 \
    --ratio 0.05 \
    --method degree \
    --output_path parcellation_datasets/test_2/perturbation_subsets/gpt2_degree_t0.1_r0.05.npy

python d_calculate_perturbation_subset.py \
    --input_graph_path parcellation_datasets/test_2/weighted_graph_analysis/gpt2_weighted_graph_analyzed.gpickle \
    --threshold 0.8 \
    --ratio 0.01 \
    --method degree \
    --output_path parcellation_datasets/test_2/perturbation_subsets/gpt2_degree_t0.8_r0.01.npy

python d_calculate_perturbation_subset.py \
    --input_graph_path parcellation_datasets/test_2/weighted_graph_analysis/gpt2_weighted_graph_analyzed.gpickle \
    --threshold 0.8 \
    --ratio 0.02 \
    --method degree \
    --output_path parcellation_datasets/test_2/perturbation_subsets/gpt2_degree_t0.8_r0.01.npy


"""

import os
import argparse
import pickle
import networkx as nx
import numpy as np
import gc

def influential_neuron_detection(graph, k, method):
    """Selects the top k influential neurons based on the specified method."""
    
    if method == 'degree':
        print(f"  Calculating node degrees for {graph.number_of_nodes()} nodes...")
        # degrees is an iterator of (node, degree) pairs
        degrees = graph.degree()
        
        # Sort by degree (value) in descending order
        # Convert degrees iterator to list for sorting
        try:
            sorted_nodes = sorted(degrees, key=lambda item: item[1], reverse=True)
            print(f"  Sorted {len(sorted_nodes)} nodes by degree.")
        except Exception as e:
            print(f"Error sorting node degrees: {e}")
            return np.array([], dtype=np.int32) # Return empty array on error

        # Select the top k node IDs
        top_k_nodes = [node_id for node_id, degree in sorted_nodes[:k]]
        print(f"  Selected top {len(top_k_nodes)} nodes.")
        
        # Convert to numpy array with specified dtype
        result_array = np.array(top_k_nodes, dtype=np.int32)
        
        # Clean up intermediate list
        del sorted_nodes
        del top_k_nodes
        gc.collect()
        
        return result_array
        
    # Add other methods here in the future
    # elif method == 'pagerank':
    #     # ... implementation ...
    #     pass
    else:
        raise ValueError(f"Unsupported influential neuron detection method: {method}")

def main():
    parser = argparse.ArgumentParser(description="Calculate perturbation subset based on thresholded graph centrality")
    parser.add_argument('--input_graph_path', required=True,
                        help='Path to the analyzed graph .gpickle file (e.g., _analyzed.gpickle)')
    parser.add_argument('--threshold', type=float, required=True,
                        help="Threshold for normalized edge weight ('co_activation_count_normalized')")
    parser.add_argument('--ratio', type=float, required=True,
                        help='Ratio of total nodes to select (k = num_nodes * ratio)')
    parser.add_argument('--method', default='degree', choices=['degree'], # Add more choices as methods are implemented
                        help='Method for selecting influential neurons')
    parser.add_argument('--output_path', required=True,
                        help='Path to save the output NumPy array (.npy) of selected node IDs')
    args = parser.parse_args()

    # --- Input Validation and Setup --- 
    if not os.path.exists(args.input_graph_path):
        print(f"Error: Input graph file not found: {args.input_graph_path}")
        return
        
    if not 0.0 <= args.threshold <= 1.0:
         print(f"Warning: Threshold {args.threshold} is outside the typical normalized range [0, 1].")
         
    if not 0.0 < args.ratio <= 1.0:
         print(f"Error: Ratio {args.ratio} must be between 0 (exclusive) and 1 (inclusive).")
         return

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # --- Load Graph ---    
    print(f"Loading graph from {args.input_graph_path}...")
    try:
        with open(args.input_graph_path, 'rb') as f:
            G = pickle.load(f)
        print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    except Exception as e:
        print(f"Error loading graph file: {e}")
        print("Full stack trace:")
        import traceback
        traceback.print_exc()
        return
        
    # Check if normalized weights exist
    if G.number_of_edges() > 0:
        try:
            first_edge_data = next(iter(G.edges(data=True)))[2]
            if 'co_activation_count_normalized' not in first_edge_data:
                print(f"Error: Edge attribute 'co_activation_count_normalized' not found in graph.")
                print("Please ensure you are using the '_analyzed.gpickle' file generated by c_analyze_weighted_activation_graph.py")
                return
        except StopIteration:
             pass # Graph has no edges, will proceed but G_threshold will also have no edges
    elif G.number_of_nodes() > 0: 
         print("Warning: Graph has nodes but no edges. Proceeding, but thresholding will have no effect.")
    else:
         print("Graph has no nodes or edges.")

    # --- Create Thresholded Graph --- 
    print(f"Creating thresholded graph G_threshold (edge weight > {args.threshold})...")
    G_threshold = nx.Graph() 
    # Add all nodes first to preserve node set, even if they become isolated
    G_threshold.add_nodes_from(G.nodes())
    
    edges_added = 0
    edges_processed = 0
    # Iterate through edges of the original graph and add only those above threshold
    for u, v, data in G.edges(data=True):
        edges_processed += 1
        norm_weight = data.get('co_activation_count_normalized', 0.0)
        if norm_weight > args.threshold:
            G_threshold.add_edge(u, v) # Add edge (unweighted in the new graph)
            edges_added += 1
        if edges_processed % 1000000 == 0: # Progress update for large graphs
             print(f"  Processed {edges_processed} / {G.number_of_edges()} original edges...")

    print(f"Created G_threshold with {G_threshold.number_of_nodes()} nodes and {edges_added} edges.")

    # Optional: Delete original graph G if memory is critical
    # print("Deleting original graph G to save memory...")
    # del G
    # gc.collect()

    # --- Calculate k --- 
    num_nodes = G_threshold.number_of_nodes()
    if num_nodes == 0:
        print("Error: Thresholded graph has no nodes. Cannot select influential neurons.")
        # Save empty array
        influential_nodes_array = np.array([], dtype=np.int32)
    else:
        k = int(num_nodes * args.ratio)
        # Ensure k is at least 1 if ratio is small but > 0 and there are nodes
        k = max(1, k) if args.ratio > 0 else k 
        # Ensure k does not exceed the number of nodes
        k = min(k, num_nodes)
        print(f"Calculated k = {k} (ratio {args.ratio} * {num_nodes} nodes)")

        # --- Select Influential Neurons --- 
        print(f"Selecting top {k} influential neurons using method '{args.method}'...")
        influential_nodes_array = influential_neuron_detection(G_threshold, k, args.method)

    # --- Save Output --- 
    print(f"Saving {len(influential_nodes_array)} selected node IDs to {args.output_path}...")
    try:
        np.save(args.output_path, influential_nodes_array)
        print("Output saved successfully.")
    except Exception as e:
        print(f"Error saving output NumPy file: {e}")
        
    print("Script finished.")

if __name__ == "__main__":
    main() 