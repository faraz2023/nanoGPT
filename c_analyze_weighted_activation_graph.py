#!/usr/bin/env python
# python c_analyze_weighted_activation_graph.py --input_path parcellation_datasets/test_1/gpt2_weighted_graph.gpickle 
# python c_analyze_weighted_activation_graph.py --input_path parcellation_datasets/test_1/gpt2_weighted_graph.gpickle --output_dir my_analysis_results
"""
Analyzes a weighted activation graph generated by c_calculate_node_weighted_activation_graph.py

Performs the following analysis:
1. Calculates and adds normalized node and edge weights.
2. Generates histograms for node weights, edge weights, and node degrees.
3. Performs thresholding analysis on normalized edge weights to study connected components.
4. Saves all results into a specified output directory.

Example usage:
python c_analyze_weighted_activation_graph.py --input_path parcellation_datasets/test_1/gpt2_weighted_graph.gpickle 
# Output will be saved in parcellation_datasets/test_1/weighted_graph_analysis/

python c_analyze_weighted_activation_graph.py --input_path parcellation_datasets/test_1/gpt2_weighted_graph.gpickle --output_dir analysis_results/test_1_analysis
"""

import os
import argparse
import pickle
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random # Import random for sampling
import gc # Import garbage collector

def normalize_dict_values(data_dict):
    """Normalize dictionary values to the range [0, 1]."""
    if not data_dict:
        return {}
    
    values = np.array(list(data_dict.values()))
    min_val = np.min(values)
    max_val = np.max(values)
    
    if max_val == min_val:
        # Handle cases where all values are the same (or only one value)
        # Normalize to 0 if min/max is 0, otherwise normalize to 1 or 0.5
        norm_val = 0.0 if max_val == 0 else 0.5 # Or 1.0 depending on desired behavior for constant values
        return {k: norm_val for k in data_dict}
    
    normalized_dict = {k: (v - min_val) / (max_val - min_val) for k, v in data_dict.items()}
    return normalized_dict

def plot_histogram(data, title, xlabel, filename, bins=50, density=True):
    """Generates and saves a normalized histogram with percentile lines."""
    if not data or len(data) == 0: # Check if data is empty
        print(f"No data provided or empty data for histogram: {title}")
        return
        
    plt.figure(figsize=(12, 7)) # Slightly larger figure for legend
    counts, bin_edges, patches = plt.hist(data, bins=bins, density=density, alpha=0.75, edgecolor='black', label=f'{xlabel} Density')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density' if density else 'Frequency')
    plt.grid(axis='y', alpha=0.5)

    # Calculate and plot percentiles
    try:
        p25, p50, p75 = np.percentile(data, [25, 50, 75])
        plt.axvline(p25, color='green', linestyle='dashed', linewidth=1.5, label=f'25th Percentile ({p25:.2e})')
        plt.axvline(p50, color='gold', linestyle='dashed', linewidth=1.5, label=f'50th Percentile ({p50:.2e})') # Median
        plt.axvline(p75, color='red', linestyle='dashed', linewidth=1.5, label=f'75th Percentile ({p75:.2e})')
        plt.legend()
    except IndexError:
        print(f"Could not calculate percentiles for {title}, possibly due to insufficient data points after filtering.")
    except Exception as e:
         print(f"Error calculating/plotting percentiles for {title}: {e}")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved histogram: {filename}")
    
    # Also save histogram data
    hist_data_filename = filename.replace('.png', '.csv')
    # Ensure counts and bin_edges match length if possible
    valid_bins = min(len(bin_edges) -1, len(counts))
    hist_df = pd.DataFrame({
        'bin_start': bin_edges[:valid_bins],
        'bin_end': bin_edges[1:valid_bins+1],
        'density': counts[:valid_bins]
        })
    hist_df.to_csv(hist_data_filename, index=False)
    print(f"Saved histogram data: {hist_data_filename}")

def main():
    parser = argparse.ArgumentParser(description="Analyze weighted activation graph")
    parser.add_argument('--input_path', required=True,
                        help='Path to the weighted graph .gpickle file')
    parser.add_argument('--output_dir', default=None,
                        help='Directory to save analysis results. Defaults to a subdir named weighted_graph_analysis/')
    parser.add_argument('--hist_sample_size', type=int, default=1000000, # Add argument for sample size
                        help='Number of edges to sample for edge weight histogram (default: 1,000,000)')
    args = parser.parse_args()

    # --- Input Validation and Setup ---
    if not os.path.exists(args.input_path):
        print(f"Error: Input graph file not found: {args.input_path}")
        return

    if args.output_dir:
        output_dir = args.output_dir
    else:
        input_dir = os.path.dirname(args.input_path)
        output_dir = os.path.join(input_dir, 'weighted_graph_analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Analysis results will be saved in: {output_dir}")

    # --- Load Graph ---    
    print(f"Loading graph from {args.input_path}...")
    try:
        with open(args.input_path, 'rb') as f:
            G = pickle.load(f)
    except Exception as e:
        print(f"Error loading graph file: {e}")
        return
    
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # --- Remove Edges with Zero or Negative Co-activation --- 
    print("Removing edges with non-positive co-activation count...")
    invalid_edges = [
        (u, v) for u, v, data in G.edges(data=True) 
        if data.get('co_activation_count', 0) <= 0 # Changed to <= 0
    ]
    
    if invalid_edges:
        G.remove_edges_from(invalid_edges)
        print(f"Removed {len(invalid_edges)} edges with non-positive co-activation count.")
        print(f"Graph now has {G.number_of_edges()} edges with positive co-activation count.")
        # Optional: Explicit GC after potentially large list creation and edge removal
        del invalid_edges 
        gc.collect()
    else:
        print("No edges with non-positive co-activation count found.")

    # --- Normalize Weights --- 
    print("Calculating normalized weights...")
    # Node weights
    node_weights = nx.get_node_attributes(G, 'activation_count')
    if node_weights:
        max_node_weight = max(node_weights.values()) if node_weights else 0
        if max_node_weight > 0:
            node_weights_normalized = {n: w / max_node_weight for n, w in node_weights.items()}
        else:
             node_weights_normalized = {n: 0.0 for n in node_weights} # All zero if max is zero
        nx.set_node_attributes(G, node_weights_normalized, 'activation_count_normalized')
        print(f"Max node activation_count: {max_node_weight}")
        # Optional: Clean up large dict if memory is extremely tight
        # del node_weights_normalized
        # gc.collect()
    else:
        print("Warning: No 'activation_count' attribute found on nodes.")
        node_weights_normalized = {}

    # Edge weights - More memory-efficient approach
    print("Calculating max edge weight (post-filtering)...") # Added post-filtering note
    max_edge_weight = 0
    edge_iterator = iter(G.edges(data=True)) # Convert view to iterator
    try:
        # Use try-except to handle potentially empty graph
        first_edge = next(edge_iterator)
        # Ensure weight is accessed correctly and is positive
        max_edge_weight = max(0, first_edge[2].get('co_activation_count', 0)) 
        for u, v, data in edge_iterator:
            weight = max(0, data.get('co_activation_count', 0)) # Clamp potential negatives
            if weight > max_edge_weight:
                max_edge_weight = weight
        # Add the first edge's weight back into comparison if it wasn't the only edge
        if G.number_of_edges() > 1:
             weight = max(0, first_edge[2].get('co_activation_count', 0))
             if weight > max_edge_weight:
                 max_edge_weight = weight
        
    except StopIteration: # handles case G has no edges
        print("Graph has no edges after filtering.")
        max_edge_weight = 0
        
    print(f"Max edge co_activation_count (post-filtering): {max_edge_weight}")

    # Set normalized edge weights using np.float32
    print("Setting normalized edge weights (float32)...")
    if max_edge_weight > 0:
        for u, v, data in G.edges(data=True):
             # Ensure original weight is non-negative before normalizing
            original_weight = max(0, data.get('co_activation_count', 0)) 
            normalized_weight = np.float32(original_weight / max_edge_weight)
            G[u][v]['co_activation_count_normalized'] = normalized_weight
    else:
         # Set all to 0.0 if max_edge_weight is 0 (or graph has no edges)
         # Check if attribute exists before assigning
         if G.number_of_edges() > 0:
             first_edge_data = next(iter(G.edges(data=True)))[2]
             if 'co_activation_count_normalized' not in first_edge_data:
                 # Initialize if not present (should be handled by calc above, but defensive)
                 for u_init, v_init in G.edges():
                    G[u_init][v_init]['co_activation_count_normalized'] = np.float32(0.0)
             else: # Already exists, just set
                 for u_set, v_set in G.edges():
                    G[u_set][v_set]['co_activation_count_normalized'] = np.float32(0.0)
         # If no edges, do nothing
            
    # --- Generate Histograms --- 
    print("Generating histograms...")
    # Node activation count histogram
    plot_histogram(
        list(node_weights.values()), 
        'Node Activation Count Distribution',
        'Activation Count',
        os.path.join(output_dir, 'node_activation_count_hist.png')
    )
    
    # Edge co-activation count histogram (Sampled)
    print(f"Sampling up to {args.hist_sample_size} edges for histogram...")
    num_edges = G.number_of_edges()
    edge_weights_sample = []
    negative_weights_found_warning = False # Flag for warning
    if num_edges > 0:
        sample_size = min(num_edges, args.hist_sample_size)
        if sample_size < num_edges:
             print(f"Warning: Sampling {sample_size} out of {num_edges} edges for histogram.")
             # Efficiently sample edges if graph is large
             # Convert G.edges to a list ONLY for sampling if necessary and feasible
             try:
                 all_edges = list(G.edges(data=True))
                 sampled_edges = random.sample(all_edges, sample_size)
                 for _, _, data in sampled_edges:
                     weight = data.get('co_activation_count', 0)
                     if weight < 0:
                         if not negative_weights_found_warning:
                             print("Warning: Negative co-activation counts detected in data, clamping to 0 for histogram.")
                             negative_weights_found_warning = True
                         weight = 0
                     edge_weights_sample.append(weight)
                 del all_edges # Clean up large list
                 del sampled_edges
                 gc.collect() 
             except MemoryError:
                  print("MemoryError during edge sampling preparation. Trying iterative sampling...")
                  # Fallback iterative sampling (might be slow, less random)
                  step = max(1, int(num_edges / sample_size)) # Ensure step is int
                  count = 0
                  for i, (_, _, data) in enumerate(G.edges(data=True)):
                      if i % step == 0 and count < sample_size:
                          weight = data.get('co_activation_count', 0)
                          if weight < 0:
                              if not negative_weights_found_warning:
                                  print("Warning: Negative co-activation counts detected during iterative sampling, clamping to 0 for histogram.")
                                  negative_weights_found_warning = True
                              weight = 0
                          edge_weights_sample.append(weight)
                          count += 1
                  print(f"Iterative sampling collected {count} weights.")
        else:
             # If sample size >= num_edges, just take all weights (if feasible)
             print("Sample size >= number of edges. Using all edge weights for histogram.")
             temp_weights_list = []
             try:
                 for _, _, data in G.edges(data=True):
                      weight = data.get('co_activation_count', 0)
                      if weight < 0:
                          if not negative_weights_found_warning:
                              print("Warning: Negative co-activation counts detected in data, clamping to 0 for histogram.")
                              negative_weights_found_warning = True
                          weight = 0
                      temp_weights_list.append(weight)
                 edge_weights_sample = temp_weights_list
                 del temp_weights_list # Clean up
                 gc.collect()
             except MemoryError:
                 print("MemoryError even when attempting to collect all edge weights. Histogram will be empty.")
                 edge_weights_sample = [] # Fallback to empty if memory fails
                 del temp_weights_list # Clean up
                 gc.collect()

    plot_histogram(
        edge_weights_sample, # Use sampled list (with negatives clamped)
        f'Edge Co-Activation Count Distribution (Sampled {len(edge_weights_sample)} edges)',
        'Co-Activation Count',
        os.path.join(output_dir, 'edge_co_activation_count_hist.png')
    )
    # Clean up sample list
    del edge_weights_sample
    gc.collect()
    
    # Node degree histogram
    print("Calculating node degrees...")
    node_degrees = [d for n, d in G.degree()]
    print("Plotting node degree histogram...")
    plot_histogram(
        node_degrees,
        'Node Degree Distribution',
        'Degree',
        os.path.join(output_dir, 'node_degree_hist.png')
    )
    del node_degrees # Clean up degree list
    gc.collect()
    
    # --- Save Graph with Normalized Attributes (BEFORE Thresholding) --- 
    updated_graph_path = os.path.join(output_dir, os.path.basename(args.input_path).replace('.gpickle', '_analyzed.gpickle'))
    print(f"Saving graph with normalized attributes (before thresholding) to {updated_graph_path}...")
    # try:
    #     with open(updated_graph_path, 'wb') as f:
    #         pickle.dump(G, f)
    #     print("Graph saved successfully.")
    # except Exception as e:
    #     print(f"Error saving graph file before thresholding: {e}")
    #     # Optionally, decide if you want to proceed without saving
    #     # return 

    # --- Thresholding Analysis (Modifies G directly)--- 
    print("\nPerforming thresholding analysis (modifying graph G directly)...")
    analysis_results = []

    # Check for edges and the normalized attribute more carefully
    has_normalized_edges = False
    if G.number_of_edges() > 0:
        try:
             first_edge_data = next(iter(G.edges(data=True)))[2]
             if 'co_activation_count_normalized' in first_edge_data:
                  has_normalized_edges = True
        except StopIteration:
             pass # No edges

    if not has_normalized_edges:
         print("Skipping thresholding analysis as normalized edge weights could not be calculated or found.")
    else:
        # --- Pre-calculate t=0.0 case (using original graph G) --- 
        print("\nAnalyzing threshold > 0.00 (using original filtered graph G)...")
        t0_num_nodes = G.number_of_nodes()
        t0_num_edges = G.number_of_edges()
        t0_num_components = np.nan
        t0_max_comp_size = np.nan
        t0_avg_comp_size = np.nan
        t0_min_comp_size = np.nan
        t0_error_message = None
        
        if t0_num_nodes > 0:
            try:
                print(f"  Calculating connected components for threshold 0.00...")
                # Directly use G for t=0.0
                components_t0 = list(nx.connected_components(G)) 
                print(f"  Found {len(components_t0)} components.")
                t0_num_components = len(components_t0)
                comp_sizes_t0 = [len(c) for c in components_t0]
                t0_max_comp_size = max(comp_sizes_t0) if comp_sizes_t0 else 0
                t0_min_comp_size = min(comp_sizes_t0) if comp_sizes_t0 else 0
                t0_avg_comp_size = np.mean(comp_sizes_t0) if comp_sizes_t0 else 0.0
                del components_t0 # Clean up component list
                del comp_sizes_t0
                gc.collect()
            except MemoryError:
                 print(f"  MemoryError occurred during component calculation for threshold 0.00. Skipping component stats.")
                 t0_error_message = "MemoryError"
                 # Attempt to clean up potential large intermediate objects within networkx
                 gc.collect()
            except Exception as e:
                 print(f"  An unexpected error occurred for threshold 0.00: {e}. Skipping component stats.")
                 t0_error_message = str(e)
                 gc.collect()
        else:
             t0_num_components = 0
             t0_max_comp_size = 0
             t0_min_comp_size = 0
             t0_avg_comp_size = 0.0
             
        analysis_results.append({
            'threshold': 0.00,
            'num_nodes': t0_num_nodes,
            'num_edges': t0_num_edges,
            'num_components': t0_num_components,
            'max_comp_size': t0_max_comp_size,
            'avg_comp_size': t0_avg_comp_size,
            'min_comp_size': t0_min_comp_size,
            'error': t0_error_message
        })
        
        # --- Loop for t > 0.0 using iterative removal on G --- 
        # No need to create G_working. Operate directly on G.
        thresholds = np.arange(0.1, 1.0, 0.1) # Start from 0.1
        for t in thresholds:
            print(f"\nAnalyzing threshold > {t:.2f} (Iterative Removal on G)...")
            num_components = np.nan
            max_comp_size = np.nan
            avg_comp_size = np.nan
            min_comp_size = np.nan
            error_message = None
            edges_to_remove = []
            components = None
            comp_sizes = None

            current_edge_count = G.number_of_edges() # Use G
            print(f"  Graph G currently has {current_edge_count} edges.")

            try:
                # Identify edges to remove for this threshold from G
                print(f"  Identifying edges with weight <= {t:.2f} to remove from G...")
                edges_to_remove = [
                    (u, v) for u, v, data in G.edges(data=True) # Use G
                    if data.get('co_activation_count_normalized', 0.0) <= t
                ]
                
                # Remove the identified edges from G
                if edges_to_remove:
                    print(f"  Removing {len(edges_to_remove)} edges from G...")
                    G.remove_edges_from(edges_to_remove) # Modify G
                    print(f"  Graph G now has {G.number_of_edges()} edges.")
                else:
                    print("  No edges to remove at this threshold.")

                # Node count remains constant
                num_nodes_in_subgraph = G.number_of_nodes() # Use G
                num_edges_in_subgraph = G.number_of_edges() # Use G
                
                # Calculate components on the updated G
                if num_nodes_in_subgraph == 0:
                    num_components = 0
                else:
                    print(f"  Calculating connected components for threshold {t:.2f} on G...")
                    components = list(nx.connected_components(G)) # Use G
                    print(f"  Found {len(components)} components.")
                    num_components = len(components)
                    comp_sizes = [len(c) for c in components]
                    max_comp_size = max(comp_sizes) if comp_sizes else 0
                    min_comp_size = min(comp_sizes) if comp_sizes else 0
                    avg_comp_size = np.mean(comp_sizes) if comp_sizes else 0.0

            except MemoryError:
                 print(f"  MemoryError occurred during processing for threshold {t:.2f}. Skipping component stats.")
                 error_message = "MemoryError"
            except Exception as e:
                 print(f"  An unexpected error occurred for threshold {t:.2f}: {e}. Skipping component stats.")
                 error_message = str(e)
            finally:
                # Explicitly delete large objects and collect garbage
                print(f"  Cleaning up memory for threshold {t:.2f}...")
                del edges_to_remove # Delete list of edges
                del components
                del comp_sizes
                gc.collect()

            analysis_results.append({
                'threshold': round(t, 2),
                'num_nodes': num_nodes_in_subgraph,
                'num_edges': num_edges_in_subgraph,
                'num_components': num_components,
                'max_comp_size': max_comp_size,
                'avg_comp_size': avg_comp_size,
                'min_comp_size': min_comp_size,
                'error': error_message
            })
            
        # No need to clean up G_working as it wasn't created

        # Save thresholding results to CSV
        analysis_df = pd.DataFrame(analysis_results)
        csv_path = os.path.join(output_dir, 'threshold_analysis.csv')
        analysis_df.to_csv(csv_path, index=False)
        print(f"\nThreshold analysis saved to: {csv_path}")

    print("Analysis complete.")

if __name__ == "__main__":
    main() 