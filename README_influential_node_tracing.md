# Influential Neuron Tracing Pipeline Documentation

This document provides an overview and detailed explanation of the Python scripts used in the pipeline for identifying and analyzing influential GELU neurons within a GPT model based on activation patterns.

## Pipeline Overview

The pipeline consists of several scripts that perform the following steps:

1.  **Generate Base Graph Structure:** (`a_generate_gelu_graph.py`) Creates the basic graph structure representing potential connections between GELU neurons in adjacent layers of a specified GPT model.
2.  **Generate Activation Data:** (`b_generate_activation_dataset.py`) Runs multiple prompts through the GPT model, records which GELU neurons activate for each generated token, and calculates activation frequencies and co-activation statistics.
3.  **(Alternative Activation Generation):** (`b_trace_gelu_activations.py`) Provides a simpler way to trace and save the activation vector for a *single* input text, primarily for debugging or specific analysis.
4.  **Calculate Weighted Graph:** (`c_calculate_weighted_activation_graph.py`) Combines the base graph structure with the activation data to create a weighted graph. Node weights represent total activation count, and edge weights represent co-activation count between connected neurons.
5.  **Analyze Weighted Graph:** (`c_analyze_weighted_activation_graph.py`) Performs further analysis on the weighted graph, including calculating normalized weights, generating histograms of weights and degrees, and performing thresholding analysis to study connected components. It saves an "analyzed" version of the graph.
6.  **Select Perturbation Subset:** (`d_calculate_perturbation_subset.py`) Filters the analyzed graph based on a normalized edge weight threshold and selects a subset of the most "influential" neurons based on a specified centrality metric (e.g., degree) and ratio.
7.  **Generate Perturbed Results:** (`e_generate_perturbed_results.py`) Runs text generation experiments using the base GPT model while perturbing (zeroing out) the activations of specific GELU neurons identified as influential by previous steps. This allows evaluating the impact of these neurons on model outputs.

## Script Details

### 1. `a_generate_gelu_graph.py`

*   **Purpose:** Generates the foundational graph structure for the GELU neurons of a specified GPT model (`gpt2`, `gpt2-medium`, etc.). It assumes full connectivity between GELU neurons in adjacent MLP layers.
*   **Functionality:**
    *   Takes the model type (`--model_type`) as input.
    *   Calculates the number of layers and the dimension of the GELU activation space (4 * embedding dimension).
    *   Generates unique node IDs for each GELU neuron across all layers.
    *   Creates edges connecting every neuron in layer `i` to every neuron in layer `i+1`.
*   **Inputs:**
    *   `--model_type`: GPT model size (e.g., 'gpt2').
    *   `--output_dir`: Directory to save the output files.
*   **Outputs:**
    *   `{model_type}_neuron_attributes.csv`: CSV file listing each neuron's ID, layer index, and dimension index within the layer.
    *   `{model_type}_neuron_edges.el`: Edge list file defining the connections between neurons in adjacent layers (source\_node\_id target\_node\_id).
*   **Example Usage:**
    ```bash
    python a_generate_gelu_graph.py --model_type gpt2 --output_dir gelu_graph
    ```

### 2. `b_generate_activation_dataset.py`

*   **Purpose:** Generates a rich dataset of GELU neuron activations by running a specified GPT model on a set of input prompts. It captures which neurons are active (> 0) after the GELU function for each token generated.
*   **Functionality:**
    *   Loads a specified GPT model (`--model_type`) and tokenizer.
    *   Registers forward hooks on the GELU activation function in each MLP block.
    *   Reads prompts from an input CSV (`--input_csv`), which should specify the prompt text, the number of new tokens to generate (`number_of_new_tokens`), and optionally the number of samples per prompt (`num_samples`).
    *   For each prompt and sample, generates tokens one by one.
    *   For each generated token, records the binary activation state (0 or 1) of every GELU neuron.
    *   Stores the list of active neuron IDs for each generated token sample.
    *   Calculates overall neuron activation frequency and co-activation statistics.
    *   Re-generates the base graph structure (attributes and edges) similar to `a_generate_gelu_graph.py` within the specified export directory.
*   **Inputs:**
    *   `--input_csv`: Path to the CSV file containing prompts.
    *   `--export_dir`: Directory to save all generated dataset files.
    *   `--model_type`: GPT model size.
    *   `--seed`: Random seed for reproducibility.
    *   `--temperature`: Temperature for token sampling during generation.
*   **Outputs (within `--export_dir`):**
    *   `{model_type}_neuron_attributes.csv`: Node attributes (ID, layer, dimension).
    *   `{model_type}_neuron_edges.el`: Base edge list.
    *   `{model_type}_activation_dataset.csv`: Detailed log for each generated token (sample ID, prompt, output token, active neuron counts, etc.).
    *   `{model_type}_neuron_activations.pickle`: Dictionary mapping sample ID to a list of active neuron IDs (Python pickle format).
    *   `{model_type}_neuron_activations.json`: Same as pickle, but JSON format (potentially larger).
    *   `{model_type}_neuron_frequency.csv`: Statistics for each neuron (ID, layer, dimension, total activation count, activation frequency across samples).
    *   `{model_type}_activation_matrix.npz`: Compressed NumPy file containing:
        *   `matrix`: A binary matrix (samples x active\_neurons) where `1` indicates activation.
        *   `neuron_ids`: An array mapping the columns of the matrix back to the original neuron IDs.
        *   `neuron_map`: A dictionary mapping neuron IDs to their column index in the matrix.
*   **Example Usage:**
    ```bash
    python b_generate_activation_dataset.py --input_csv parcellation_datasets/test_1.csv --export_dir parcellation_datasets/test_1 --model_type gpt2
    ```

### 3. `b_trace_gelu_activations.py`

*   **Purpose:** A simpler script designed to trace and record the GELU activation pattern for a *single* input text prompt. Useful for debugging or focused analysis on specific inputs.
*   **Functionality:**
    *   Loads a specified GPT model and tokenizer.
    *   Registers forward hooks on GELU activations.
    *   Processes a single input text (`--input_text`).
    *   Performs a forward pass (generating one new token by default).
    *   Captures the GELU activation states (> 0) for the *last* token processed (which corresponds to the prediction for the *next* token).
    *   Flattens these activations into a single binary vector representing all GELU neurons across all layers.
*   **Inputs:**
    *   `--model_type`: GPT model size.
    *   `--input_text`: The text prompt to analyze.
    *   `--output_dir`: Directory to save the activation outputs.
    *   `--max_num_new_tokens`: (Note: Although present, the hook captures activation relevant to the *next* token prediction based on the input. Generating more tokens affects subsequent states but not the primary output of this script's trace function).
*   **Outputs (within `--output_dir`):**
    *   `{model_type}_activations.pkl`: Python pickle file containing the flat binary activation vector.
    *   `{model_type}_active_neurons.txt`: Text file listing the IDs of the neurons that were active (value 1 in the vector).
*   **Example Usage:**
    ```bash
    python b_trace_gelu_activations.py --model_type gpt2 --input_text "The quick brown fox" --output_dir activations
    ```

### 4. `c_calculate_weighted_activation_graph.py`

*   **Purpose:** Constructs a weighted NetworkX graph using the base structure and activation data generated by `b_generate_activation_dataset.py`.
*   **Functionality:**
    *   Infers the `model_type` from filenames in the `--input_dir`.
    *   Loads node attributes (`_neuron_attributes.csv`), base edges (`_neuron_edges.el`), activation frequencies (`_neuron_frequency.csv`), and the activation matrix (`_activation_matrix.npz`).
    *   Creates an empty NetworkX graph.
    *   Adds nodes to the graph, assigning attributes (`layer`, `dimension`) and the total activation count (`activation_count` from the frequency file) as a node weight.
    *   Iterates through the base edges (connections between adjacent layers).
    *   For each edge (u, v), calculates the `co_activation_count` by computing the dot product of the corresponding neuron columns in the binary `activation_matrix`. This count represents how many times neurons `u` and `v` were active *in the same sample*.
    *   Adds edges to the graph with the calculated `co_activation_count` as an edge attribute.
*   **Inputs:**
    *   `--input_dir`: Directory containing the outputs of `b_generate_activation_dataset.py`.
    *   `--output_path`: Full path to save the resulting weighted graph as a `.gpickle` file.
*   **Outputs:**
    *   A `.gpickle` file (specified by `--output_path`) containing the NetworkX graph object. Nodes have `layer`, `dimension`, and `activation_count` attributes. Edges have the `co_activation_count` attribute.
*   **Example Usage:**
    ```bash
    python c_calculate_weighted_activation_graph.py --input_dir parcellation_datasets/test_1 --output_path parcellation_datasets/test_1/gpt2_weighted_graph.gpickle
    ```

### 5. `c_analyze_weighted_activation_graph.py`

*   **Purpose:** Performs analysis on the weighted graph generated by `c_calculate_weighted_activation_graph.py`. It calculates normalized weights, generates statistical plots, performs thresholding analysis, and saves an enhanced graph file.
*   **Functionality:**
    *   Loads the weighted graph (`.gpickle`) specified by `--input_path`.
    *   Removes edges with zero or negative `co_activation_count`.
    *   Calculates normalized node weights (`activation_count_normalized`) by dividing each node's `activation_count` by the maximum activation count across all nodes. Adds this as a node attribute.
    *   Calculates normalized edge weights (`co_activation_count_normalized`) by dividing each edge's `co_activation_count` by the maximum co-activation count across all edges (post-filtering). Adds this as an edge attribute (using float32/float16 for memory efficiency).
    *   **(Commented Out):** Generates histograms for original node activation counts, original edge co-activation counts (using sampling for efficiency), and node degrees. Saves plots as PNG and data as CSV.
    *   Saves the graph with the *newly added normalized attributes* to a file named `{original_name}_analyzed.gpickle` in the `--output_dir`. **Crucially, the subsequent thresholding analysis happens *after* this save.**
    *   **(Currently Exited After Saving):** Performs thresholding analysis:
        *   Iteratively applies thresholds (0.0, 0.1, 0.2, ... 0.9) to the `co_activation_count_normalized` edge attribute.
        *   For each threshold `t`, it removes edges with normalized weight `<= t`.
        *   Calculates statistics for the graph at each threshold level: number of nodes, number of edges, number of connected components, and min/max/average component size.
        *   Saves these thresholding statistics to `threshold_analysis.csv`.
*   **Inputs:**
    *   `--input_path`: Path to the weighted graph `.gpickle` file (output of script `c`).
    *   `--output_dir`: Directory to save analysis results (histograms, CSVs, the analyzed graph). Defaults to a subdirectory `weighted_graph_analysis` within the input graph's directory.
    *   `--hist_sample_size`: (Relevant if histogram code is uncommented) Number of edges to sample for the edge weight histogram.
*   **Outputs (within `--output_dir`):**
    *   `{input_graph_name}_analyzed.gpickle`: The graph with added `activation_count_normalized` (node attribute) and `co_activation_count_normalized` (edge attribute). **This is the recommended input for script `d`.**
    *   **(Commented Out Outputs):**
        *   `node_activation_count_hist.png`, `.csv`
        *   `edge_co_activation_count_hist.png`, `.csv`
        *   `node_degree_hist.png`, `.csv`
    *   **(If Exit is Removed):** `threshold_analysis.csv`: Table summarizing graph properties at different edge weight thresholds.
*   **Example Usage:**
    ```bash
    # Analyze the graph and save results in a default sub-directory
    python c_analyze_weighted_activation_graph.py --input_path parcellation_datasets/test_1/gpt2_weighted_graph.gpickle

    # Specify a custom output directory
    python c_analyze_weighted_activation_graph.py --input_path parcellation_datasets/test_1/gpt2_weighted_graph.gpickle --output_dir analysis_results/test_1_analysis
    ```

### 6. `d_calculate_perturbation_subset.py`

*   **Purpose:** Identifies a subset of potentially "influential" neurons based on graph centrality after filtering edges by a normalized weight threshold. This subset can be used for downstream experiments (e.g., ablation studies).
*   **Functionality:**
    *   Loads an *analyzed* graph (`_analyzed.gpickle` from script `e`) specified by `--input_graph_path`. It expects edges to have the `co_activation_count_normalized` attribute.
    *   Creates a new, unweighted graph (`G_threshold`) containing only the edges from the input graph whose `co_activation_count_normalized` is *greater than* the specified `--threshold`. All original nodes are preserved.
    *   Calculates the number of nodes `k` to select based on the total number of nodes in the thresholded graph and the specified `--ratio` (k = num\_nodes * ratio). Ensures `k` is at least 1 and no more than the total number of nodes.
    *   Selects the top `k` nodes based on the specified `--method`. Currently, only 'degree' centrality is implemented (selects nodes with the highest number of connections in the *thresholded* graph).
    *   Saves the node IDs of the selected top `k` neurons as a NumPy array (`.npy`).
*   **Inputs:**
    *   `--input_graph_path`: Path to the *analyzed* graph `.gpickle` file (output of script `e`). **Crucially, this should be the `_analyzed.gpickle` file, not the original weighted graph.**
    *   `--threshold`: Normalized edge weight threshold (0.0 to 1.0). Edges with `co_activation_count_normalized` > threshold are kept.
    *   `--ratio`: Fraction of total nodes to select as the influential subset (0.0 to 1.0).
    *   `--method`: Centrality measure to use for ranking nodes (currently only 'degree').
    *   `--output_path`: Path to save the NumPy array (`.npy`) containing the selected neuron IDs.
*   **Outputs:**
    *   A `.npy` file (specified by `--output_path`) containing a 1D array of integer node IDs representing the selected influential subset.
*   **Example Usage:**
    ```bash
    # Select top 5% nodes by degree after thresholding edges at 0.1 normalized weight
    python d_calculate_perturbation_subset.py \
        --input_graph_path parcellation_datasets/test_1/weighted_graph_analysis/gpt2_weighted_graph_analyzed.gpickle \
        --threshold 0.1 \
        --ratio 0.05 \
        --method degree \
        --output_path parcellation_datasets/test_1/perturbation_subsets/gpt2_degree_t0.1_r0.05.npy
    ```

### 7. `e_generate_perturbed_results.py`

*   **Purpose:** Runs text generation experiments using the base GPT model while perturbing (zeroing out) the activations of specific GELU neurons identified as influential by previous steps. This allows evaluating the impact of these neurons on model outputs.
*   **Functionality:**
    *   Loads a specified GPT model (`--model_type`) and tokenizer.
    *   Takes a list of perturbation subset filenames (`--perturbation_subset_filenames`) as input. These are typically `.npy` files generated by `d_calculate_perturbation_subset.py`, containing the node IDs of neurons to perturb. The special value `'none'` can be included to run a baseline without perturbation.
    *   For each subset file:
        *   Loads the neuron IDs.
        *   Looks up the corresponding layer and dimension for each neuron ID using the `{model_type}_neuron_attributes.csv` file found in the `--dataset_path`.
        *   Registers PyTorch forward hooks on the GELU activation module (`mlp.gelu` or similar, depending on `model.py`) within each transformer block.
        *   The hook function checks the current layer index and zeroes out the activations at the specified dimensions corresponding to the neurons in the current subset.
        *   Uses a context manager (`apply_perturbations`) to ensure hooks are correctly applied before generation and removed afterward.
    *   Reads prompts from an input CSV (`--input_csv`), which specifies the prompt text, number of new tokens (`number_of_new_tokens`), and number of samples per prompt (`num_samples`).
    *   For each prompt and sample, generates text using `model.generate` with the active perturbation hooks.
    *   Saves the results for each perturbation subset to a separate CSV file.
*   **Inputs:**
    *   `--dataset_path`: Path to the directory containing `_neuron_attributes.csv` and the `perturbation_subsets` subdirectory (e.g., `parcellation_datasets/test_1`).
    *   `--input_csv`: Path to the CSV file containing prompts (e.g., `parcellation_datasets/test_1.csv`). Must have columns `prompts`, `number_of_new_tokens`, `num_samples`.
    *   `--perturbation_subset_filenames`: A list of one or more filenames located in `{dataset_path}/perturbation_subsets/`. Can include `.npy` files and/or the string `'none'`.
    *   `--model_type`: GPT model size (e.g., 'gpt2').
    *   `--seed`: Random seed for reproducibility.
    *   `--temperature`: Temperature for token sampling.
    *   `--top_k`: Top-k sampling parameter.
    *   `--device`: Device to run the model on ('cuda' or 'cpu').
*   **Outputs (within `{dataset_path}/perturbation_results/`):**
    *   One CSV file for each input perturbation subset filename. The filename will match the subset name (e.g., `gpt2_degree_t0.1_r0.05.csv`, `none.csv`).
    *   Each CSV file contains columns: `prompt`, `sample_index`, `generated_text`, `perturbation_subset` (the original filename or 'none').
*   **Example Usage:**
    ```bash
    python e_generate_perturbed_results.py \
        --dataset_path parcellation_datasets/test_1 \
        --input_csv parcellation_datasets/test_1.csv \
        --model_type gpt2 \
        --perturbation_subset_filenames gpt2_degree_t0.1_r0.05.npy gpt2_degree_t0.5_r0.01.npy none \
        --device cuda
    ```

---


## END-to-END, Step-by-Step Example


```bash
# first generate the LLM activation graph
python a_generate_gelu_graph.py --model_type gpt2 --output_dir gelu_graph

# then create a dataset of activations
python b_generate_activation_dataset.py --input_csv parcellation_datasets/test_2.csv --export_dir parcellation_datasets/test_2 --model_type gpt2 --temperature 0.0 --seed 42
# if you want to tracr actiations for a single input text
python b_trace_gelu_activations.py --model_type gpt2 --input_text "The quick brown fox" --output_dir activations


# Calculate weighted activation graph
python c_calculate_weighted_activation_graph.py --input_dir parcellation_datasets/test_2 --output_path parcellation_datasets/test_2/gpt2_weighted_graph.gpickle
#then run the analyss script to generate normalized weights graph
python c_analyze_weighted_activation_graph.py --input_path parcellation_datasets/test_2/gpt2_weighted_graph.gpickle 

#calculate perturbation subsets based on the normalized weights graph
python d_calculate_perturbation_subset.py \
    --input_graph_path parcellation_datasets/test_2/weighted_graph_analysis/gpt2_weighted_graph_analyzed.gpickle \
    --threshold 0.8 \
    --ratio 0.01 \
    --method degree \
    --output_path parcellation_datasets/test_2/perturbation_subsets/gpt2_degree_t0.8_r0.01.npy
# or do batch calculation
./d_batch_perturbation_calculate.sh

```
This README provides a guide to understanding and using the scripts in this pipeline. Ensure that the output of one script matches the expected input format for the next script in the sequence. 