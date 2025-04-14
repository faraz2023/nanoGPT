# GPT GELU Neuron Activation Analysis

This repository contains tools for analyzing GELU neuron activations in GPT-2 models. The toolkit allows you to:

1. Generate a graph representation of GELU neurons
2. Trace neuron activations for specific inputs
3. Create datasets of neuron activations for multiple prompts
4. Analyze and visualize activation patterns
5. Visualize neuron activations for specific samples

## Requirements

Install the required dependencies:

```bash
pip install torch transformers numpy pandas matplotlib seaborn networkx tqdm scikit-learn
```

## Components

### 1. Generate GELU Neuron Graph

Creates a graph representation of GELU activation neurons in the GPT model.

```bash
python a_generate_gelu_graph.py --model_type gpt2 --output_dir gelu_graph
```

Arguments:
- `--model_type`: GPT-2 model size (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- `--output_dir`: Directory to store output files

Outputs:
- Edge list (.el) file representing neuron connections
- CSV file with node attributes (layer, dimension)

### 2. Trace GELU Activations

Traces GELU neuron activations for a specific input prompt.

```bash
python b_trace_gelu_activations.py --model_type gpt2 --input_text "Hello, how are you?" --output_dir activations
```

Arguments:
- `--model_type`: GPT-2 model size
- `--input_text`: Text prompt to trace activations for
- `--output_dir`: Directory to store outputs

Outputs:
- Pickle file with activation vector
- Text file with active neuron IDs

### 3. Generate Activation Dataset

Creates a dataset of neuron activations for multiple prompts from a CSV file, with support for multiple samples per prompt. Also generates the GELU graph structure for the model used.

```bash
python generate_activation_dataset.py --input_csv parcellation_datasets/test_1.csv --export_dir dataset_output --model_type gpt2 --seed 42 --temperature 0.8
```

Arguments:
- `--input_csv`: Path to CSV file with prompts
- `--export_dir`: Directory to export the dataset
- `--model_type`: GPT-2 model size
- `--seed`: Random seed for reproducibility (default: 42)
- `--temperature`: Temperature for sampling (default: 0.8)

Input CSV format:
```
prompts,number_of_new_tokens,num_samples
"I am feeling like gloomy day and ",20,5
"I think love is ",20,5
"My opinion on communism is that ",20,5
```

The `num_samples` column specifies how many different completions to generate for each prompt.

Outputs:
- CSV file with activation data for each generated token, including:
  - sample_id: Unique identifier for each token
  - prompt_sample_idx: Which sample number this is for the prompt
  - token_position: Position in the generated sequence
  - input_text: Original prompt
  - output: Generated token
  - output_id: Token ID
  - number_of_active_neurons: Total count of active neurons
  - layer_N_num_active_neurons: Count of active neurons in layer N
- Pickle/JSON file with the active neuron IDs for each sample
- Edge list (.el) file representing neuron connections
- CSV file with node attributes (layer, dimension)
- Neuron frequency CSV file with activation counts and frequencies
- Compressed NPZ file with activation matrix (samples x neurons)

### 4. Analyze Activations

Analyzes and visualizes neuron activation patterns across different prompts.

```bash
python analyze_activations.py --dataset_csv dataset_output/gpt2_activation_dataset.csv --activations_file dataset_output/gpt2_neuron_activations.json --output_dir analysis_output --model_type gpt2
```

Arguments:
- `--dataset_csv`: Path to the activation dataset CSV
- `--activations_file`: Path to the activations file
- `--output_dir`: Directory to save analysis outputs
- `--model_type`: GPT-2 model size

Outputs:
- Visualizations of token clusters based on neuron activations
- Analysis of the most important neurons
- Identification of token-specific neurons
- Comparison of samples from the same prompt
- Analysis of position-specific neurons

### 5. Visualize Sample Activations

Generates heatmap visualizations of neuron activations for specific sample IDs.

```bash
python visualize_sample_activations.py --dataset_dir dataset_output --sample_ids 0,50,200 --export_dir visualizations --model_type gpt2
```

Arguments:
- `--dataset_dir`: Directory containing the dataset and activation files
- `--sample_ids`: Comma-separated list of sample IDs to visualize
- `--export_dir`: Directory to save the visualizations
- `--model_type`: GPT-2 model size (default: gpt2)
- `--figure_rows`: Number of rows to arrange layer plots in (default: single row)
- `--vis_mode`: Visualization mode (default, cinematic, cinematic_no_text)
  - `default`: Blue background with red activations, includes all metadata text
  - `cinematic`: Black background with white activations, minimal text (input only)
  - `cinematic_no_text`: Black background with white activations, no text at all

Each layer is represented as a heatmap with active neurons highlighted. The default visualization includes detailed information about active neurons per layer, while cinematic modes are designed for artistic/presentation purposes with transparent backgrounds.

Example commands:
```bash
# Default visualization
python visualize_sample_activations.py --dataset_dir dataset_output --sample_ids 0,50,200 --export_dir visualizations

# Artistic visualization with minimal text, 3 rows of layers
python visualize_sample_activations.py --dataset_dir dataset_output --sample_ids 0,50,200 --export_dir visualizations --vis_mode cinematic --figure_rows 3

# Completely text-free visualization with transparent background, 4 rows
python visualize_sample_activations.py --dataset_dir dataset_output --sample_ids 0,50,200 --export_dir visualizations --vis_mode cinematic_no_text --figure_rows 4
```

The script will generate PNG files for each sample ID, with each visualization showing the activity pattern across all layers of the model.

## Example Workflow

1. Generate an activation dataset from a set of prompts with multiple samples:
   ```bash
   python generate_activation_dataset.py --input_csv parcellation_datasets/test_1.csv --export_dir dataset_output --model_type gpt2 --seed 42 --temperature 0.8
   ```
   Note: This now also creates the GELU graph structure and additional activation maps automatically.

2. Analyze the activation patterns:
   ```bash
   python analyze_activations.py --dataset_csv dataset_output/gpt2_activation_dataset.csv --activations_file dataset_output/gpt2_neuron_activations.json --output_dir analysis_output --model_type gpt2
   ```

3. Visualize the graph with activations:
   ```bash
   python visualize_gelu_graph.py --model_type gpt2 --graph_dir dataset_output --activation_file dataset_output/gpt2_neuron_activations.pkl --output_dir visualizations
   ```

## Data Outputs

The `generate_activation_dataset.py` script now produces the following files in the export directory:

1. `{model_type}_activation_dataset.csv` - Main dataset with token-level information
2. `{model_type}_neuron_activations.pickle` - Pickle file with active neuron IDs per sample
3. `{model_type}_neuron_activations.json` - JSON version of the active neuron IDs
4. `{model_type}_neuron_attributes.csv` - Node attributes for the GELU graph (layer, dimension)
5. `{model_type}_neuron_edges.el` - Edge list file for the GELU graph
6. `{model_type}_neuron_frequency.csv` - Neuron activation frequency statistics
7. `{model_type}_activation_matrix.npz` - Compressed activation matrix (samples x neurons)

## Comparing Samples for the Same Prompt

With multiple samples per prompt, you can perform additional analyses:

1. Identify consistently activated neurons across different completions of the same prompt
2. Compare neuron activations for different prompts but similar generated tokens
3. Study the impact of temperature on neuron activation patterns

The extended dataset includes the `prompt_sample_idx` field to track which sample number each token belongs to, making it easy to group and compare activations across different completions.

## Notes on Part 2 (Future Work)

For the second part of the project (not yet implemented), we'll build on this foundation to:

1. Predict neuron activations based on input tokens
2. Extract interpretable patterns from the activation data
3. Create a model that can predict the next token based solely on GELU activations

This will help us understand how specific neurons contribute to the language model's behavior and potentially identify specialized neurons or circuits. 