#!/usr/bin/env python
"""
Generates text completions using a GPT model with specific GELU neurons perturbed (zeroed out).

Loads a base model, applies perturbations based on specified neuron subsets,
runs prompts from an input CSV, and saves the generated outputs for each perturbation setting.

python e_generate_perturbed_results.py \
    --dataset_path parcellation_datasets/test_2 \
    --input_csv parcellation_datasets/test_2.csv \
    --model_type gpt2 \
    --perturbation_subset_filenames gpt2_degree_t0.5_r0.9.npy none \
    --device cuda # or cpu

python e_generate_perturbed_results.py \
    --dataset_path parcellation_datasets/test_2 \
    --input_csv parcellation_datasets/test_2_tiny.csv \
    --model_type gpt2 \
    --perturbation_subset_filenames gpt2_degree_t0.5_r0.9.npy none \
    --device cuda > z.txts
"""

import os
import argparse
import pickle
import pandas as pd
import numpy as np
import torch
from contextlib import contextmanager
import gc

# Assuming model.py and configuration exist in the same directory or PYTHON PATH
# Need to adjust the import based on your project structure if model.py is elsewhere
# from model import ModelArgs, Transformer # Example if using Llama-style model.py
# Or for nanoGPT style:
from model import GPTConfig, GPT # Assuming nanoGPT structure
from transformers import GPT2Tokenizer # Use standard tokenizer

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate perturbed model results.")
    parser.add_argument('--dataset_path', required=True,
                        help='Path to the dataset directory (e.g., parcellation_datasets/test_1)')
    parser.add_argument('--input_csv', required=True,
                        help='Path to the input CSV file containing prompts (e.g., parcellation_datasets/test_1.csv)')
    parser.add_argument('--perturbation_subset_filenames', nargs='+', required=True,
                        help="List of perturbation subset filenames (e.g., 'gpt2_degree_t0.8_r0.01.npy' or 'none') located in dataset_path/perturbation_subsets/")
    parser.add_argument('--model_type', default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                       help='GPT-2 model size to use.')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed for reproducibility.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for sampling.')
    parser.add_argument('--top_k', type=int, default=1, help='Top-k sampling.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cpu or cuda).')

    return parser.parse_args()

def load_model_and_tokenizer(model_type, device):
    """Loads the GPT model and tokenizer."""
    print(f"Loading model: {model_type} on device: {device}")
    # Load the model configuration from nanoGPT/huggingface checkpoint
    # Use GPT.from_pretrained for nanoGPT style
    model = GPT.from_pretrained(model_type)
    model.eval()
    model.to(device)
    
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    tokenizer.pad_token = tokenizer.eos_token # Often needed for batch generation if padding

    print("Model and tokenizer loaded.")
    return model, tokenizer

# Placeholder for the perturbation logic
perturbation_hooks = []
neurons_to_perturb = {} # Global or passed around, format: {layer_idx: tensor_of_dims}

def perturbation_hook(module, input, output):
    """Forward hook to zero out specific neuron activations."""
    global neurons_to_perturb
    # This hook assumes it's attached to the activation layer (e.g., GELU) within an MLP block
    
    # Find the layer index - this requires knowing the model structure precisely.
    # This is fragile and depends on the exact naming in model.py
    layer_idx = -1
    module_name = ""
    for name, mod in model.named_modules(): # Accessing global 'model' - needs refactoring
        if mod is module:
            module_name = name
            break
    # print("From perturbation hook: ", module_name)
    # Example logic for nanoGPT structure: model.transformer.h[layer_idx].mlp.act
    try:
        if module_name.startswith('transformer.h.') and module_name.endswith('.mlp.gelu'):
            layer_idx = int(module_name.split('.')[2])
    except (ValueError, IndexError):
        print(f"Warning: Could not determine layer index for module {module_name}")
        layer_idx = -1
        raise ValueError(f"Warning: Could not determine layer index for module {module_name}")

    # print("Layer index: ", layer_idx)
    if layer_idx != -1 and layer_idx in neurons_to_perturb:
        dims_to_zero = neurons_to_perturb[layer_idx]
        # Ensure output is mutable or create a clone if needed
        # Output shape for GELU is likely (batch_size, sequence_length, gelu_dim)
        # We need to zero out specific indices in the last dimension
        if isinstance(output, torch.Tensor):
             output[:, :, dims_to_zero] = 0.0
        elif isinstance(output, tuple): # Sometimes hooks get tuples
             # Assuming the relevant tensor is the first element
             if isinstance(output[0], torch.Tensor):
                  output[0][:, :, dims_to_zero] = 0.0
             else:
                  raise ValueError(f"Warning: Hooked module output[0] is not a tensor (layer {layer_idx}). Type: {type(output[0])}")
                  print(f"Warning: Hooked module output[0] is not a tensor (layer {layer_idx}). Type: {type(output[0])}")
        else:
            raise ValueError(f"Warning: Hooked module output is not a tensor or tuple (layer {layer_idx}). Type: {type(output)}")
            print(f"Warning: Hooked module output is not a tensor or tuple (layer {layer_idx}). Type: {type(output)}")

    # print("================================================")   
    return output # Must return the output for the forward pass to continue


@contextmanager
def apply_perturbations(model, perturbation_df, device):
    """Context manager to apply and remove perturbation hooks."""
    global perturbation_hooks, neurons_to_perturb
    perturbation_hooks = []
    neurons_to_perturb = {}

    if not perturbation_df.empty:
        print(f"  Applying perturbations for {len(perturbation_df)} neurons...")
        # Group by layer
        grouped = perturbation_df.groupby('layer')['dimension'].apply(list).to_dict()

        #export grouped to .json with pretty print
        # import json
        # with open('zz_grouped.json', 'w') as f:
        #     json.dump(grouped, f, indent=4)
        
        # Convert lists to tensors on the correct device for efficient indexing
        for layer_idx, dims in grouped.items():
            neurons_to_perturb[layer_idx] = torch.tensor(dims, dtype=torch.long, device=device)

            
        # Register hooks on the activation modules (e.g., GELU) in each MLP layer
        # This depends heavily on the model architecture definition in model.py
        for i, layer in enumerate(model.transformer.h):
            print(i, layer)
            print("-----")
            # Adjust 'mlp.act' if your model uses a different name/structure
            hook = layer.mlp.gelu.register_forward_hook(perturbation_hook)
            perturbation_hooks.append(hook)
        print(f"  Registered {len(perturbation_hooks)} hooks.")
    else:
        print("  No perturbations to apply (running baseline).")

    try:
        yield # Execute the code block within the 'with' statement
    finally:
        print("  Removing perturbation hooks...")
        for hook in perturbation_hooks:
            hook.remove()
        perturbation_hooks = []
        neurons_to_perturb = {}
        gc.collect() # Clean up memory
        print("  Hooks removed.")


def generate_text(model, tokenizer, prompt, num_new_tokens, temperature, top_k, device):
    """Generates text for a single prompt."""
    model.eval() # Ensure model is in eval mode
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate output using the model's generate method
    # Ensure generate parameters match what model.py expects
    with torch.no_grad():
         # Need to check model.py for exact generate args. Assuming nanoGPT style.
         output_ids = model.generate(input_ids, 
                                    max_new_tokens=num_new_tokens, 
                                    temperature=temperature, 
                                    top_k=top_k)
    
    # Decode the generated tokens, skipping the prompt part
    generated_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
    
    return generated_text

def main():
    args = parse_args()

    if args.top_k > 1 or args.temperature != 0.0:
        print("Warning: Top-k sampling is not supported. Setting top-k to 1 and temperature to 0.0.")
        args.top_k = 1
        args.temperature = 0.0
        exit(0)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # --- Load Base Model ---
    # Declaring model globally for the hook function - needs refactoring ideally
    global model 
    model, tokenizer = load_model_and_tokenizer(args.model_type, args.device)

    # --- Load Neuron Attributes ---
    neuron_attributes_path = os.path.join(args.dataset_path, f"{args.model_type}_neuron_attributes.csv")
    if not os.path.exists(neuron_attributes_path):
        print(f"Error: Neuron attributes file not found: {neuron_attributes_path}")
        return
    try:
        neuron_attributes_df = pd.read_csv(neuron_attributes_path)
        print(f"Loaded neuron attributes for {len(neuron_attributes_df)} neurons.")
    except Exception as e:
        print(f"Error loading neuron attributes CSV: {e}")
        return

    # --- Load Prompts ---
    if not os.path.exists(args.input_csv):
        print(f"Error: Input prompts CSV not found: {args.input_csv}")
        return
    try:
        prompts_df = pd.read_csv(args.input_csv)
        # Validate required columns
        if not {'prompts', 'number_of_new_tokens', 'num_samples'}.issubset(prompts_df.columns):
             raise ValueError("Input CSV must contain columns: 'prompts', 'number_of_new_tokens', 'num_samples'")
        print(f"Loaded {len(prompts_df)} prompts from {args.input_csv}.")
    except Exception as e:
        print(f"Error loading prompts CSV: {e}")
        return
        
    # --- Create Output Directory ---
    output_base_dir = os.path.join(args.dataset_path, "perturbation_results")
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Results will be saved in: {output_base_dir}")

    # --- Iterate Through Perturbation Subsets ---
    for subset_filename in args.perturbation_subset_filenames:
        print(f"--- Processing perturbation subset: {subset_filename} ---")
        
        perturbation_subset_df = pd.DataFrame(columns=['node_id', 'layer', 'dimension']) # Default empty
        
        if subset_filename.lower() != 'none':
            subset_path = os.path.join(args.dataset_path, "perturbation_subsets", subset_filename)
            if not os.path.exists(subset_path):
                print(f"Warning: Perturbation subset file not found: {subset_path}. Skipping.")
                continue
            try:
                subset_node_ids = np.load(subset_path)
                if subset_node_ids.size > 0:
                     # Filter attributes based on loaded node IDs
                     perturbation_subset_df = neuron_attributes_df[neuron_attributes_df['node_id'].isin(subset_node_ids)].copy()
                     print(f"  Loaded {len(subset_node_ids)} node IDs. Found {len(perturbation_subset_df)} matching neurons to perturb.")
                else:
                     print("  Loaded subset file, but it contains no node IDs.")
            except Exception as e:
                print(f"Error loading perturbation subset file {subset_path}: {e}. Skipping.")
                raise e
                continue
        else:
             print("  Running baseline (no perturbation).")
             

        # Prepare output filename
        subset_name = os.path.splitext(subset_filename)[0] if subset_filename.lower() != 'none' else 'none'
        output_csv_path = os.path.join(output_base_dir, f"{subset_name}.csv")

        # --- Apply Perturbations and Generate ---
        results = []
        # Use context manager to handle hook registration/removal
        with apply_perturbations(model, perturbation_subset_df, args.device):
            
            # Loop through prompts
            for index, row in prompts_df.iterrows():
                prompt_text = row['prompts']
                num_tokens = row['number_of_new_tokens']
                num_samples = row['num_samples']
                print(f"  Generating for prompt {index+1}/{len(prompts_df)} ('{prompt_text[:50]}...') - {num_samples} samples, {num_tokens} tokens...")

                # Loop through samples for the current prompt
                for sample_idx in range(num_samples):
                    generated_text = generate_text(model, tokenizer, prompt_text, num_tokens, 
                                                 args.temperature, args.top_k, args.device)
                    
                    results.append({
                        'prompt': prompt_text,
                        'sample_index': sample_idx,
                        'generated_text': generated_text,
                        'perturbation_subset': subset_filename 
                    })
                    
                    if (sample_idx + 1) % 5 == 0: # Print progress within samples
                         print(f"    Sample {sample_idx+1}/{num_samples} completed.")

        # --- Save Results for this Subset ---
        results_df = pd.DataFrame(results)
        try:
            results_df.to_csv(output_csv_path, index=False)
            print(f"Results saved to {output_csv_path}")
        except Exception as e:
            print(f"Error saving results CSV to {output_csv_path}: {e}")
            
        # Optional: Clear memory explicitly if needed
        del results_df
        del results
        del perturbation_subset_df
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    print("Script finished.")

if __name__ == "__main__":
    # Need to make model accessible globally for the hook
    # This is not ideal, consider refactoring later if needed
    model = None 
    main() 