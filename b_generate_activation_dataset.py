#!/usr/bin/env python

# python b_generate_activation_dataset.py --input_csv parcellation_datasets/test_1.csv --export_dir parcellation_datasets/test_1
# python b_generate_activation_dataset.py --input_csv parcellation_datasets/test_1.csv --export_dir parcellation_datasets/test_1_small
# python b_generate_activation_dataset.py --input_csv parcellation_datasets/test_1.csv --export_dir parcellation_datasets/test_1_medium --model_type gpt2-medium

"""
Generate a dataset of GELU neuron activations for multiple prompts.
This script processes a CSV file of prompts and generates GELU neuron activations
for each token generated by the model, supporting multiple samples per prompt.
It also creates a graph representation of GELU activation neurons with node attributes.
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import GPT2Tokenizer
from tqdm import tqdm
from model import GPT, GPTConfig

class ActivationDatasetGenerator:
    def __init__(self, model_type='gpt2', device='cuda', seed=42, temperature=0.8):
        self.model_type = model_type
        self.device = device
        self.seed = seed
        self.temperature = temperature
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        
        # Load GPT-2 config
        if model_type == 'gpt2':
            n_layer, n_head, n_embd = 12, 12, 768
        elif model_type == 'gpt2-medium':
            n_layer, n_head, n_embd = 24, 16, 1024
        elif model_type == 'gpt2-large':
            n_layer, n_head, n_embd = 36, 20, 1280
        elif model_type == 'gpt2-xl':
            n_layer, n_head, n_embd = 48, 25, 1600
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.gelu_dim = 4 * n_embd
        
        # Load model and tokenizer
        print(f"Loading {model_type} model...")
        config = GPTConfig(
            n_layer=n_layer, 
            n_head=n_head, 
            n_embd=n_embd,
            bias=True
        )
        self.model = GPT.from_pretrained(model_type, dict(dropout=0.0))
        self.model.eval()
        self.model.to(device)
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        
        # Register hooks
        self.gelu_activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture GELU activations"""
        for i, block in enumerate(self.model.transformer.h):
            def get_activation(name):
                def hook(module, input, output):
                    # Store GELU activations for the last token
                    # Shape is [batch_size, seq_len, 4*n_embd]
                    # We're interested in the last token's activations
                    self.gelu_activations[name] = (output[:, -1, :] > 0).cpu().numpy()
                return hook
            
            # Register hook on the GELU activation in the MLP
            block.mlp.gelu.register_forward_hook(get_activation(f'gelu_layer_{i}'))
    
    def generate_gelu_graph(self, export_dir):
        """Generate a graph representation of GELU activation neurons"""
        print("Generating GELU neuron graph...")
        
        # Calculate the dimensions of GELU activations
        gelu_dim = self.gelu_dim
        n_layer = self.n_layer
        
        # Total number of GELU neurons
        total_neurons = n_layer * gelu_dim
        print(f"Model: {self.model_type}")
        print(f"Layers: {n_layer}, Embedding dimension: {self.n_embd}, GELU dimension: {gelu_dim}")
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
        attr_file = os.path.join(export_dir, f"{self.model_type}_neuron_attributes.csv")
        with open(attr_file, 'w') as f:
            f.write("node_id,layer,dimension\n")
            for node_id, layer, dim in node_attrs:
                f.write(f"{node_id},{layer},{dim}\n")
        
        # Write edges to edge list file
        edge_file = os.path.join(export_dir, f"{self.model_type}_neuron_edges.el")
        with open(edge_file, 'w') as f:
            for src, tgt in edges:
                f.write(f"{src} {tgt}\n")
        
        print(f"Node attributes written to {attr_file}")
        print(f"Edge list written to {edge_file}")
        print(f"Total nodes: {len(node_attrs)}")
        print(f"Total edges: {len(edges)}")
        
        return attr_file, edge_file
    
    def generate_dataset(self, input_csv_path, export_dir):
        """Generate a dataset of neuron activations for prompts in the input CSV"""
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        # First, generate the GELU graph structure
        self.generate_gelu_graph(export_dir)
        
        # Read input CSV
        df = pd.read_csv(input_csv_path)
        
        # Prepare output dataframe
        output_data = []
        # Dictionary to store active neuron lists for each sample
        activation_data = {}
        
        # Process each prompt
        sample_id = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
            prompt = row['prompts']
            max_new_tokens = int(row['number_of_new_tokens'])
            num_samples = int(row['num_samples']) if 'num_samples' in row else 1
            
            # Process each sample for this prompt
            for sample_idx in range(num_samples):
                # Set different seed for each sample but ensure deterministic behavior
                sample_seed = self.seed + sample_id
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed(sample_seed)
                np.random.seed(sample_seed)
                
                # Get activations and generated tokens for this prompt sample
                sample_data, sample_activations = self._process_prompt(
                    prompt, max_new_tokens, sample_id, sample_idx
                )
                
                # Add to output data
                output_data.extend(sample_data)
                # Store activations
                activation_data.update(sample_activations)
                
                # Update sample_id for next prompt
                sample_id += len(sample_data)
        
        # Create output dataframe
        output_df = pd.DataFrame(output_data)
        
        # Save output data
        output_csv_path = os.path.join(export_dir, f"{self.model_type}_activation_dataset.csv")
        output_df.to_csv(output_csv_path, index=False)
        print(f"Dataset saved to {output_csv_path}")
        
        # Save activation data (neuron lists)
        activations_path = os.path.join(export_dir, f"{self.model_type}_neuron_activations.pickle")
        with open(activations_path, 'wb') as f:
            pickle.dump(activation_data, f)
        print(f"Neuron activations saved to {activations_path}")
        
        # Save a smaller JSON version with just the neuron IDs
        activations_json_path = os.path.join(export_dir, f"{self.model_type}_neuron_activations.json")
        with open(activations_json_path, 'w') as f:
            json.dump(activation_data, f)
        print(f"Neuron activations (JSON) saved to {activations_json_path}")
        
        # Create an activation map
        self.generate_activation_maps(activation_data, export_dir)
        
        return output_csv_path, activations_path
    
    def generate_activation_maps(self, activation_data, export_dir):
        """Generate activation maps and statistics from the collected data"""
        print("Generating activation maps and statistics...")
        
        # Calculate neuron activation frequency
        activation_counts = {}
        total_samples = len(activation_data)
        
        # Count how many times each neuron was activated
        all_neurons = set()
        for sample_id, neurons in activation_data.items():
            for neuron_id in neurons:
                all_neurons.add(neuron_id)
                activation_counts[neuron_id] = activation_counts.get(neuron_id, 0) + 1
        
        # Calculate activation frequency (percentage of samples where neuron was active)
        activation_frequency = {neuron_id: count / total_samples 
                               for neuron_id, count in activation_counts.items()}
        
        # Save activation frequency
        freq_df = pd.DataFrame([
            {'neuron_id': neuron_id, 
             'activation_count': activation_counts[neuron_id],
             'activation_frequency': activation_frequency[neuron_id],
             'layer': neuron_id // self.gelu_dim,
             'dimension': neuron_id % self.gelu_dim}
            for neuron_id in activation_counts
        ])
        
        # Save neuron activation frequency
        freq_path = os.path.join(export_dir, f"{self.model_type}_neuron_frequency.csv")
        freq_df.to_csv(freq_path, index=False)
        print(f"Neuron activation frequency saved to {freq_path}")
        
        # Create neuron co-activation matrix (binary)
        activation_matrix = np.zeros((total_samples, len(all_neurons)), dtype=np.int8)
        neuron_to_idx = {neuron_id: i for i, neuron_id in enumerate(sorted(all_neurons))}
        
        # Fill the activation matrix
        for i, (sample_id, neurons) in enumerate(activation_data.items()):
            for neuron_id in neurons:
                activation_matrix[i, neuron_to_idx[neuron_id]] = 1
        
        # Save the activation matrix
        matrix_path = os.path.join(export_dir, f"{self.model_type}_activation_matrix.npz")
        np.savez_compressed(matrix_path, 
                           matrix=activation_matrix, 
                           neuron_ids=np.array(list(neuron_to_idx.keys())),
                           neuron_map=neuron_to_idx)
        print(f"Activation matrix saved to {matrix_path}")
        
        # Return the paths to the generated files
        return freq_path, matrix_path
    
    def _process_prompt(self, prompt, max_new_tokens, start_sample_id, sample_idx=0):
        """Process a single prompt and generate activations for each token"""
        # Reset activations
        self.gelu_activations = {}
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        input_text_ids = input_ids[0].tolist()
        
        # Store the initial prompt tokens for reference
        tokens_so_far = input_ids.clone()
        
        # Sample data to collect
        sample_data = []
        # Dictionary to store neuron activations for each sample
        sample_activations = {}
        
        # Generate tokens one by one
        with torch.no_grad():
            for i in range(max_new_tokens):
                # Forward pass for next token prediction
                logits, _ = self.model(tokens_so_far)
                
                # Apply temperature
                logits = logits[:, -1, :] / self.temperature
                
                # Apply softmax to convert logits to probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Sample from distribution
                next_token_id = torch.multinomial(probs, num_samples=1)
                
                # Get the text for the predicted token
                next_token_text = self.tokenizer.decode(next_token_id[0])
                
                # Get activation vector for this token
                activation_vector = self._get_activation_vector()
                active_neuron_ids = np.where(activation_vector == 1)[0].tolist()
                
                # Count active neurons per layer
                layer_active_counts = []
                for layer in range(self.n_layer):
                    start_idx = layer * self.gelu_dim
                    end_idx = start_idx + self.gelu_dim
                    layer_neurons = activation_vector[start_idx:end_idx]
                    layer_active_counts.append(np.sum(layer_neurons))
                
                # Create sample record
                current_sample_id = start_sample_id + i
                sample_record = {
                    'sample_id': current_sample_id,
                    'prompt_sample_idx': sample_idx,
                    'token_position': i,
                    'input_text': prompt,
                    'output': next_token_text,
                    'input_text_ids': input_text_ids,
                    'output_id': next_token_id.item(),
                    'number_of_active_neurons': len(active_neuron_ids)
                }
                
                # Add layer-specific active neuron counts
                for layer, count in enumerate(layer_active_counts):
                    sample_record[f'layer_{layer}_num_active_neurons'] = int(count)
                
                # Add to sample data
                sample_data.append(sample_record)
                
                # Store active neuron IDs for this sample
                sample_activations[str(current_sample_id)] = active_neuron_ids
                
                # Update prompt with the newly generated token
                tokens_so_far = torch.cat((tokens_so_far, next_token_id), dim=1)
                
                # Update prompt text for next iteration
                prompt = prompt + next_token_text
        
        return sample_data, sample_activations
    
    def _get_activation_vector(self):
        """Get the activation vector from the current GELU activations"""
        activation_vector = np.zeros(self.n_layer * self.gelu_dim, dtype=np.int8)
        
        for layer_idx in range(self.n_layer):
            layer_key = f'gelu_layer_{layer_idx}'
            if layer_key in self.gelu_activations:
                # Get the activation pattern for this layer (binary 0/1)
                layer_activations = self.gelu_activations[layer_key][0]  # batch size is 1
                
                # Calculate the start index for this layer in the flat vector
                start_idx = layer_idx * self.gelu_dim
                end_idx = start_idx + self.gelu_dim
                
                # Assign activations to the correct positions
                activation_vector[start_idx:end_idx] = layer_activations
        
        return activation_vector


def main():
    parser = argparse.ArgumentParser(description="Generate GELU neuron activation dataset")
    parser.add_argument('--input_csv', required=True,
                        help='Path to input CSV file with prompts')
    parser.add_argument('--export_dir', required=True,
                        help='Directory to export the dataset')
    parser.add_argument('--model_type', default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='GPT-2 model size to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Temperature for sampling (default: 0.8)')
    args = parser.parse_args()
    
    # Initialize dataset generator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = ActivationDatasetGenerator(
        model_type=args.model_type, 
        device=device,
        seed=args.seed,
        temperature=args.temperature
    )
    
    # Generate dataset
    generator.generate_dataset(args.input_csv, args.export_dir)


if __name__ == "__main__":
    main() 