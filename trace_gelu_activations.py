#!/usr/bin/env python
"""
Trace GELU activations for a given input using GPT model.
This script hooks into the GELU activation function and records which neurons are activated.
"""

import os
import pickle
import argparse
import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model import GPT, GPTConfig, MLP

class GELUActivationTracer:
    def __init__(self, model_type='gpt2', device='cuda'):
        self.model_type = model_type
        self.device = device
        
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
        
        # Store activations
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
        
    def trace(self, text_input):
        """Trace GELU activations for the given input text"""
        # Reset activations
        self.gelu_activations = {}
        
        # Tokenize input
        tokens = self.tokenizer.encode(text_input, return_tensors='pt').to(self.device)
        
        # Forward pass
        with torch.no_grad():
            self.model.generate(tokens, max_new_tokens=1, temperature=1.0)
        
        return self.get_activation_vector()
    
    def get_activation_vector(self):
        """Convert the activations to a flat vector of 0s and 1s"""
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
    
    def save_activations(self, activation_vector, output_path):
        """Save the activation vector to a file"""
        with open(output_path, 'wb') as f:
            pickle.dump(activation_vector, f)
        print(f"Activation vector saved to {output_path}")
    
    def get_active_neuron_ids(self, activation_vector):
        """Get the IDs of active neurons"""
        return np.where(activation_vector == 1)[0]


def main():
    parser = argparse.ArgumentParser(description="Trace GELU activations in GPT model")
    parser.add_argument('--model_type', default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                       help='GPT-2 model size to use')
    parser.add_argument('--input_text', default='Hello, how are you?', 
                       help='Input text to trace activations for')
    parser.add_argument('--output_dir', default='activations', 
                       help='Directory to store activation outputs')
    parser.add_argument('--max_num_new_tokens', default=10, 
                       help='Maximum number of new tokens to generate')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tracer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tracer = GELUActivationTracer(model_type=args.model_type, device=device)
    
    # Trace activations
    print(f"Tracing activations for input: {args.input_text}")
    activation_vector = tracer.trace(args.input_text)
    
    # Get active neurons
    active_neuron_ids = tracer.get_active_neuron_ids(activation_vector)
    print(f"Number of active neurons: {len(active_neuron_ids)} out of {len(activation_vector)}")
    
    # Save outputs
    output_file = os.path.join(args.output_dir, 
                              f"{args.model_type}_activations.pkl")
    tracer.save_activations(activation_vector, output_file)
    
    # Save active neuron IDs to a text file for easy viewing
    active_ids_file = os.path.join(args.output_dir, 
                                  f"{args.model_type}_active_neurons.txt")
    with open(active_ids_file, 'w') as f:
        f.write("Active neuron IDs:\n")
        f.write(str(active_neuron_ids.tolist()))
    
    print(f"Active neuron IDs saved to {active_ids_file}")


if __name__ == "__main__":
    main() 