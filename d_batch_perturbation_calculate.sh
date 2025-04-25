#!/bin/bash

set -e

# Define the input graph path
INPUT_GRAPH="parcellation_datasets/test_2/weighted_graph_analysis/gpt2_weighted_graph_analyzed.gpickle"
# Define the base output directory
OUTPUT_DIR="parcellation_datasets/test_2/perturbation_subsets"
# Define the method
METHOD="degree"
# Define the model type prefix for the filename
MODEL_TYPE="gpt2"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define the threshold values
thresholds=(0.5) #(0.7 0.8 0.9)
# Define the ratio values
ratios=(0.9) #(0.01 0.02 0.05 0.1)

# Check if the input graph exists
if [ ! -f "$INPUT_GRAPH" ]; then
    echo "Error: Input graph file not found: $INPUT_GRAPH"
    exit 1
fi

# Loop through each threshold
for t in "${thresholds[@]}"; do
  # Loop through each ratio
  for r in "${ratios[@]}"; do
    # Construct the output filename
    output_filename="${MODEL_TYPE}_${METHOD}_t${t}_r${r}.npy"
    output_path="${OUTPUT_DIR}/${output_filename}"

    echo "Running with threshold=${t}, ratio=${r}"
    echo "Output path: ${output_path}"

    # Run the python script
    python d_calculate_perturbation_subset.py \
      --input_graph_path "$INPUT_GRAPH" \
      --threshold "$t" \
      --ratio "$r" \
      --method "$METHOD" \
      --output_path "$output_path"

    # Optional: Add a check for the exit status of the python script
    if [ $? -ne 0 ]; then
        echo "Error running script for threshold=${t}, ratio=${r}"
        # Decide whether to continue or exit the script on error
        # exit 1 # Uncomment to exit on first error
    fi
    echo "----------------------------------------"

  done
done

echo "Script finished."