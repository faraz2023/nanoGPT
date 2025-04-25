#!/bin/bash

# --- Configuration ---
# Set the base path for your dataset
DATASET_PATH="parcellation_datasets/test_2"
# Set the model type
MODEL_TYPE="gpt2"
# Set the device ('cuda' or 'cpu')
DEVICE="cuda"
# --- End Configuration ---

# Derive other paths from DATASET_PATH
INPUT_CSV="${DATASET_PATH}.csv" # Assumes csv has same name as dir, adjust if needed
PERTURBATION_SUBSET_DIR="${DATASET_PATH}/perturbation_subsets"

# Check if the perturbation subset directory exists
if [ ! -d "$PERTURBATION_SUBSET_DIR" ]; then
    echo "Error: Perturbation subset directory not found: $PERTURBATION_SUBSET_DIR"
    exit 1
fi

# Check if the input CSV file exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "Error: Input CSV file not found: $INPUT_CSV"
    echo "Note: Assumed CSV name matches DATASET_PATH (${DATASET_PATH}.csv). Adjust script if needed."
    exit 1
fi


# Find all .npy files in the directory and get their basenames
# Store them in an array
subset_files=()
while IFS= read -r -d $'\0' file; do
    subset_files+=("$(basename "$file")")
done < <(find "$PERTURBATION_SUBSET_DIR" -maxdepth 1 -name '*.npy' -print0)

# Add 'none' to the list
subset_files+=("none")

# Check if any files were found (besides 'none')
if [ ${#subset_files[@]} -eq 1 ]; then
    echo "Warning: No .npy files found in $PERTURBATION_SUBSET_DIR. Running only with 'none'."
elif [ ${#subset_files[@]} -gt 1 ]; then
    echo "Found the following perturbation files to process:"
    printf " - %s\n" "${subset_files[@]}"
else
    # This case should not happen if 'none' is always added, but included for safety
    echo "Error: Could not find any .npy files or construct the file list."
    exit 1
fi
echo "----------------------------------------"

# Construct and run the python command
echo "Running e_generate_perturbed_results.py..."

python e_generate_perturbed_results.py \
    --dataset_path "$DATASET_PATH" \
    --input_csv "$INPUT_CSV" \
    --model_type "$MODEL_TYPE" \
    --perturbation_subset_filenames "${subset_files[@]}" \
    --device "$DEVICE" \
    --temperature 0.0 \
    --top_k 1

# Check the exit status
if [ $? -ne 0 ]; then
    echo "Error running e_generate_perturbed_results.py"
    exit 1
fi

echo "----------------------------------------"
echo "Script finished."