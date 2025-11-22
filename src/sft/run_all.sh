#!/bin/bash

# Create necessary directories
echo "Creating output directories..."
mkdir -p /home/hyeonbin/Pred-Sent/models/trained_models/pythia
if [ $? -ne 0 ]; then
    echo "Error: Failed to create output directory. Please check permissions and try again."
    exit 1
fi

# Function to run training and evaluation for a dataset
run_dataset() {
    local dataset=$1
    local mode=$2  # "cot" or "no-cot"
    
    echo "Running training for ${dataset} with mode=${mode} on GPUs 0-3"
    
    # Run training
    if [ "$mode" = "cot" ]; then
        bash src/sft/scripts/pythia/pythia_run_${dataset}_cot.sh
        bash src/sft/scripts/pythia/run_parallel_eval_${dataset}_cot.sh
    else
        bash src/sft/scripts/pythia/pythia_run_${dataset}_no_cot.sh
        bash src/sft/scripts/pythia/run_parallel_eval_${dataset}.sh
    fi
}

# Run datasets sequentially
datasets=("blocksworld7" "csqa" "prosqa" "gsm8k")
modes=("cot" "no-cot")

for mode in "${modes[@]}"; do
    for dataset in "${datasets[@]}"; do
        run_dataset "${dataset}" "${mode}"
    done
done

echo "All training and evaluation completed!" 