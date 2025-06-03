#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

# Example Usage
RESULT_FILE="PUT YOUR RESULT FILE HERE"
MODEL_PATH="PUT YOUR MODEL PATH HERE"
DATA_FILE="PUT YOUR DATA FILE HERE"

python eval.py \
    --model "${MODEL_PATH}" \
    --data_file "${DATA_FILE}" \
    --result_file "${RESULT_FILE}" \
    --batch_size 10000 \
    --temp 0
