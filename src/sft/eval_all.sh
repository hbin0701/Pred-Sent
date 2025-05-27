#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

step=12888
result_file="results/qwen-cot-step${step}.jsonl"

python eval.py \
    --model "/home/hyeonbin/Latent_Step/src/sft/checkpoints/qwen2.5/checkpoint-${step}" \
    --data_file "/home/hyeonbin/Latent_Step/data/csqa/test.json" \
    --result_file "${result_file}" \
    --batch_size 1000000000 \
    --temp 0
