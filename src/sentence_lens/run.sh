#!/bin/bash
python sentence_lens.py \
  --model_dir "PUT_MODEL_PATH_HERE" \
  --data_path "../../data/csqa/test.json" \
  --num_iterations 8 \
  --output_dir "PUT_OUTPUT_DIR_HERE"