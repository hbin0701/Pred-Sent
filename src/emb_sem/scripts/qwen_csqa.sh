#!/bin/bash

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="49e3bf1d97a7148ae772622876fd9ac8b08ce60e"
export WANDB_ENTITY="hbin0701"

# Define argument variables
TRAIN_FILE="fw-edu/train"
VALID_FILE="/home/hyeonbin/Pred-Sent/data/csqa/valid.json"
TEST_FILE="/home/hyeonbin/Pred-Sent/data/csqa/test.json"

# Using Qwen checkpoint-136 which showed best accuracy of 0.602
ENCODER_MODEL="/home/hyeonbin/Pred-Sent/models/qwen_trained/csqa-sft-cot-new/checkpoint-51"
DECODER_MODEL="/home/hyeonbin/Pred-Sent/models/qwen_trained/csqa-sft-cot-new/checkpoint-51"
TOKENIZER_MODEL="/home/hyeonbin/Pred-Sent/models/Qwen2.5-0.5B"

SHARE_PARAM="True"

BATCH_SIZE=128
NUM_EPOCHS=5
LEARNING_RATE="5e-4"
MAX_LENGTH=512
NUM_WORKERS=1

EXP_NAME="emb-sem-qwen-csqa"
PROJ_NAME="$EXP_NAME"
SAVE_DIR="./checkpoints/qwen_csqa"

mkdir -p $SAVE_DIR

# Launch the training script using accelerate
accelerate launch --config_file acc_config.yaml ../main.py \
  --train_file "$TRAIN_FILE" \
  --valid_file "$VALID_FILE" \
  --test_file "$TEST_FILE" \
  --encoder_model_name "$ENCODER_MODEL" \
  --decoder_model_name "$DECODER_MODEL" \
  --tokenizer_model_name "$TOKENIZER_MODEL" \
  --per_device_batch_size "$BATCH_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --lr "$LEARNING_RATE" \
  --max_length "$MAX_LENGTH" \
  --num_workers "$NUM_WORKERS" \
  --proj_name "$PROJ_NAME" \
  --exp_name "$EXP_NAME" \
  --save_dir "$SAVE_DIR" \
  --share_param "$SHARE_PARAM" 