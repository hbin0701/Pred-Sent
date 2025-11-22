#!/bin/bash

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="49e3bf1d97a7148ae772622876fd9ac8b08ce60e"
export WANDB_ENTITY="hbin0701"

# Define argument variables
TRAIN_FILE="../../../data/gsm8k/train.json"
VALID_FILE="../../../data/gsm8k/valid.json"
TEST_FILE="../../../data/gsm8k/test.json"

# Use the checkpoint that will be created by emb_sem training (epoch 5)
ENCODER_MODEL="../../emb_sem/checkpoints/gsm8k/5/encoder"
DECODER_MODEL="../../emb_sem/checkpoints/gsm8k/5/decoder"
TOKENIZER_MODEL="/home/hyeonbin/Pred-Sent/models/SmolLM-135M"

SHARE_PARAM="True"

BATCH_SIZE=128
NUM_EPOCHS=30
LEARNING_RATE="5e-4"
MAX_LENGTH=512
NUM_WORKERS=1

EXP_NAME="emb-ctx-gsm8k"
PROJ_NAME="$EXP_NAME"
SAVE_DIR="./checkpoints/gsm8k"

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
  --share_param "$SHARE_PARAM" \
  --use_cont