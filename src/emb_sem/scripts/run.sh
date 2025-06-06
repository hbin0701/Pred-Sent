#!/bin/bash

# Set environment variables
export TOKENIZERS_PARALLELISM=false

# Define argument variables
TRAIN_FILE="../../../data/gsm8k/train.json"
VALID_FILE="../../../data/gsm8k/valid.json"
TEST_FILE="../../../data/gsm8k/test.json"

ENCODER_MODEL="SFT_MODEL_PATH"
DECODER_MODEL="SFT_MODEL_PATH"
TOKENIZER_MODEL="gpt2"

SHARE_PARAM="True"

BATCH_SIZE=256
NUM_EPOCHS=3
LEARNING_RATE="5e-4"
MAX_LENGTH=512
NUM_WORKERS=1

PROJ_NAME="PROJ_NAME"
EXP_NAME="EXP_NAME"
SAVE_DIR="SAVE_DIR"

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
