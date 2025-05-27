# running next step predition#!/bin/bash

# Set environment variables
export TOKENIZERS_PARALLELISM=false

# Define argument variables
TRAIN_FILE="../../../../data/gsm8k/train.json"
VALID_FILE="../../../../data/gsm8k/valid.json"
TEST_FILE="../../../../data/gsm8k/test.json"

# Define model paths
ENCODER1_MODEL="EMB_SEM_MODEL_PATH" 
DECODER1_MODEL="EMB_SEM_MODEL_PATH"
# LEAVE THESE EMPTY, if you will only do CTX without IB.

ENCODER2_MODEL="SFT_MODEL_PATH"
DECODER2_MODEL="SFT_MODEL_PATH"
TOKENIZER_MODEL="gpt2"

SHARE_PARAM="True"

# Training parameters
BATCH_SIZE=128
NUM_EPOCHS=30
LEARNING_RATE="5e-4"
MAX_LENGTH=512
NUM_WORKERS=1
CONTRASTIVE_WEIGHT=0.5
TEMPERATURE=0.1

PROJ_NAME="PROJ_NAME"
EXP_NAME="EXP_NAME"
SAVE_DIR="/mnt/nas/hyeonbin/LS_MODELS/gsm8k/stage1.6/gpt2"
TASK="gsm8k"

# [NOTE!] Use "use_cont" if you want to do CTX-IB, eliminate it otherwise.
accelerate launch --config_file acc_config.yaml ../main.py \
  --train_file "$TRAIN_FILE" \
  --valid_file "$VALID_FILE" \
  --test_file "$TEST_FILE" \
  --encoder1_model_name "$ENCODER1_MODEL" \
  --decoder1_model_name "$DECODER1_MODEL" \
  --encoder2_model_name "$ENCODER2_MODEL" \
  --decoder2_model_name "$DECODER2_MODEL" \
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
  --use_cont \
  --contrastive_weight "$CONTRASTIVE_WEIGHT" \
  --temperature "$TEMPERATURE" \
  --task "$TASK"