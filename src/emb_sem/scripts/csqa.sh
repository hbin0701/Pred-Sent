#!/bin/bash

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="49e3bf1d97a7148ae772622876fd9ac8b08ce60e"
export WANDB_ENTITY="hbin0701"

# Define argument variables
TRAIN_FILE="fw-edu/train"
VALID_FILE="/home/work/tmp_hyeonbin/pred-sent/data/csqa/test.json"
TEST_FILE="/home/work/tmp_hyeonbin/pred-sent/data/csqa/test.json"

ENCODER_MODEL="/home/work/tmp_hyeonbin/pred-sent/models/trained/qwen_csqa_cot/best"
DECODER_MODEL="/home/work/tmp_hyeonbin/pred-sent/models/trained/qwen_csqa_cot/best"
# ENCODER_MODEL="/home/work/tmp_hyeonbin/pred-sent/src/checkpoints/csqa_/4/encoder"
# DECODER_MODEL="/home/work/tmp_hyeonbin/pred-sent/src/checkpoints/csqa_/4/decoder"
TOKENIZER_MODEL="/home/work/tmp_hyeonbin/pred-sent/models/trained/qwen_csqa_cot/best"

SHARE_PARAM="True"
USE_LORA="True"
BATCH_SIZE=256
NUM_EPOCHS=5
LEARNING_RATE="5e-4"
MAX_LENGTH=512
NUM_WORKERS=1

EXP_NAME="emb-sem-csqa"
PROJ_NAME="Qwen_3_0.6B_EMB_SEM_CSQA"
SAVE_DIR="./checkpoints/csqa"

mkdir -p $SAVE_DIR

# Launch the training script using accelerate
accelerate launch --config_file /home/work/tmp_hyeonbin/pred-sent/src/emb_sem/scripts/acc_config.yaml /home/work/tmp_hyeonbin/pred-sent/src/emb_sem/main.py \
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