#!/bin/bash

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="49e3bf1d97a7148ae772622876fd9ac8b08ce60e"
export WANDB_ENTITY="hbin0701"

# Define argument variables
TRAIN_FILE="/home/work/tmp_hyeonbin/pred-sent/data/csqa/train.json"
VALID_FILE="/home/work/tmp_hyeonbin/pred-sent/data/csqa/valid.json"
TEST_FILE="/home/work/tmp_hyeonbin/pred-sent/data/csqa/test.json"

# Use the checkpoint that will be created by emb_sem training (epoch 5)
ENCODER_MODEL1="/home/work/tmp_hyeonbin/pred-sent/src/emb_sem/checkpoints/csqa/2/encoder"
DECODER_MODEL1="/home/work/tmp_hyeonbin/pred-sent/src/emb_sem/checkpoints/csqa/2/decoder"
ENCODER_MODEL2="/home/work/tmp_hyeonbin/pred-sent/models/trained/qwen_csqa_cot/best"
DECODER_MODEL2="/home/work/tmp_hyeonbin/pred-sent/models/trained/qwen_csqa_cot/best"
TOKENIZER_MODEL="/home/work/tmp_hyeonbin/pred-sent/models/trained/qwen_csqa_cot/best"

SHARE_PARAM="True"
TASK="csqa"
BATCH_SIZE=64
NUM_EPOCHS=300
LEARNING_RATE="1e-5"
MAX_LENGTH=
NUM_WORKERS=1
USE_LORA="True"

EXP_NAME="emb-ctx-csqa"
PROJ_NAME="Qwen_3_0.6B_EMB_CTX_CSQA"
SAVE_DIR="./checkpoints/csqa_re"

mkdir -p $SAVE_DIR

# Launch the training script using accelerate
accelerate launch --config_file /home/work/tmp_hyeonbin/pred-sent/src/emb_ctx/scripts/acc_config.yaml /home/work/tmp_hyeonbin/pred-sent/src/emb_ctx/main.py \
  --train_file "$TRAIN_FILE" \
  --valid_file "$VALID_FILE" \
  --test_file "$TEST_FILE" \
  --encoder1_model_name "$ENCODER_MODEL1" \
  --decoder1_model_name "$DECODER_MODEL1" \
  --encoder2_model_name "$ENCODER_MODEL2" \
  --decoder2_model_name "$DECODER_MODEL2" \
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
  --task "$TASK"