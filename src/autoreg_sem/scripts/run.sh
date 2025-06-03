#!/bin/bash
# Example script to launch training with Accelerate on 4 GPUs
export TOKENIZERS_PARALLELISM=false

ENCODER_PATH="PUT YOUR ENCODER PATH HERE"
LATENT_MODEL_PATH="SFT_PATH"
DECODER_PATH=="PUT YOUR DECODER PATH HERE"

# for i in {30..88} do
TOKENIZER_PATH="gpt2" 

# MAKE SURE YOU CHANGE THE DATA PATH.
TRAIN_FILE="../../../data/gsm8k/train.json"
EVAL_FILE="../../../data/gsm8k/valid.json"
TEST_FILE="../../../data/gsm8k/test.json"

BATCH_SIZE=128
NUM_EPOCHS=200
LR=5e-4

EXP_NAME="EXP_NAME"
PROJ_NAME="PROJ_NAME"
SAVE_DIR="SAVE_DIR"
WANDB_KEY="YOUR_WANDB_API_KEY"  # Set your Wandb API key here
WANDB_ENTITY="YOUR_WANDB_ENTITY"  # Set your Wandb entity (username or team name) here

task="gsm8k" # either "gsm8k" or "csqa" (for other tasks than gsm8k)

accelerate launch --config_file acc_config.yaml ../main.py \
  --encoder_path "${ENCODER_PATH}" \
  --latent_model_path "${LATENT_MODEL_PATH}" \
  --decoder_path "${DECODER_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --eval_file "${EVAL_FILE}" \
  --test_file "${TEST_FILE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --lr "${LR}" \
  --proj_name "${PROJ_NAME}" \
  --exp_name "${EXP_NAME}" \
  --save_dir "${SAVE_DIR}" \
  --task "${task}" \
  --wandb_key "${WANDB_KEY}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --freeze \
  --use_cont