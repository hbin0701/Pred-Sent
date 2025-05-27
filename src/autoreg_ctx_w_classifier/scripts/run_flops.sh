#!/bin/bash
# Example script to launch training with Accelerate on 4 GPUs
export TOKENIZERS_PARALLELISM=false

ENCODER_PATH="ENCODER_PATH/encoder_model"
DECODER_PATH="DECODER_PATH/decoder_model"
TRANSLATOR_PATH="TRANSLATOR_PATH/translator_model"

# Or write this to override previous 3 models.
COMPLETE_MODEL_PATH="COMPLETE_MODEL_PATH"

# for i in {30..88} do
TOKENIZER_PATH="gpt2"

# for now, make it separate. then we make it two the same model or something.

TRAIN_FILE="../../../data/csqa/train.json"
EVAL_FILE="../../../data/csqa/valid.json"
TEST_FILE="../../../data/csqa/test.json"

# Make sure you use large loss for contrastive learning.
BATCH_SIZE=1
NUM_EPOCHS=300
LR=5e-4

PROJ_NAME="PROJ_ANME" 
EXP_NAME="EXP_NAME"
SAVE_DIR="DIR_TO_SAVE_MODEL"

task="csqa"

CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file acc_config.yaml ../main.py \
  --encoder_path "${ENCODER_PATH}" \
  --decoder_path "${DECODER_PATH}" \
  --translator_path "${TRANSLATOR_PATH}" \
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
  --freeze \
  --use_cont \
  --complete_model_path "${COMPLETE_MODEL_PATH}"  # disable this if you want to use the encoder, decoder, translator model.
  # done