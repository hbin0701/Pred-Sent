#!/bin/bash
#=============================================================================
# Context-based Embedding Training Script
#=============================================================================

# Environment settings
export TOKENIZERS_PARALLELISM=false

#=============================================================================
# Data paths
#=============================================================================
TRAIN_FILE="../../../data/gsm8k/train.json"
VALID_FILE="../../../data/gsm8k/valid.json"
TEST_FILE="../../../data/gsm8k/test.json"
TASK="gsm8k"

#=============================================================================
# Model configuration
#=============================================================================
# SET THESE AS YOUR SEMANTIC EMBEDDING MODELS.  
# (IF YOU WANT TO DO CTX-B / No-Contrastive, YOU CAN IGNORE THESE. MAKE SURE TO REMOVE "--use_cont" FLAG.)
ENCODER1_MODEL="EMB_SEM_MODEL_PATH" 
DECODER1_MODEL="EMB_SEM_MODEL_PATH"

# SET AS SFT MODELS.
ENCODER2_MODEL="SFT_MODEL_PATH"
DECODER2_MODEL="SFT_MODEL_PATH"

# Tokenizer
TOKENIZER_MODEL="gpt2"

# Model sharing parameters
SHARE_PARAM="True"

#=============================================================================
# Training hyperparameters
#=============================================================================
BATCH_SIZE=128
NUM_EPOCHS=30
LEARNING_RATE="5e-4"
MAX_LENGTH=512
NUM_WORKERS=1
CONTRASTIVE_WEIGHT=0.5
TEMPERATURE=0.1
#=============================================================================
# Experiment tracking
#=============================================================================
# WANDB SETTING
PROJ_NAME="PROJ_NAME"
EXP_NAME="EXP_NAME"
WANDB_KEY="YOUR_WANDB_API_KEY"
WANDB_ENTITY="YOUR_WANDB_ENTITY"

# SAVE_DIR
SAVE_DIR="SAVE_DIR"

#=============================================================================
# Launch training
#=============================================================================
# Note: Remove "--use_cont" flag if you don't want to use CTX-C
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
  --task "$TASK" \
  --wandb_key "$WANDB_KEY" \
  --wandb_entity "$WANDB_ENTITY"