#!/bin/bash
# Example script to launch training with Accelerate
export TOKENIZERS_PARALLELISM=false

# Get the epoch number from command line argument
# EPOCH_NUM=$1
# PORT=$2
# if [ -z "$EPOCH_NUM" ] || [ -z "$PORT" ]; then
#     echo "Usage: $0 <epoch_num> <port>"
#     exit 1
# fi


# COMPLETE_MODEL_PATH="/home/hyeonbin/Pred-Sent/src/autoreg_ctx/scripts/checkpoints/epoch_${EPOCH_NUM}"
ENCODER_PATH="/home/work/tmp_hyeonbin/pred-sent/models/trained/qwen_csqa_emb_ctx/encoder2"
DECODER_PATH="/home/work/tmp_hyeonbin/pred-sent/models/trained/qwen_csqa_emb_ctx/decoder2"
LATENT_MODEL_PATH="/home/work/tmp_hyeonbin/pred-sent/models/trained/qwen_csqa_cot/best"
# COMPLETE_MODEL_PATH="/home/work/tmp_hyeonbin/pred-sent/src/autoreg_ctx/scripts/checkpoints_re/epoch_200"
TOKENIZER_PATH="/home/work/tmp_hyeonbin/pred-sent/models/trained/qwen_csqa_cot/best"

# MAKE SURE YOU CHANGE THE DATA PATH.
TRAIN_FILE="/home/work/tmp_hyeonbin/pred-sent/data/csqa/train.json"
EVAL_FILE="/home/work/tmp_hyeonbin/pred-sent/data/csqa/valid.json"
TEST_FILE="/home/work/tmp_hyeonbin/pred-sent/data/csqa/test.json"

# Make sure you use large loss for contrastive learning.
BATCH_SIZE=128
NUM_EPOCHS=300
LR=1e-4

EXP_NAME="Qwen_3_0.6B_AUTOREG_CTX_CSQA_CONT_WEIGHT_1_BS_64_SINGLE_GPU_LR_1e-4_LOSS_TYPE_KL_ALPHA_0.5_TEMP_1.0_w_aussian_noise_0.5"
PROJ_NAME="Qwen_3_0.6B_AUTOREG_CTX_CSQA"
SAVE_DIR="/home/work/tmp_hyeonbin/pred-sent/src/autoreg_ctx/scripts/checkpoints_re"
WANDB_KEY="49e3bf1d97a7148ae772622876fd9ac8b08ce60e"
WANDB_ENTITY="hbin0701"

task="csqa"
LOSS_TYPE=${LOSS_TYPE:-"kl"}
KL_ALPHA=${KL_ALPHA:-0.5}
KL_TEMPERATURE=${KL_TEMPERATURE:-1.0}

accelerate launch --main_process_port 38333 --config_file /home/work/tmp_hyeonbin/pred-sent/src/autoreg_ctx/scripts/acc_config.yaml /home/work/tmp_hyeonbin/pred-sent/src/autoreg_ctx/main.py \
  --encoder_path "${ENCODER_PATH}" \
  --decoder_path "${DECODER_PATH}" \
  --latent_model_path "${LATENT_MODEL_PATH}" \
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
  --loss_type "${LOSS_TYPE}" \
  --kl_alpha "${KL_ALPHA}" \
  --kl_temperature "${KL_TEMPERATURE}" \
  --wandb_key "${WANDB_KEY}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --freeze \
  --use_cont
  