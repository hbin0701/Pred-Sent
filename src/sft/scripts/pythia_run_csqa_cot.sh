#!/bin/bash

export WANDB_API_KEY="49e3bf1d97a7148ae772622876fd9ac8b08ce60e"
export WANDB_PROJECT="LATENT_STEP_REBUTTAL"
export WANDB_ENTITY="hbin0701"
export WANDB_NAME="pythia_sft_csqa_cot"   

model_name_or_path="/home/hyeonbin/Pred-Sent/models/pythia-160m"
save_dir="/mnt/nas/hyeonbin/LS_MODELS/rebuttal/pythia/csqa-sft-cot"

data_dir="/home/hyeonbin/Pred-Sent/data/csqa"
MODE="cot"

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
  --config_file src/sft/scripts/config.yaml \
  --main_process_port=40999 \
  src/sft/train_generator.py \
  --model_name_or_path ${model_name_or_path} \
  --data_dir ${data_dir} \
  --target_set train \
  --save_dir ${save_dir} \
  --num_train_epoches 20 \
  --save_strategy epoch \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing True \
  --learning_rate 1e-4 \
  --weight_decay 0 \
  --lr_scheduler_type "constant" \
  --warmup_steps 0 \
  --save_best False \
  --save_total_limit 50 \
  --logging_dir ./wandb \
  --logging_steps 8 \
  --seed 42 \
  --save_model_only True \
  --mode ${MODE} 