export WANDB_API_KEY="WANDB_API_KEY" # put your wandb key here
export WANDB_PROJECT="WANDB_PROJECT" # put your project name here
export WANDB_ENTITY="WANDB_ENTITY" # put your wandb id here.


model_name_or_path="SFT_MODEL_PATH"
save_dir="SAVE_DIR"
export WANDB_NAME="WANDB_NAME"   

data_dir="DATA_DIR"
# lr: 1e-6 for Mistral and 1e-5 for Others.

accelerate launch \
  --config_file ./config.yaml \
  --main_process_port=40999 \
  ../train_generator.py \
  --model_name_or_path ${model_name_or_path} \
  --data_dir ${data_dir} \
  --target_set train \
  --save_dir ${save_dir} \
  --num_train_epoches 30 \
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
  --save_model_only True