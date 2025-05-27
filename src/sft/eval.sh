# #!/bin/bash

# data_file="/home/hyeonbin/Latent_Step/data/gsm8k/test.json"

# # Array of available GPUs
# gpus=(4 5 6 7)
# num_gpus=${#gpus[@]}
# total_checkpoints=30

# # Create results directory if it doesn't exist
# mkdir -p "/home/hyeonbin/Latent_Step/src/sft/results"

# # Process checkpoints in batches
# for ((batch_start=1; batch_start<=total_checkpoints; batch_start+=num_gpus)); do
#     # Calculate batch end (ensuring we don't exceed total_checkpoints)
#     batch_end=$((batch_start+num_gpus-1))
#     if [ $batch_end -gt $total_checkpoints ]; then
#         batch_end=$total_checkpoints
#     fi
    
#     echo "Processing batch from checkpoint $batch_start to $batch_end"
    
#     # Launch jobs for this batch
#     for ((i=batch_start; i<=batch_end; i++)); do
#         # Calculate the checkpoint number (i * 1432)
#         checkpoint=$((i * 1432))
#         model_path="/mnt/nas/hyeonbin/LS_MODELS/gsm8k/sft/gpt2-medium/checkpoint-${checkpoint}"
#         result_file="/home/hyeonbin/Latent_Step/src/sft/results/MEDIUM_cot_${checkpoint}.jsonl"
        
#         # Determine which GPU to use for this job
#         gpu_index=$(((i-batch_start) % num_gpus))
#         gpu_id=${gpus[$gpu_index]}
        
#         echo "Running evaluation for checkpoint-${checkpoint} on GPU ${gpu_id}..."
        
#         # Check if model path exists
#         if [ ! -d "$model_path" ]; then
#             echo "Warning: Model path $model_path does not exist. Skipping..."
#             continue
#         fi
        
#         # Run evaluation in background
#         CUDA_VISIBLE_DEVICES=${gpu_id} python eval.py \
#             --data_file "$data_file" \
#             --model "$model_path" \
#             --result_file "$result_file" \
#             --temp 0 &
#     done
    
#     # Wait for all processes in this batch to complete
#     wait
#     echo "Batch $batch_start to $batch_end completed."
# done

# echo "All evaluations completed."

python eval.py --model "/mnt/nas/hyeonbin/LS_MODELS/csqa/sft/csqa_gpt2-large/checkpoint-255" --result_file "/home/hyeonbin/Latent_Step/src/new/stage2.5/sft_CSQA_LARGE.jsonl"

