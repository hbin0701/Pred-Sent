import os
import wandb
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collate import collate_fn
from utils import clear_gpu_cache_if_needed, pprint
import json
import time

def mock_train(args, model, dataset_train, eval_dataloader, test_dataloader, optimizer, lr_scheduler, device, num_epochs=3, accelerator=None, save_dir="./checkpoints/stage2"):
    """
    Mock training function that only runs inference on test data.
    """
    model.eval()  # Set to eval mode since we're only doing inference
    if accelerator.is_main_process:
        wandb.login(key="a8d82c69a1bf33d1957f8d524e50257d29fd9379")
        wandb.init(project=args.proj_name, entity="byeongguk", name=args.exp_name, config=vars(args))
    
    # Run inference on test data
    print("\nRunning inference on test data...")
    test_results = test(model, test_dataloader, device, accelerator, step=0, mode="test")
    
    # # Run inference on eval data if available
    # if eval_dataloader is not None:
    #     print("\nRunning inference on eval data...")
    #     eval_results = test(model, eval_dataloader, device, accelerator, step=0, mode="eval")
    
    wandb.finish()
    return test_results

def test(model, dataloader, device, accelerator, step, mode="test", MAX_SAMPLES=1500):
    model.eval()
    total_acc = 0
    total_pos_acc = 0
    total_prev_pos_acc = 0

    total_semi_acc = 0
    total_semi_pos_acc = 0
    total_semi_prev_pos_acc = 0

    total_samples = 0
    total_steps = 0

    total_gt_ls_acc = 0
    total_gt_non_ls_acc = 0
    
    total_stage_acc = {}  # to accumulate keys like "stage0_acc", "stage1_acc", etc.
    total_stage_num = {}

    # FLOPs tracking
    total_acc_flops = 0
    total_semi_flops = 0

    CURR_SAMPLES = 0

    for idx, batch in tqdm(enumerate(dataloader), desc="Testing", leave=False):
        encoder_input_ids = batch["encoder_input_ids"].to(device)
        encoder_attention_mask = batch["encoder_attention_mask"].to(device)
        steps_input_ids = batch["steps_input_ids"].to(device)
        steps_attention_mask = batch["steps_attention_mask"].to(device)
        steps_valid_mask = batch["steps_valid_mask"].to(device)
        
        CURR_SAMPLES += steps_input_ids.size(0)
        if CURR_SAMPLES > MAX_SAMPLES:
            break

        with torch.no_grad():
            metrics = model.test(encoder_input_ids, encoder_attention_mask, steps_input_ids, steps_attention_mask, steps_valid_mask, step, mode)

        # In your batch loop:
        batch_size = encoder_input_ids.size(0)
        total_acc += metrics["acc"]
        total_pos_acc += metrics["pos_acc"]
        total_prev_pos_acc += metrics["prev_pos_acc"]

        total_semi_acc += metrics["semi_acc"]
        total_semi_pos_acc += metrics["semi_pos_acc"]
        total_semi_prev_pos_acc += metrics["semi_prev_pos_acc"]

        total_steps += metrics["total_steps"]
        total_samples += batch_size
        
        total_gt_ls_acc += metrics["gt_ls_acc"]
        total_gt_non_ls_acc += metrics["gt_non_ls_acc"]

        # Accumulate FLOPs
        total_acc_flops += metrics["acc_flops"]
        total_semi_flops += metrics["semi_flops"]

        # Accumulate stage metrics
        for key, value in metrics.items():
            if key.startswith("stage") and key.endswith("_acc"):
                if key not in total_stage_acc:
                    total_stage_acc[key] = 0
                total_stage_acc[key] += value

            if key.startswith("stage") and key.endswith("total"):
                if key not in total_stage_num:
                    total_stage_num[key] = 0
                total_stage_num[key] += value

    print("Total Samples: ", total_samples)
    
    # After processing all batches, compute overall metrics:
    overall_acc = total_acc / total_samples * 100
    overall_pos_acc = total_pos_acc / total_samples * 100
    overall_prev_pos_acc = total_prev_pos_acc / total_samples * 100

    overall_semi_acc = total_semi_acc / total_samples * 100
    overall_semi_pos_acc = total_semi_pos_acc / total_samples * 100
    overall_semi_prev_pos_acc = total_semi_prev_pos_acc / total_samples * 100

    # Calculate average FLOPs per sample
    avg_acc_flops = total_acc_flops / total_samples
    avg_semi_flops = total_semi_flops / total_samples

    # For gt_non_ls, note the denominator is (total_steps - total_samples) as in your code.
    if total_samples == total_steps:
        print("Total Steps Equal to Total Samples!")
        total_steps += total_samples
    
    try:
        overall_gt_ls_acc = total_gt_ls_acc / total_samples * 100        
        overall_gt_non_ls_acc = total_gt_non_ls_acc / (total_steps - total_samples) * 100
        overall_gt_acc = (total_gt_ls_acc + total_gt_non_ls_acc) / total_steps * 100        
        overall_stage_acc = { key: (total_stage_acc[key] / total_stage_num[key.replace("acc", "total")] * 100) for key in total_stage_acc }
    except:
        overall_gt_ls_acc = 0
        overall_gt_non_ls_acc = 0
        overall_gt_acc = 0
        overall_stage_acc = {}

    # Prepare the log dictionary
    log_dict = {
        f"{mode}_acc/acc/overall": overall_acc, 
        f"{mode}_acc/acc/pos": overall_pos_acc, 
        f"{mode}_acc/acc/prev_pos": overall_prev_pos_acc, 
        
        f"{mode}_acc/gt/ls": overall_gt_ls_acc,
        f"{mode}_acc/gt/non_ls": overall_gt_non_ls_acc,
        f"{mode}_acc/gt/overall": overall_gt_acc,

        f"{mode}_acc/semi/overall": overall_semi_acc,
        f"{mode}_acc/semi/pos": overall_semi_pos_acc,
        f"{mode}_acc/semi/prev_pos": overall_semi_prev_pos_acc,

        f"{mode}_flops/acc": avg_acc_flops,
        f"{mode}_flops/semi": avg_semi_flops,
    }

    # Add stage accuracies to the log dictionary:
    for key, value in overall_stage_acc.items():
        log_dict[f"{mode}_acc/stages/{key}"] = value

    # Log everything with wandb:
    wandb.log(log_dict)
    
    # Print results
    print(f"\n{mode.upper()} Results:")
    print(f"Accuracy: {overall_acc:.2f}%")
    print(f"Position Accuracy: {overall_pos_acc:.2f}%")
    print(f"Previous Position Accuracy: {overall_prev_pos_acc:.2f}%")
    print(f"\nSemi-Accuracy: {overall_semi_acc:.2f}%")
    print(f"Semi-Position Accuracy: {overall_semi_pos_acc:.2f}%")
    print(f"Semi-Previous Position Accuracy: {overall_semi_prev_pos_acc:.2f}%")
    print(f"\nFLOPs per sample:")
    print(f"cont flops: {avg_acc_flops:,.0f}")
    print(f"cont flops: {avg_acc_flops / 1e9:.2f} GFLOPs")
    print(f"Semi flops: {avg_semi_flops:,.0f}")
    print(f"Semi flops: {avg_semi_flops / 1e9:.2f} GFLOPs \n\n")
    print("===========================")
    print("============================")
    
    return log_dict
