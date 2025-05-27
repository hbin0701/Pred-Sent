import os
import wandb
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collate import collate_fn
from utils import clear_gpu_cache_if_needed, pprint
import json
import time

def make_new_dataloader(dataset, tokenizer, args, eval_accelerator):

    dl = DataLoader(
        dataset,
        batch_size=64,
        collate_fn=lambda x: collate_fn(x, tokenizer),
        drop_last=False,  # Don't drop uneven batches
    )
    
    return eval_accelerator.prepare(dl)

def train(args, model, dataset_train, eval_dataloader, test_dataloader, optimizer, lr_scheduler, device, num_epochs=3, accelerator=None, save_dir="./checkpoints/stage2", eval_accelerator=None, noise=None, noise_type=None):
    model.train()
    if accelerator.is_main_process:
        wandb.login(key="PUT_YOUR_KEY_HERE")
        wandb.init(project=args.proj_name, entity="hbin0701", name=args.exp_name, config=vars(args))
        
    os.makedirs(save_dir, exist_ok=True)
    step = 0

    for epoch in range(num_epochs):
        total_loss_val, total_ce, total_cont = 0.0, 0.0, 0.0
        dataset_train.processed_data = dataset_train.processed_data.shuffle(seed=epoch)
        train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, model.tokenizer))
        train_dataloader = accelerator.prepare(train_dataloader)

        for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True, leave=True):
            break
            encoder_input_ids = batch["encoder_input_ids"].to(device)            
            encoder_attention_mask = batch["encoder_attention_mask"].to(device)
            steps_tokens = batch["steps_input_ids"].to(device)
            steps_attention_mask = batch["steps_attention_mask"].to(device)
            steps_valid_mask = batch["steps_valid_mask"].to(device)

            optimizer.zero_grad()
            loss_dict = model(encoder_input_ids, encoder_attention_mask, steps_tokens, steps_attention_mask, steps_valid_mask, accelerator)
            total_loss = loss_dict["total_loss"]
            ce_loss = loss_dict["ce_loss"]
            cont_loss = loss_dict["cont_loss"]
            distance_loss = loss_dict["distance_loss"]
            distance_loss1 = loss_dict["distance_loss1"]
            distance_loss2 = loss_dict["distance_loss2"]
            mse_loss = loss_dict["mse_loss"]

            accelerator.backward(total_loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            torch.cuda.empty_cache()

            total_loss_val += total_loss.item()
            total_ce += ce_loss.item()
            total_cont += cont_loss.item()
            step += 1

            # clear_gpu_cache_if_needed(device, threshold=0.7)
            # torch.cuda.empty_cache()
            
            if accelerator.is_main_process:
                wandb.log({
                    "train/step_loss": total_loss.item(), 
                    "train/ce_loss": ce_loss.item(), 
                    "train/cont_loss": cont_loss.item(),
                    "train/distance_loss": distance_loss.item(),
                    "train/distance_loss1": distance_loss1.item(),
                    "train/distance_loss2": distance_loss2.item(),
                    "train/mse_loss": mse_loss.item()
                })

        avg_loss = total_loss_val / len(train_dataloader)
        avg_ce = total_ce / len(train_dataloader)
        avg_cont = total_cont / len(train_dataloader)
        
        if accelerator.is_main_process:
            wandb.log({
                "train/epoch_loss": avg_loss,  
                "train/epoch_ce_loss": avg_ce, 
                "train/epoch_cont_loss": avg_cont
            })

        if (epoch + 1) % 1 == 0:
            # if accelerator.is_main_process:
                # Save
                # if (epoch + 1) % 10 == 0:
                #     save_path = os.path.join(save_dir, f"epoch_{epoch+1}")
                #     model.save_model(save_path)
                #     print(f"Saved checkpoint at epoch {epoch+1}")
            
            # Create eval and test dataloaders with distributed samplers for evaluatio            
            # Run evaluation on test dataset
            
            new_train_dataloader = make_new_dataloader(dataset_train, model.tokenizer, args, eval_accelerator)
            test_dataloader = make_new_dataloader(test_dataloader.dataset, model.tokenizer, args, eval_accelerator)

            # eval_metrics = evaluate(model, test_dataloader, device, accelerator)
            
            # Run evaluation on eval dataset
            # eval_dataloader = make_new_dataloader(eval_dataloader.dataset, unwrapped_model.tokenizer, args, eval_accelerator)
            # eval_metrics = evaluate(model, eval_dataloader, device, accelerator)
            
            # if accelerator.is_main_process:
            #     eval_loss = eval_metrics['total_loss']
            #     # Log evaluation metrics
            #     eval_log = {"eval/total_loss": eval_loss, "eval/ce_loss": eval_metrics.get('ce_loss', 0.0)}
                
            #     # Add L1 loss to eval logs if available
            #     if 'l1_loss' in eval_metrics and hasattr(unwrapped_model, "use_l1_loss") and unwrapped_model.use_l1_loss:
            #         eval_log["eval/l1_loss"] = eval_metrics['l1_loss']
                
            #     wandb.log(eval_log)
            #     print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.6f} | Eval Loss: {eval_loss:.6f}")
                
            # Use the wrapped model for test metrics calculation as requested
            # test(model, new_train_dataloader, device, accelerator, step, mode="train")
            test(model, test_dataloader, device, accelerator, step, mode="test", noise=noise, noise_type=noise_type)
            
            # del new_train_dataloader
            # del test_dataloader

            import gc; gc.collect()
            torch.cuda.empty_cache()


    wandb.finish()

def test(model, dataloader, device, accelerator, step, mode="train", MAX_SAMPLES=800, noise=None, noise_type=None):
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
    
    # Initialize empty dictionaries for stage metrics
    total_stage_acc = {}  
    total_stage_num = {}

    CURR_SAMPLES = 0
    
    # Process batches distributed across GPUs
    for idx, batch in tqdm(enumerate(dataloader), desc="Testing", leave=False):

        if mode == "train" and idx > 5:
            break
        
        encoder_input_ids = batch["encoder_input_ids"].to(device)
        encoder_attention_mask = batch["encoder_attention_mask"].to(device)
        steps_input_ids = batch["steps_input_ids"].to(device)
        steps_attention_mask = batch["steps_attention_mask"].to(device)
        steps_valid_mask = batch["steps_valid_mask"].to(device)
        
        # Use the accelerator's process batch size
        batch_size = encoder_input_ids.size(0)
        CURR_SAMPLES += batch_size
        
        # if CURR_SAMPLES > MAX_SAMPLES:
        #     break

        with torch.no_grad():
            metrics = model.test(encoder_input_ids, encoder_attention_mask, steps_input_ids, steps_attention_mask, steps_valid_mask, step, mode, accelerator, noise=noise, noise_type=noise_type)

        # Accumulate metrics on each GPU
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

    # Gather results from all processes
    gathered_total_acc = accelerator.gather(torch.tensor([total_acc], device=device)).sum().item()
    gathered_total_pos_acc = accelerator.gather(torch.tensor([total_pos_acc], device=device)).sum().item()
    gathered_total_prev_pos_acc = accelerator.gather(torch.tensor([total_prev_pos_acc], device=device)).sum().item()
    gathered_total_semi_acc = accelerator.gather(torch.tensor([total_semi_acc], device=device)).sum().item()
    gathered_total_semi_pos_acc = accelerator.gather(torch.tensor([total_semi_pos_acc], device=device)).sum().item()
    gathered_total_semi_prev_pos_acc = accelerator.gather(torch.tensor([total_semi_prev_pos_acc], device=device)).sum().item()
    gathered_total_steps = accelerator.gather(torch.tensor([total_steps], device=device)).sum().item()
    gathered_total_samples = accelerator.gather(torch.tensor([total_samples], device=device)).sum().item()
    gathered_total_gt_ls_acc = accelerator.gather(torch.tensor([total_gt_ls_acc], device=device)).sum().item()
    gathered_total_gt_non_ls_acc = accelerator.gather(torch.tensor([total_gt_non_ls_acc], device=device)).sum().item()
    
    if mode=="test" and len(dataloader.dataset) != gathered_total_samples:
        raise AssertionError("Dataloader dataset length does not match gathered total samples")
    
    # First, gather all stage keys across processes to ensure all processes know about all keys
    all_stage_acc_keys = list(total_stage_acc.keys())
    all_stage_num_keys = list(total_stage_num.keys())
    
    # Use all_gather to collect keys from all processes
    gathered_acc_keys_list = accelerator.gather_for_metrics([all_stage_acc_keys])
    gathered_num_keys_list = accelerator.gather_for_metrics([all_stage_num_keys])
    
    # Flatten and deduplicate the gathered keys
    all_unique_acc_keys = list(set(key for keys in gathered_acc_keys_list for key in keys))
    all_unique_num_keys = list(set(key for keys in gathered_num_keys_list for key in keys))
    
    # Ensure each process has zeros for keys it doesn't have
    for key in all_unique_acc_keys:
        if key not in total_stage_acc:
            total_stage_acc[key] = 0
    
    for key in all_unique_num_keys:
        if key not in total_stage_num:
            total_stage_num[key] = 0
    
    # Now gather the values for each key
    gathered_stage_acc = {}
    gathered_stage_num = {}
    
    for key in all_unique_acc_keys:
        gathered_stage_acc[key] = accelerator.gather(torch.tensor([total_stage_acc[key]], device=device)).sum().item()
    
    for key in all_unique_num_keys:
        gathered_stage_num[key] = accelerator.gather(torch.tensor([total_stage_num[key]], device=device)).sum().item()
    
    # Only the main process computes and logs the final metrics
    if accelerator.is_main_process:
        # After processing all batches, compute overall metrics:
        overall_acc = gathered_total_acc / gathered_total_samples * 100
        overall_pos_acc = gathered_total_pos_acc / gathered_total_samples * 100
        overall_prev_pos_acc = gathered_total_prev_pos_acc / gathered_total_samples * 100

        overall_semi_acc = gathered_total_semi_acc / gathered_total_samples * 100
        overall_semi_pos_acc = gathered_total_semi_pos_acc / gathered_total_samples * 100
        overall_semi_prev_pos_acc = gathered_total_semi_prev_pos_acc / gathered_total_samples * 100

        # For gt_non_ls, note the denominator is (total_steps - total_samples) as in your code.
        if gathered_total_samples == gathered_total_steps:
            print("Total Steps Equal to Total Samples!")
            gathered_total_steps += gathered_total_samples
        
        try:
            overall_gt_ls_acc = gathered_total_gt_ls_acc / gathered_total_samples * 100        
            overall_gt_non_ls_acc = gathered_total_gt_non_ls_acc / (gathered_total_steps - gathered_total_samples) * 100
            overall_gt_acc = (gathered_total_gt_ls_acc + gathered_total_gt_non_ls_acc) / gathered_total_steps * 100        
            overall_stage_acc = { key: (gathered_stage_acc[key] / gathered_stage_num[key.replace("acc", "total")] * 100) for key in gathered_stage_acc }
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
        }

        # Add stage accuracies to the log dictionary:
        for key, value in overall_stage_acc.items():
            log_dict[f"{mode}_acc/stages/{key}"] = value

        # Log everything with wandb:
        wandb.log(log_dict)
        
        # Save metrics to a jsonl file
        metrics_to_save = {
            "timestamp": time.time(),
            "step": step,
            "acc": {
                "overall": overall_acc,
                "pos": overall_pos_acc,
                "prev_pos": overall_prev_pos_acc
            },
            "semi": {
                "overall": overall_semi_acc,
                "pos": overall_semi_pos_acc,
                "prev_pos": overall_semi_prev_pos_acc
            },
            "gt": {
                "ls": overall_gt_ls_acc,
                "non_ls": overall_gt_non_ls_acc,
                "overall": overall_gt_acc
            },
            "stages": {k: v for k, v in overall_stage_acc.items()}
        }
        
        return log_dict
    return {}

def evaluate(model, eval_dataloader, device, accelerator):
    model.eval()
    total_eval_loss = 0.0
    total_eval_ce_loss = 0.0
    total_eval_cont_loss = 0.0
    total_eval_distance_loss = 0.0
    total_eval_distance_loss1 = 0.0
    total_eval_distance_loss2 = 0.0
    total_eval_mse_loss = 0.0
    n_batches = 0
    
    print("evaluating ... ")
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            encoder_input_ids = batch["encoder_input_ids"].to(device)
            encoder_attention_mask = batch["encoder_attention_mask"].to(device)
            steps_tokens = batch["steps_input_ids"].to(device)
            steps_attention_mask = batch["steps_attention_mask"].to(device)
            steps_valid_mask = batch["steps_valid_mask"].to(device)

            loss_dict = model(encoder_input_ids, encoder_attention_mask, steps_tokens, steps_attention_mask, steps_valid_mask, accelerator)
            total_eval_loss += loss_dict["total_loss"].item()
            total_eval_ce_loss += loss_dict["ce_loss"].item()
            total_eval_cont_loss += loss_dict["cont_loss"].item()
            total_eval_distance_loss += loss_dict["distance_loss"].item()
            total_eval_distance_loss1 += loss_dict["distance_loss1"].item()
            total_eval_distance_loss2 += loss_dict["distance_loss2"].item()
            total_eval_mse_loss += loss_dict["mse_loss"].item()
            n_batches += 1

    # Gather results from all processes
    all_losses = [
        torch.tensor([total_eval_loss], device=device),
        torch.tensor([total_eval_ce_loss], device=device),
        torch.tensor([total_eval_cont_loss], device=device),
        torch.tensor([total_eval_distance_loss], device=device),
        torch.tensor([total_eval_distance_loss1], device=device),
        torch.tensor([total_eval_distance_loss2], device=device),
        torch.tensor([total_eval_mse_loss], device=device),
        torch.tensor([n_batches], device=device)
    ]
    
    for i in range(len(all_losses)):
        all_losses[i] = accelerator.gather(all_losses[i]).sum()
    
    gathered_total_eval_loss, gathered_total_eval_ce_loss, gathered_total_eval_cont_loss, \
    gathered_total_eval_distance_loss, gathered_total_eval_distance_loss1, gathered_total_eval_distance_loss2, \
    gathered_total_eval_mse_loss, gathered_n_batches = [t.item() for t in all_losses]

    # Compute average losses
    avg_total_loss = gathered_total_eval_loss / gathered_n_batches
    avg_ce_loss = gathered_total_eval_ce_loss / gathered_n_batches
    avg_cont_loss = gathered_total_eval_cont_loss / gathered_n_batches
    avg_distance_loss = gathered_total_eval_distance_loss / gathered_n_batches
    avg_distance_loss1 = gathered_total_eval_distance_loss1 / gathered_n_batches
    avg_distance_loss2 = gathered_total_eval_distance_loss2 / gathered_n_batches
    avg_mse_loss = gathered_total_eval_mse_loss / gathered_n_batches

    model.train()

    if accelerator.is_main_process:
        wandb.log({
            "eval/total_loss": avg_total_loss,
            "eval/ce_loss": avg_ce_loss,
            "eval/cont_loss": avg_cont_loss,
            "eval/distance_loss": avg_distance_loss,
            "eval/distance_loss1": avg_distance_loss1,
            "eval/distance_loss2": avg_distance_loss2,
            "eval/mse_loss": avg_mse_loss
        })

    return {
        "total_loss": avg_total_loss,
        "ce_loss": avg_ce_loss,
        "cont_loss": avg_cont_loss,
        "distance_loss": avg_distance_loss,
        "distance_loss1": avg_distance_loss1,
        "distance_loss2": avg_distance_loss2,
        "mse_loss": avg_mse_loss
    }