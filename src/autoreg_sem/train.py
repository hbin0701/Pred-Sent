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

def train(args, model, dataset_train, eval_dataloader, test_dataloader, optimizer, lr_scheduler, device, num_epochs=3, accelerator=None, save_dir="./checkpoints/stage2"):
    model.train()
    if accelerator.is_main_process:
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
            wandb.init(
                project=args.proj_name, 
                entity=args.wandb_entity, 
                name=args.exp_name, 
                config=vars(args)
            )
        
    os.makedirs(save_dir, exist_ok=True)
    step = 0

    for epoch in range(num_epochs):
        total_loss_val, total_ce, total_cont = 0.0, 0.0, 0.0
        dataset_train.processed_data = dataset_train.processed_data.shuffle(seed=epoch)
        train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, model.tokenizer))
        train_dataloader = accelerator.prepare(train_dataloader)

        for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True, leave=True):

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

            accelerator.backward(total_loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            total_loss_val += total_loss.item()
            total_ce += ce_loss.item()
            total_cont += cont_loss.item()
            step += 1

            clear_gpu_cache_if_needed(device, threshold=0.7)

            if accelerator.is_main_process:
                wandb.log({
                    "train/step_loss": total_loss.item(), 
                    "train/ce_loss": ce_loss.item(), 
                    "train/cont_loss": cont_loss.item()
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

        if (epoch + 1) % 10 == 0:
            if accelerator.is_main_process:
                # Save
                save_path = os.path.join(save_dir, f"epoch_{epoch+1}")
                model.save_model(save_path)
                print(f"Saved checkpoint at epoch {epoch+1}")

                unwrapped_model = accelerator.unwrap_model(model)   

                train_dataset = dataset_train
                eval_dataset = eval_dataloader.dataset
                test_dataset = test_dataloader.dataset

                unwrapped_train_dataloader = DataLoader(
                    train_dataset, 
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=lambda x: collate_fn(x, unwrapped_model.tokenizer)
                )

                unwrapped_eval_dataloader = DataLoader(
                    eval_dataset, 
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=lambda x: collate_fn(x, unwrapped_model.tokenizer)
                )
                
                unwrapped_test_dataloader = DataLoader(
                    test_dataset, 
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=lambda x: collate_fn(x, unwrapped_model.tokenizer)
                )
                
                eval_metrics = evaluate(unwrapped_model, unwrapped_eval_dataloader, device, accelerator)
                eval_loss = eval_metrics['total_loss']
            
                # Log evaluation metrics
                eval_log = {"eval/total_loss": eval_loss, "eval/ce_loss": eval_metrics.get('ce_loss', 0.0)}
                
                # Add L1 loss to eval logs if available
                if 'l1_loss' in eval_metrics and hasattr(model, "use_l1_loss") and model.use_l1_loss:
                    eval_log["eval/l1_loss"] = eval_metrics['l1_loss']
                
                wandb.log(eval_log)
                print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.6f} | Eval Loss: {eval_loss:.6f}")
            
                print(test(unwrapped_model, unwrapped_train_dataloader, device, accelerator, step, mode="train", MAX_SAMPLES=len(test_dataset)))
                print(test(unwrapped_model, unwrapped_eval_dataloader, device, accelerator, step, mode="eval", MAX_SAMPLES=len(test_dataset)))
                print(test(unwrapped_model, unwrapped_test_dataloader, device, accelerator, step, mode="test", MAX_SAMPLES=len(test_dataset)))
                
            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()

    wandb.finish()

def test(model, dataloader, device, accelerator, step, mode="train", MAX_SAMPLES=1500):
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
            # import pdb; pdb.set_trace()

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
    
    # Create logs directory if it doesn't exist
    # os.makedirs("logs", exist_ok=True)
    
    # # Append to the jsonl file
    # with open(f"logs/{mode}_log_{model.ep}.jsonl", "a") as f:
    #     f.write(json.dumps(metrics_to_save) + "\n")

    return log_dict

def evaluate(model, eval_dataloader, device, accelerator):
    model.eval()
    total_eval_loss = 0.0
    total_eval_ce_loss = 0.0
    total_eval_cont_loss = 0.0
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
            n_batches += 1

    avg_total_loss = total_eval_loss / n_batches
    avg_ce_loss = total_eval_ce_loss / n_batches
    avg_cont_loss = total_eval_cont_loss / n_batches

    model.train()

    if accelerator.is_main_process:
        wandb.log({
            "eval/total_loss": avg_total_loss,
            "eval/ce_loss": avg_ce_loss,
            "eval/cont_loss": avg_cont_loss
        })

    return {
        "total_loss": avg_total_loss,
        "ce_loss": avg_ce_loss,
        "cont_loss": avg_cont_loss
    }
