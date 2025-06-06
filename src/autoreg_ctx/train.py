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

def train(args, model, dataset_train, eval_dataloader, test_dataloader, optimizer, lr_scheduler, device, num_epochs=3, accelerator=None, save_dir="./checkpoints/stage2", eval_accelerator=None):
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
                if (epoch + 1) % 10 == 0:
                    save_path = os.path.join(save_dir, f"epoch_{epoch+1}")
                    model.save_model(save_path)
                    print(f"Saved checkpoint at epoch {epoch+1}")
            
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
            test(model, test_dataloader, device, accelerator, step, mode="test")

            import gc; gc.collect()
            torch.cuda.empty_cache()

    wandb.finish()

def test(model, dataloader, device, accelerator, step, mode="train", MAX_SAMPLES=800):
    model.eval()
    total_cont = 0
    total_cont_pos = 0

    total_disc_acc = 0
    total_disc_pos_acc = 0

    total_samples = 0

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
            metrics = model.test(encoder_input_ids, encoder_attention_mask, steps_input_ids, steps_attention_mask, steps_valid_mask, step, mode, accelerator)

        # Accumulate metrics on each GPU
        total_cont += metrics["cont"]
        total_cont_pos += metrics["cont_pos"]

        total_disc_acc += metrics["disc_acc"]
        total_disc_pos_acc += metrics["disc_pos_acc"]

        total_samples += batch_size

    # Gather results from all processes
    gathered_total_cont = accelerator.gather(torch.tensor([total_cont], device=device)).sum().item()
    gathered_total_cont_pos = accelerator.gather(torch.tensor([total_cont_pos], device=device)).sum().item()
    gathered_total_disc_acc = accelerator.gather(torch.tensor([total_disc_acc], device=device)).sum().item()
    gathered_total_disc_pos_acc = accelerator.gather(torch.tensor([total_disc_pos_acc], device=device)).sum().item()
    gathered_total_samples = accelerator.gather(torch.tensor([total_samples], device=device)).sum().item()

    if mode=="test" and len(dataloader.dataset) != gathered_total_samples:
        raise AssertionError("Dataloader dataset length does not match gathered total samples")
    
    # Now gather the values for each key
    gathered_stage_acc = {}
    gathered_stage_num = {}
    
    # Only the main process computes and logs the final metrics
    if accelerator.is_main_process:
        # After processing all batches, compute overall metrics:
        overall_cont = gathered_total_cont / gathered_total_samples * 100
        overall_cont_pos = gathered_total_cont_pos / gathered_total_samples * 100
        overall_disc_acc = gathered_total_disc_acc / gathered_total_samples * 100
        overall_disc_pos_acc = gathered_total_disc_pos_acc / gathered_total_samples * 100
        
        # Prepare the log dictionary
        if model.task == "gsm8k":
            log_dict = {
                f"{mode}_acc/cont/overall": overall_cont, 
                f"{mode}_acc/cont/pos": overall_cont_pos, 
                f"{mode}_acc/disc/overall": overall_disc_acc,
                f"{mode}_acc/disc/pos": overall_disc_pos_acc,
            }
        else:
            log_dict = {
                f"{mode}_acc/cont/overall": overall_cont, 
                f"{mode}_acc/cont/pos": overall_cont_pos,
            }

        # Log everything with wandb:
        wandb.log(log_dict)
        
        # Save metrics to a jsonl file
        metrics_to_save = {
            "timestamp": time.time(),
            "step": step,
            "acc": {
                "overall": overall_cont,
                "pos": overall_cont_pos,
            },
            "disc": {
                "overall": overall_disc_acc,
                "pos": overall_disc_pos_acc,
            }
        }
        
        return log_dict
    return {}

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

    # Gather results from all processes
    all_losses = [
        torch.tensor([total_eval_loss], device=device),
        torch.tensor([total_eval_ce_loss], device=device),
        torch.tensor([total_eval_cont_loss], device=device),
        torch.tensor([n_batches], device=device)
    ]
    
    for i in range(len(all_losses)):
        all_losses[i] = accelerator.gather(all_losses[i]).sum()
    
    gathered_total_eval_loss, gathered_total_eval_ce_loss, gathered_total_eval_cont_loss, \
    gathered_n_batches = [t.item() for t in all_losses]

    # Compute average losses
    avg_total_loss = gathered_total_eval_loss / gathered_n_batches
    avg_ce_loss = gathered_total_eval_ce_loss / gathered_n_batches
    avg_cont_loss = gathered_total_eval_cont_loss / gathered_n_batches

    model.train()

    if accelerator.is_main_process:
        wandb.log({
            "eval/total_loss": avg_total_loss,
            "eval/ce_loss": avg_ce_loss,
            "eval/cont_loss": avg_cont_loss,
        })

    return {
        "total_loss": avg_total_loss,
        "ce_loss": avg_ce_loss,
        "cont_loss": avg_cont_loss,
    }