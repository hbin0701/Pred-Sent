import os
import glob
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
import wandb

from utils import clear_gpu_cache_if_needed
from evaluate import test_model

def train_stage1(args, accelerator, model, optimizer, lr_scheduler, 
                 train_dataloader, train_gen_dataloader, valid_dataloader, test_dataloader, device):
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        for idx, batch in tqdm(enumerate(train_dataloader), 
                                 desc="Processing Training Batches", 
                                 total=len(train_dataloader)):

            encoder_input_ids = batch["encoder_input_ids"].to(device)
            encoder_attention_mask = batch["encoder_attention_mask"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            decoder_attention_mask = batch["decoder_attention_mask"].to(device)

            loss, _ = model(encoder_input_ids, encoder_attention_mask, 
                            decoder_input_ids, decoder_attention_mask)
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            total_train_loss += loss.item()
            train_steps += 1

            if (idx + 1) % 10 == 0:
                clear_gpu_cache_if_needed(device, threshold=0.7)

            if accelerator.is_main_process:
                wandb.log({"train_loss": loss.item()})

        avg_train_loss = total_train_loss / train_steps if train_steps > 0 else 0.0

        model.eval()
        total_eval_loss = 0.0
        eval_steps = 0
        with torch.no_grad():
            for batch in valid_dataloader:
                encoder_input_ids = batch["encoder_input_ids"].to(device)
                encoder_attention_mask = batch["encoder_attention_mask"].to(device)
                decoder_input_ids = batch["decoder_input_ids"].to(device)
                decoder_attention_mask = batch["decoder_attention_mask"].to(device)

                loss, _ = model(encoder_input_ids, encoder_attention_mask, 
                                decoder_input_ids, decoder_attention_mask)
                total_eval_loss += loss.item()
                eval_steps += 1

        avg_eval_loss = total_eval_loss / eval_steps if eval_steps > 0 else 0.0

        if accelerator.is_main_process:
            wandb.log({"eval_loss": avg_eval_loss, "epoch": epoch + 1, "global_step": global_step})
            unwrapped_model = accelerator.unwrap_model(model)
            # Determine checkpoint version.
            new_checkpoint = f"{args.save_dir}/{epoch+1}"
            os.makedirs(new_checkpoint, exist_ok=True)
            unwrapped_model.encoder.save_pretrained(f"{new_checkpoint}/encoder", save_function=accelerator.save)
            unwrapped_model.decoder.save_pretrained(f"{new_checkpoint}/decoder", save_function=accelerator.save)
        
            MAX_SAMPLES = len(test_dataloader.dataset)
            print(test_model(model, train_gen_dataloader, device, accelerator, global_step, mode="train", MAX_SAMPLES=MAX_SAMPLES))
            print(test_model(model, valid_dataloader, device, accelerator, global_step, mode="eval", MAX_SAMPLES=MAX_SAMPLES))
            print(test_model(model, test_dataloader, device, accelerator, global_step, mode="test", MAX_SAMPLES=MAX_SAMPLES))
            torch.cuda.empty_cache()

        accelerator.wait_for_everyone()
