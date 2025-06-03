import os
import warnings
warnings.filterwarnings('ignore')  # Suppress all other warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from accelerate import Accelerator
import wandb

from dataset import StepsDataset
from models import AutoEncoderModel
from collate import dual_collate_fn
from train import train_stage1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="train.json", help="Path to train.json file")
    parser.add_argument("--valid_file", type=str, default="valid.json", help="Path to valid.json file")
    parser.add_argument("--test_file", type=str, default="valid.json", help="Path to valid.json file")
    
    parser.add_argument("--encoder_model_name", type=str, default="gpt2", help="Pretrained encoder model")
    parser.add_argument("--decoder_model_name", type=str, default="gpt2", help="Pretrained decoder model")
    parser.add_argument("--tokenizer_model_name", type=str, default="gpt2", help="Pretrained tokenizer")
    
    parser.add_argument("--per_device_batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    
    parser.add_argument("--proj_name", type=str, default="My_project")
    parser.add_argument("--exp_name", type=str, default="My_experiment")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--wandb_key", type=str, default=None, help="WandB API key")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity name")

    parser.add_argument("--share_param", type=bool, default=False) # whether for encoder and decoder to share parameters.

    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    if accelerator.is_main_process:
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
            wandb.init(
                project=args.proj_name, 
                entity=args.wandb_entity, 
                name=args.exp_name, 
                config=vars(args)
            )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    model = AutoEncoderModel(tokenizer, args.encoder_model_name, args.decoder_model_name, args.share_param)
    model.to(device)

    # Create datasets.
    train_dataset = StepsDataset(args.train_file, tokenizer, max_length=args.max_length)
    valid_dataset = StepsDataset(args.valid_file, tokenizer, max_length=args.max_length)
    test_dataset = StepsDataset(args.test_file, tokenizer, max_length=args.max_length)

    # Create DataLoaders.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: dual_collate_fn(batch, tokenizer)
    )
    train_gen_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: dual_collate_fn(batch, tokenizer)
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: dual_collate_fn(batch, tokenizer)
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: dual_collate_fn(batch, tokenizer)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_update_steps_per_epoch = len(train_dataloader)
    total_training_steps = args.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_training_steps,
    )

    model, optimizer, train_dataloader, train_gen_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, train_gen_dataloader, lr_scheduler
    )

    train_stage1(args, accelerator, model, optimizer, lr_scheduler, 
                 train_dataloader, train_gen_dataloader, valid_dataloader, test_dataloader, device)

if __name__ == "__main__":
    main()
