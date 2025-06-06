import os
import random
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')  # Suppress all other warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings

import argparse
from transformers import AutoTokenizer, get_scheduler
from accelerate import Accelerator
from dataset import StepsDataset
from models import AutoRegressiveModel
from train import train
from collate import collate_fn

def set_seed(seed: int):
    # Set seed for Python's built-in random module
    random.seed(seed)    
    np.random.seed(seed)    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    # Ensure that each DataLoader worker has a fixed seed
    seed = 42  # or use a parameterized seed value
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

def main():
    # Set global seed for reproducibility before any random operations or Accelerator instantiation
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--encoder_path", type=str, required=False)
    parser.add_argument("--latent_model_path", type=str, required=False)
    parser.add_argument("--decoder_path", type=str, required=False)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--complete_model_path", type=str, default=None, help="Path to load complete model from")
    
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    
    parser.add_argument("--proj_name", type=str, default="My_project")
    parser.add_argument("--exp_name", type=str, default="My_experiment")
    parser.add_argument("--wandb_key", type=str, default=None, help="Weights & Biases API key")
    parser.add_argument("--wandb_entity", type=str, default="hbin0701", help="Weights & Biases entity (username or team name)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument('--freeze', action='store_true', help='Whether to freeze the encoder & decoder or not.')
    parser.add_argument('--share_param', action='store_true', help='Whether for encoder & decoder to share the weight or not.')
    parser.add_argument('--use_cont', action='store_true', help='Whether to use contrastive loss or not.')
    
    args = parser.parse_args()
    
    # Validate that either complete_model_path or all component paths are provided
    if args.complete_model_path is None and (args.encoder_path is None or args.latent_model_path is None or args.decoder_path is None):
        raise ValueError("Either complete_model_path or all of encoder_path, latent_model_path, and decoder_path must be provided")
    
    set_seed(args.seed)

    # Initialize Accelerator AFTER setting the seed
    accelerator = Accelerator()
    device = accelerator.device
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset_train = StepsDataset(args.train_file, tokenizer)
    dataset_eval = StepsDataset(args.eval_file, tokenizer)
    dataset_test = StepsDataset(args.test_file, tokenizer)
    
    # Set worker_init_fn for each DataLoader to ensure DataLoader workers are seeded
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=lambda x: collate_fn(x, tokenizer),
        worker_init_fn=worker_init_fn
    )
    
    dataloader_eval = torch.utils.data.DataLoader(
        dataset_eval, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, tokenizer),
        worker_init_fn=worker_init_fn
    )
    
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, tokenizer),
        worker_init_fn=worker_init_fn
    )
    
    if args.complete_model_path:
        # Load the complete model from the provided path
        model = AutoRegressiveModel.load_model(
            args.complete_model_path, 
            tokenizer=tokenizer, 
            task=args.task, 
            use_cont=args.use_cont,
            freeze=args.freeze
        )
    else:
        # Load model components separately as before
        model = AutoRegressiveModel(
            tokenizer, args.encoder_path, args.latent_model_path, args.decoder_path, 
            args.task, args.freeze, args.share_param, args.use_cont
        )
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Calculate total steps and steps per epoch
    total_steps = len(dataloader_train) * args.num_epochs
    steps_per_epoch = len(dataloader_train)
    
    # Use CosineAnnealingWarmRestarts with T_0=1 epoch and T_mult=2
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=steps_per_epoch,  # Restart every epoch
        T_mult=2,  # Double the restart interval after each restart
        eta_min=args.lr * 0.1  # Minimum learning rate is 10% of initial lr
    )
    
    # Prepare model, optimizer, and schedulers with Accelerator
    model, optimizer, dataloader_eval, dataloader_test, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader_eval, dataloader_test, lr_scheduler
    )
    
    train(
        args, model, dataset_train, dataloader_eval, dataloader_test, optimizer,
        lr_scheduler, device, args.num_epochs, accelerator, args.save_dir
    )

if __name__ == "__main__":
    main()
