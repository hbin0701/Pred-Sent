import json
import os
import warnings
warnings.filterwarnings('ignore')  # Suppress all other warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings

import argparse
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler, AutoModelForCausalLM
from accelerate import Accelerator
import wandb

from dataset import ContrastiveStepDataset
from problem_dataset import ProblemDataset
from models import ContrastiveStepPredictor
from collate import contrastive_collate_fn, problem_collate_fn
from train import train_contrastive

def set_seed(seed: int):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed (int): Seed value to use
    """
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


def create_dataloaders(dataset, batch_size, num_workers, collate_fn, shuffle=False):
    """Helper function to create DataLoader objects"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

def setup_optimizer_and_scheduler(model, lr, num_warmup_steps, total_training_steps):
    """Create optimizer and learning rate scheduler for a model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )
    return optimizer, lr_scheduler

def prepare_with_accelerator(accelerator, model, optimizer, lr_scheduler, dataloader):
    """Prepare model, optimizer, scheduler with an accelerator"""
    return accelerator.prepare(model, optimizer, lr_scheduler, dataloader)

def main():
    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument("--train_file", type=str, default="train.json", help="Path to train.json file")
    parser.add_argument("--valid_file", type=str, default="valid.json", help="Path to valid.json file")
    parser.add_argument("--test_file", type=str, default="valid.json", help="Path to valid.json file")
    
    # Model parameters
    parser.add_argument("--encoder_model_name", type=str, default="gpt2", help="Pretrained encoder model (deprecated)")
    parser.add_argument("--decoder_model_name", type=str, default="gpt2", help="Pretrained decoder model (deprecated)")
    parser.add_argument("--encoder1_model_name", type=str, default=None, help="Pretrained encoder1 model for restoration")
    parser.add_argument("--decoder1_model_name", type=str, default=None, help="Pretrained decoder1 model for restoration")
    parser.add_argument("--encoder2_model_name", type=str, default=None, help="Pretrained encoder2 model for prediction")
    parser.add_argument("--decoder2_model_name", type=str, default=None, help="Pretrained decoder2 model for prediction")
    parser.add_argument("--tokenizer_model_name", type=str, default="gpt2", help="Pretrained tokenizer")
    parser.add_argument("--share_param", type=bool, default=False, help="Whether encoder and decoder share parameters")
    
    # Training parameters
    parser.add_argument("--per_device_batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_warmup_steps", type=int, default=100, help="Number of steps for the linear warmup")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Number of steps to accumulate gradients")
    
    # Input processing parameters
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    # Experiment tracking parameters
    parser.add_argument("--proj_name", type=str, default="My_project", help="Project name for wandb")
    parser.add_argument("--exp_name", type=str, default="Context_Step_Prediction", help="Experiment name for wandb")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_stage1.6", help="Directory to save checkpoints")

    # Loss function parameters
    parser.add_argument('--use_dist', action='store_true', help='Whether to use distance loss')
    parser.add_argument('--use_cont', action='store_true', help='Whether to apply contrastive loss during training')
    parser.add_argument("--contrastive_weight", type=float, default=0.1, help="Weight for contrastive loss")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for NT-Xent contrastive loss")
    parser.add_argument("--task", type=str, default="gsm8k", help="Task to use: csqa, gsm8k")
    
    # Problem solving parameters
    parser.add_argument("--problem_gen_steps", type=int, default=10, help="Number of steps to generate for problem solving")

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Handle backward compatibility for model paths
    if args.encoder1_model_name is None:
        args.encoder1_model_name = args.encoder_model_name
    if args.decoder1_model_name is None:
        args.decoder1_model_name = args.decoder_model_name
    if args.encoder2_model_name is None:
        args.encoder2_model_name = args.encoder_model_name
    if args.decoder2_model_name is None:
        args.decoder2_model_name = args.decoder_model_name

    # Initialize wandb for experiment tracking
    # For contrastive learning, create two separate accelerators
    accelerator1 = Accelerator(mixed_precision="bf16")
    accelerator2 = Accelerator(mixed_precision="bf16")
    device = accelerator1.device  # Use device from first accelerator for consistency
    
    if accelerator1.is_main_process:
        wandb.login(key="PUT_YOUR_KEY_HERE")
        wandb.init(project=args.proj_name, entity="hbin0701", name=args.exp_name, config=vars(args))

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # Initialize models, datasets, and collate functions based on training mode
    # Create two separate models: one for restoration and one for prediction
    model1 = ContrastiveStepPredictor(
        tokenizer, args.encoder1_model_name, args.decoder1_model_name, 
        args.share_param, args.task
    )
    
    model2 = ContrastiveStepPredictor(
        tokenizer, args.encoder2_model_name, args.decoder2_model_name, 
        args.share_param, args.task
    )
    
    # Create contrastive datasets
    train_dataset = ContrastiveStepDataset(args.train_file, tokenizer, max_length=args.max_length)
    valid_dataset = ContrastiveStepDataset(args.valid_file, tokenizer, max_length=args.max_length)
    test_dataset = ContrastiveStepDataset(args.test_file, tokenizer, max_length=args.max_length)
    
    
    # Use contrastive collate function
    collate_function = lambda batch: contrastive_collate_fn(batch, tokenizer)
    
    # Move models to device
    model1.to(device)
    model2.to(device)
    model1.max_length = args.max_length
    model2.max_length = args.max_length

    # Create problem datasets for evaluation
    train_problem_dataset = ProblemDataset(args.train_file, tokenizer, max_length=args.max_length)
    valid_problem_dataset = ProblemDataset(args.valid_file, tokenizer, max_length=args.max_length)
    test_problem_dataset = ProblemDataset(args.test_file, tokenizer, max_length=args.max_length)
    

    # Create DataLoaders for training and validation
    train_dataloader = create_dataloaders(train_dataset, args.per_device_batch_size, args.num_workers, 
                                         collate_function, shuffle=True)
    train_gen_dataloader = create_dataloaders(train_dataset, args.per_device_batch_size, args.num_workers, 
                                            collate_function, shuffle=False)
    valid_dataloader = create_dataloaders(valid_dataset, args.per_device_batch_size, args.num_workers, 
                                        collate_function, shuffle=False)
    test_dataloader = create_dataloaders(test_dataset, args.per_device_batch_size, args.num_workers, 
                                       collate_function, shuffle=False)
        
    # Problem dataset collate function
    problem_collate = lambda batch: problem_collate_fn(batch, tokenizer)
    
    # Create DataLoaders for problem evaluation
    train_problem_dataloader = create_dataloaders(train_problem_dataset, args.per_device_batch_size, 
                                                args.num_workers, problem_collate, shuffle=False)
    valid_problem_dataloader = create_dataloaders(valid_problem_dataset, args.per_device_batch_size, 
                                                args.num_workers, problem_collate, shuffle=False)
    test_problem_dataloader = create_dataloaders(test_problem_dataset, args.per_device_batch_size, 
                                               args.num_workers, problem_collate, shuffle=False)
    
    # Calculate total training steps for learning rate scheduling
    num_update_steps_per_epoch = len(train_dataloader)
    total_training_steps = args.num_epochs * num_update_steps_per_epoch
    
    # If gradient accumulation is enabled, adjust the number of update steps
    if args.grad_accum_steps > 1:
        num_update_steps_per_epoch = len(train_dataloader) // args.grad_accum_steps
        total_training_steps = args.num_epochs * num_update_steps_per_epoch
        print(f"Using gradient accumulation with {args.grad_accum_steps} steps")
        print(f"Effective batch size: {args.per_device_batch_size * args.grad_accum_steps}")
        print(f"Updates per epoch: {num_update_steps_per_epoch}")
        print(f"Total training steps: {total_training_steps}")

    # Create optimizers and schedulers
    # Create separate optimizers and schedulers for each model
    optimizer1, lr_scheduler1 = setup_optimizer_and_scheduler(
        model1, args.lr, args.num_warmup_steps, total_training_steps
    )
    
    optimizer2, lr_scheduler2 = setup_optimizer_and_scheduler(
        model2, args.lr, args.num_warmup_steps, total_training_steps
    )
    
    # Create temporary dataloaders for model preparation
    train_dataloader_copy1 = create_dataloaders(train_dataset, args.per_device_batch_size, args.num_workers, 
                                                collate_function, shuffle=True)
    train_dataloader_copy2 = create_dataloaders(train_dataset, args.per_device_batch_size, args.num_workers, 
                                                collate_function, shuffle=True)
    
    # Prepare models with their respective accelerators
    model1, optimizer1, lr_scheduler1, _ = prepare_with_accelerator(
        accelerator1, model1, optimizer1, lr_scheduler1, train_dataloader_copy1
    )
    
    model2, optimizer2, lr_scheduler2, _ = prepare_with_accelerator(
        accelerator2, model2, optimizer2, lr_scheduler2, train_dataloader_copy2
    )
    
    # Prepare the dataloaders with accelerator2
    train_dataloader, train_gen_dataloader, valid_dataloader, test_dataloader = accelerator2.prepare(
        train_dataloader, train_gen_dataloader, valid_dataloader, test_dataloader
    )

    # Create dictionaries for model, optimizer, scheduler, and accelerator
    models = {"model1": model1, "model2": model2}
    optimizers = {"optimizer1": optimizer1, "optimizer2": optimizer2}
    lr_schedulers = {"lr_scheduler1": lr_scheduler1, "lr_scheduler2": lr_scheduler2}
    accelerators = {"accelerator1": accelerator1, "accelerator2": accelerator2}

    train_contrastive(
        args, accelerators, models, optimizers, device, lr_schedulers,
        train_dataloader, train_gen_dataloader, valid_dataloader, test_dataloader,
        train_problem_dataloader, valid_problem_dataloader, test_problem_dataloader,
        args.problem_gen_steps
    )

if __name__ == "__main__":
    main() 