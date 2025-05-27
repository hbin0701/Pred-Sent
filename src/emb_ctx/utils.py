import torch
import os
import numpy as np
from typing import Union, Tuple, Optional

def clear_gpu_cache_if_needed(device: torch.device, threshold: float = 0.7) -> None:
    """
    Clear GPU cache if memory usage exceeds the specified threshold.
    
    Args:
        device: The CUDA device to check memory on
        threshold: Memory usage threshold (0.0-1.0) at which to clear cache
        
    Returns:
        None
    """
    if not torch.cuda.is_available():
        return
        
    free, total = torch.cuda.mem_get_info(device)
    mem_used_percent = (total - free) / total

    if mem_used_percent > threshold:
        print(f"Memory usage exceeds {threshold*100:.0f}%, emptying cache...")
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device)
        mem_used_percent = (total - free) / total
        print(f"After emptying cache: {mem_used_percent*100:.2f}%")

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across PyTorch, NumPy, and Python.
    
    Args:
        seed: The random seed to use
        
    Returns:
        None
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in the specified directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to the latest checkpoint or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
        
    # List all subdirectories (epochs)
    subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    
    # Convert to integers if they are numeric
    numeric_subdirs = []
    for d in subdirs:
        try:
            numeric_subdirs.append(int(d))
        except ValueError:
            pass
    
    if not numeric_subdirs:
        return None
        
    # Get the latest epoch
    latest_epoch = max(numeric_subdirs)
    return os.path.join(checkpoint_dir, str(latest_epoch))

def compute_metrics(predictions, targets) -> dict:
    """
    Compute evaluation metrics for predictions.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Dictionary of metrics
    """
    correct = 0
    total = len(predictions)
    
    for pred, target in zip(predictions, targets):
        if pred.strip() == target.strip():
            correct += 1
            
    accuracy = correct / total if total > 0 else 0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    } 