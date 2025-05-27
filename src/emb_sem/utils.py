import torch

def clear_gpu_cache_if_needed(device, threshold=0.7):
    free, total = torch.cuda.mem_get_info(device)
    mem_used_percent = (total - free) / total
    # print(f"Memory usage: {mem_used_percent*100:.2f}%")

    if mem_used_percent > threshold:
        print("Memory usage exceeds 70%, emptying cache...")
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device)
        mem_used_percent = (total - free) / total
        print(f"After emptying cache: {mem_used_percent*100:.2f}%")
