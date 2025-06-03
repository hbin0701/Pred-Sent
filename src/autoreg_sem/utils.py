import re
import random
import numpy as np
import torch
from colorama import Fore, Style, init

def pprint(log_dict):
    # Iterate through the dictionary sorted by key for consistent ordering.
    for key, value in sorted(log_dict.items()):
        # Split keys on "/" to highlight different levels
        parts = key.split('/')
        # Join parts with a separator (no color)
        formatted_key = " / ".join(parts)
        # Round the value for consistency
        formatted_value = str(round(value, 4))
        print(f"{formatted_key}: {formatted_value}")
    
    print()

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


def check(text):
    try:
        idx1 = text.rindex("=")
        idx2 = text.rindex(">>")
        return text[idx1+1:idx2]
    except Exception:
        return -1

def check_eq(eq1, eq2):
    return eq1.strip() == eq2.strip()

def extract_final_answer(text, task):
    
    if task in ["gsm8k"]:
        try:
                ridx = text.rindex(">>")
                return text[ridx + len(">>\n"):].split("\n")[0].strip()
        except:
            # print("Not found", text)
            return text
        
    else:
        idx = text.find("###")
        if idx == -1:
            return -1
        ans = text[idx:].split("\n")[0].replace("###", "").strip()

        # if random.random() < 0.2:
        #     print("Answer", ans)

        return ans

def compare_last_formula(text):
    try:
        idx1 = text.rindex("=")
        idx2 = text.rindex(">>")
        return text[idx1+1:idx2]
    except Exception:
        return -1

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
