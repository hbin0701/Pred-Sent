import os
import json
import pickle
import torch
from tqdm import tqdm
from datasets import load_dataset

class StepsDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, cache_dir=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if cache_dir is None:
            cache_dir = os.path.dirname(file_path)

        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(cache_dir, os.path.basename(file_path) + f".cache_{max_length}.pkl")

        if os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file}")
            with open(cache_file, "rb") as f:
                self.processed_data = pickle.load(f)
        else:
            if 'fw-edu' in file_path:
                mode = file_path.split("/")[-1].replace(".json", "")
                self.data = load_dataset("hbin0701/fw-edu")[mode]
            else:
                with open(file_path, "r") as f:
                    self.data = json.load(f)
                
            self.processed_data = []
            for sample in tqdm(self.data):
                if not len([x.strip() for x in sample['steps'] if x.strip()]) > 0:
                    continue
                
                # Optionally use question/answer fields if needed
                steps = sample["steps"]

                steps = [x.strip() + "\n" for x in steps]
                for idx, step in enumerate(steps):
                    encoder_text = step
                    decoder_text = step.strip()

                    encoder_encoding = self.tokenizer(
                        encoder_text,
                        truncation=True,
                        max_length=self.max_length,
                    )
                    decoder_encoding = self.tokenizer(
                        decoder_text,
                        truncation=True,
                        max_length=self.max_length,
                    )
                    self.processed_data.append({
                        "encoder_input_ids": encoder_encoding["input_ids"],
                        "encoder_attention_mask": encoder_encoding["attention_mask"],
                        "decoder_input_ids": decoder_encoding["input_ids"],
                        "decoder_attention_mask": decoder_encoding["attention_mask"],
                        "step_num": idx
                    })

            with open(cache_file, "wb") as f:
                pickle.dump(self.processed_data, f)
            print(f"Processed dataset saved to {cache_file}")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]
