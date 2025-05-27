import os
import json
import pickle
import torch
from tqdm import tqdm
from datasets import load_dataset

class ProblemDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, cache_dir=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if cache_dir is None:
            cache_dir = os.path.dirname(file_path)

        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(cache_dir, os.path.basename(file_path) + f".problem_dataset_cache_{max_length}.pkl")

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
                if 'question' not in sample or 'answer' not in sample:
                    continue
                
                question = sample.get("question", "").strip() + "\n"
                answer = sample.get("answer", "").strip()
                
                # Skip samples without question or answer
                if not question or not answer:
                    continue
                
                # Encode question for the encoder
                encoder_encoding = self.tokenizer(
                    question,
                    truncation=True,
                    max_length=self.max_length,
                )
                
                self.processed_data.append({
                    "encoder_input_ids": encoder_encoding["input_ids"],
                    "encoder_attention_mask": encoder_encoding["attention_mask"],
                    "answer": answer,
                    "question": question
                })

            with open(cache_file, "wb") as f:
                pickle.dump(self.processed_data, f)
            print(f"Processed dataset saved to {cache_file}")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx] 