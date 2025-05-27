import os
import json
import pickle
import torch
from tqdm import tqdm
from datasets import load_dataset

class ContrastiveStepDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, cache_dir=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if cache_dir is None:
            cache_dir = os.path.dirname(file_path)

        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(cache_dir, os.path.basename(file_path) + f".contrastive_step_cache_{max_length}.pkl")

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
                
                if 'prosqa' in file_path.lower():
                    sample['steps'].append("### " + sample['answer'])
                    
                steps = [x.strip() for x in sample["steps"] if x.strip()]
                question = sample.get("question", "")  # Get question if available
                
                # Process each step for contrastive learning
                for N in range(len(steps)):
                    # Model 1: Restoration - encoder_input_ids1 and decoder_input_ids1
                    
                    # Variation 1:
                    encoder_encoding1 = self.tokenizer(
                        steps[N].strip() + "\n",
                        truncation=True,
                        max_length=self.max_length,
                    )
                    
                    decoder_encoding1 = self.tokenizer(
                        steps[N].strip(),
                        truncation=True,
                        max_length=self.max_length,
                    )

                    # Model 2: Prediction - encoder_input_ids2 and decoder_input_ids2
                    context2 = (question + "\n" + "\n".join(steps[:N])).strip() + "\n"
                    target2 = steps[N].strip() + "\n"
                    
                    encoder_encoding2 = self.tokenizer(
                        context2,
                        truncation=True,
                        max_length=self.max_length,
                    )
                    
                    decoder_encoding2 = self.tokenizer(
                        target2,
                        truncation=True,
                        max_length=self.max_length,
                    )
                    
                    # remove bos token for encoder_input_ids1 & encoder_input_ids2

                    if 'llama' in self.tokenizer.name_or_path:
                        TGT_IDX = 1 
                    else:
                        TGT_IDX = 0

                    encoder_encoding1["input_ids"] = encoder_encoding1["input_ids"][TGT_IDX:]
                    encoder_encoding1["attention_mask"] = encoder_encoding1["attention_mask"][TGT_IDX:]
                    decoder_encoding1["input_ids"] = decoder_encoding1["input_ids"][TGT_IDX:]
                    decoder_encoding1["attention_mask"] = decoder_encoding1["attention_mask"][TGT_IDX:]
                    
                    # remove bos token for decoder_encoding2.
                    decoder_encoding2["input_ids"] = decoder_encoding2["input_ids"][TGT_IDX:]
                    decoder_encoding2["attention_mask"] = decoder_encoding2["attention_mask"][TGT_IDX:]  
                                        
                    self.processed_data.append({
                        "encoder_input_ids1": encoder_encoding1["input_ids"],
                        "encoder_attention_mask1": encoder_encoding1["attention_mask"],
                        "decoder_input_ids1": decoder_encoding1["input_ids"],
                        "decoder_attention_mask1": decoder_encoding1["attention_mask"],
                        
                        "encoder_input_ids2": encoder_encoding2["input_ids"],
                        "encoder_attention_mask2": encoder_encoding2["attention_mask"],
                        "decoder_input_ids2": decoder_encoding2["input_ids"],
                        "decoder_attention_mask2": decoder_encoding2["attention_mask"],
                        
                        "step_num": N
                    })

            with open(cache_file, "wb") as f:
                pickle.dump(self.processed_data, f)
            print(f"Processed dataset saved to {cache_file}")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]
