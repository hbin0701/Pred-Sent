import os
from datasets import load_dataset
from torch.utils.data import Dataset



class StepsDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024, num_proc=16):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load the raw JSON dataset.
        if 'fw-edu' in file_path:
            mode = file_path.split("/")[-1].replace(".json", "")
            raw_dataset = load_dataset("hbin0701/fw-edu")[mode]
        else:
            raw_dataset = load_dataset("json", data_files={"train": file_path})["train"]
            raw_dataset = raw_dataset.filter(lambda elem: len([x.strip() for x in elem['steps'] if x.strip()]) > 0)

        def process_sample(example):
            question = example["question"].strip()
            steps = example["steps"]
            answer = example["answer"]
            # Ensure each step ends with a newline.

            if not ('gsm8k' in file_path or 'edu' in file_path):
                if not steps[-1].startswith("###"):
                    steps.append(f"### {answer}")
                
            steps = [step.strip() + "\n" for step in steps if step.strip() != ""]
            steps = steps[:20]

            # Create encoder input by combining question and steps.
            
            if 'edu' not in file_path:
                inp = question + "\n" + "".join(steps)
            else:
                inp = ''.join(steps)
            
            # Might have to fix... GT and stuffs of course.
            encoder_encoding = tokenizer(
                inp,
                truncation=True,
                max_length=max_length,
            )
            steps_tok = []

            for step in steps:
                step_tokens = tokenizer(
                    step,
                    truncation=True,
                    max_length=max_length,
                )
                steps_tok.append(step_tokens)
            
            return {
                "encoder_input_ids": encoder_encoding["input_ids"],
                "encoder_attention_mask": encoder_encoding["attention_mask"],
                "steps_enc": steps_tok,
                "length": len(encoder_encoding["input_ids"])
            }

        def is_valid_sample(sample):
            # Check that all steps in steps_enc have 100 or fewer tokens.
            # otherwise, it's likely to have weird tokens.
            return all(len(step["input_ids"]) <= 100 for step in sample["steps_enc"])
        
        processed_dataset = raw_dataset.map(process_sample, num_proc=num_proc)
        processed_dataset = processed_dataset.filter(is_valid_sample)

        # Filter out examples with too many steps.
        # processed_dataset = processed_dataset.filter(lambda example: len(example['steps_enc']) < 25)
        self.processed_data = processed_dataset

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]
