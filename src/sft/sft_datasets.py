import json
import os
import re
import torch
import torch.nn.functional as F
from typing import Optional, Sequence, List, Set, Dict, Any, Union
import transformers
import logging
from dataclasses import dataclass
import pathlib
from datasets import load_dataset

from utils.datasets import read_jsonl, get_few_shot_prompt, left_pad_sequences, right_pad_sequences, mask_labels
from utils.constants import DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, IGNORE_INDEX
from utils.gsm8k.decoding import extract_answer

def get_examples(data_dir, split, mode):
    
    examples = []

    if data_dir == 'fineweb-edu':
        f = load_dataset('hbin0701/fw-edu')[split]
    else:
        if split == "train":
            data_dir += "/train.json"
        elif split == "val":
            data_dir += "/valid.json"
        elif split == "test":
            data_dir += "/test.json"

        if '.jsonl' in data_dir:
            f = read_jsonl(data_dir)
        elif '.json' in data_dir:
            with open(data_dir, 'r') as f:
                f = json.load(f)
        
    for elem in f:
        new_elem = {}
        
        if elem['question'] != '':
            new_elem["question"] = elem["question"].strip() + "\n"
        else:
            new_elem["question"] = ''
        
        # Enable this for NO-SFT Setting.
        # elem['steps'] = [elem['steps'][-1]]
        #         
        if data_dir != 'fineweb-edu' or "###" in elem['steps'][-1] or "<answer>" in elem['steps'][-1]:
            new_elem["answer"] = "\n".join(elem["steps"])      
        else:
            new_elem["answer"] = "\n".join(elem["steps"]) + "\n### " + elem["answer"]
     
        examples.append(new_elem)

    print("DATA AMOUNT:", len(examples))
    return examples


def make_finetuning_generator_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: dataclass) -> Dict:
    train_dataset = FineTuningGeneratorDataset(
                        tokenizer=tokenizer, 
                        data_dir=data_args.data_dir, 
                        mode=data_args.mode,
                        target_set=data_args.target_set,
                        loss_on_prefix=data_args.loss_on_prefix,
                    )
    val_dataset = None

    return dict(train_dataset=train_dataset, val_dataset=val_dataset)


def make_test_generator_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: dataclass, inference_args: dataclass) -> Dict:
    test_dataset = TestGeneratorDataset(
        tokenizer=tokenizer, 
        data_dir=data_args.data_dir,
        target_set=data_args.target_set
    )
    return test_dataset

class FineTuningGeneratorDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizer = None, 
        mode: str="ft",
        data_dir: str = 'data/gsm8k', 
        target_set: str = 'train',
        loss_on_prefix=False,
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.target_set = target_set
        self.loss_on_prefix = False
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        print("+ [Dataset] Loading Training Data")
        self.examples = get_examples(self.data_dir, target_set, mode)
        qns_str = [ex["question"] for ex in self.examples]
        ans_str = [ex["answer"] for ex in self.examples]
        
        print("+ [Dataset] Tokenizing Testing Data")

        qns_tokens = []
        ans_tokens = []

        for x, y in zip(qns_str, ans_str):
            x_ = tokenizer(x, padding=False).input_ids
            y_ = tokenizer(y, padding=False, add_special_tokens=False, max_length=1023, truncation=True).input_ids

            qns_tokens.append(x_)
            ans_tokens.append(y_)

        # qns_tokens = tokenizer(qns_str, padding=False, max_length=1024).input_ids
        # ans_tokens = tokenizer(ans_str, padding=False, add_special_tokens=False, max_length=1024).input_ids

        self.qns_str = qns_str
        self.ans_str = ans_str
        self.qns_tokens = qns_tokens
        self.ans_tokens = ans_tokens

        print("MAX QNS TOKENS", max([len(qns_tokens[i]) for i in range(len(qns_tokens))]))
        print("MAX ANS TOKENS", max([len(ans_tokens[i]) for i in range(len(ans_tokens))]))

        self.max_len = max([
                len(qns_tokens[i]) + len(ans_tokens[i]) + 1
                for i in range(len(qns_tokens))
            ]
        )
        
        print("Example Sample", x, y)
        print(f"Max tokens: {self.max_len}")        
        print("Length:", len(self.qns_tokens))
        
    def __len__(self):
        return len(self.qns_tokens)

    def __getitem__(self, idx):
        qn_tokens = self.qns_tokens[idx]
        ans_tokens = self.ans_tokens[idx]

        input_ids = qn_tokens + ans_tokens + [self.eos_token_id]
        # input_ids = qn_tokens + ans_tokens
        labels = input_ids

        masks = (
            ([1] if self.loss_on_prefix else [0]) * len(qn_tokens)
            + ([1] * len(ans_tokens))
            + ([1])
        )
        labels = mask_labels(labels, masks)

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        return dict(input_ids=input_ids, labels=labels)

    def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids, attention_mask = right_pad_sequences(input_ids, padding_value=self.pad_token_id, return_attention_mask=True)
        labels = right_pad_sequences(labels, padding_value=IGNORE_INDEX, return_attention_mask=False)
        
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )




class TestGeneratorDataset(torch.utils.data.Dataset):
    """Left Padding"""
    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizer = None, 
        data_dir: str = 'data/gsm8k', 
        target_set: str = None,
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.target_set = target_set
        self.pad_token_id = tokenizer.pad_token_id

        print("+ [Dataset] Loading Testing Data")
        self.examples = get_examples(data_dir, target_set)
        qns_str = [ex["question"] for ex in self.examples]
        ans_str = [ex["answer"] for ex in self.examples]
        gts_str = [extract_answer(ans) for ans in ans_str]

        print("+ [Dataset] Tokenizing Testing Data")
        qns_tokens = tokenizer(qns_str, padding=False, max_length=1024, truncate=True).input_ids
        ans_tokens = tokenizer(ans_str, padding=False, add_special_tokens=False, max_length=1024, truncate=True).input_ids

        self.qns_str = qns_str
        self.qns_tokens = qns_tokens
        self.ans_str = ans_str
        self.gts_str = gts_str

        self.max_len = max([
                len(qns_tokens[i]) + len(ans_tokens[i]) + 1
                for i in range(len(qns_tokens))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns_tokens[idx]
        qn_str = self.qns_str[idx]
        ans_str = self.ans_str[idx]
        gt = self.gts_str[idx]

        input_ids = torch.tensor(qn_tokens)
        return dict(
            idx=idx, 
            input_ids=input_ids, 
            input=qn_str,
            question=qn_str,
            reference=gt,
            record_data=dict(answer=ans_str, ground_truth=gt),
        )

    def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, Any]:
        idx, input_ids, input, question, reference, record_data = tuple([instance[key] for instance in instances] for key in ("idx", "input_ids", "input", "question", "reference", "record_data"))
        record_data = {k: [instance[k] for instance in record_data] for k in record_data[0].keys()}

        input_ids, attention_mask = left_pad_sequences(input_ids, padding_value=self.pad_token_id, return_attention_mask=True)
        
        return dict(
            idx=idx,
            input_ids=input_ids,
            attention_mask=attention_mask,
            input=input,
            question=question,
            reference=reference,
            record_data=record_data,
        )
