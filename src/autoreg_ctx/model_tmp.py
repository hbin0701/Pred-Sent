import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from utils import check_eq, extract_final_answer, compare_last_formula  # Adjust import if needed
import random
import wandb
import pandas as pd
import os
import pickle
from peft import PeftModel, LoraConfig, get_peft_model
class AutoRegressiveModel(nn.Module):
    def __init__(self, tokenizer, encoder_path, latent_model_path, decoder_path, task, freeze, share_param, use_cont):
        """
        Loads the encoder, latent model, and decoder models.
        Initializes the tokenizer from a fixed checkpoint.
        Encoder and Latent Model can be optionally frozen based on freeze arg.
        Adds projection layers between encoder-latent model and latent model-decoder.
        """
        super().__init__()
        self.task = task
        self.dropout_rate = 0.2 # TODO: Make this configurable
        self.encoder = AutoModelForCausalLM.from_pretrained(encoder_path)
        self.latent_model = AutoModelForCausalLM.from_pretrained(latent_model_path)
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_path)
        if 'qwen' in encoder_path.lower():
            lora_config = LoraConfig(
                    r=1024,
                    lora_alpha=2048,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    bias="none",
                    task_type="CAUSAL_LM"
                )
            self.encoder = get_peft_model(self.encoder, lora_config)
            # Encoder and Translator need SFT PATH.
            PUT_YOUR_SFT_PATH_HERE = "/fsx/byeongguk/Pred-Sent-1/ckpts/qwen-csqa-cot"
            self.encoder = PeftModel.from_pretrained(PUT_YOUR_SFT_PATH_HERE, encoder_path)
            self.latent_model = get_peft_model(self.latent_model, lora_config)
            self.decoder = PeftModel.from_pretrained(PUT_YOUR_SFT_PATH_HERE, decoder_path)
        # Enable gradient checkpointing for memory efficiency
        # self.encoder.gradient_checkpointing_enable()
        # self.latent_model.gradient_checkpointing_enable()
        # self.decoder.gradient_checkpointing_enable()
        # Freeze encoder & translator parameters only if freeze is True.
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
        else:
            # If not freezing, ensure they are trainable
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = True
        # Initialize the common tokenizer.
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Cache the <|new_line|> token id.
        self.im_end_token_id = self.tokenizer.encode("<|new_line|>")[0]
        # Set pad token IDs.
        self.encoder.config.pad_token_id = self.tokenizer.eos_token_id
        self.latent_model.config.pad_token_id = self.tokenizer.eos_token_id
        self.decoder.config.pad_token_id = self.tokenizer.eos_token_id
        # Define projection layers
        hidden_size = self.encoder.config.hidden_size # Assuming encoder, latent_model, decoder have same hidden size
        self.encoder_to_latent_model_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.latent_model_to_decoder_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        # Move models and projections to bfloat16.
        self.encoder.to(torch.bfloat16)
        self.latent_model.to(torch.bfloat16)
        self.decoder.to(torch.bfloat16)
        self.encoder_to_latent_model_proj.to(torch.bfloat16)
        self.latent_model_to_decoder_proj.to(torch.bfloat16)
        self.use_cont = use_cont
        # Store configuration for saving/loading
        self.config = {
            "task": task,
            "freeze": freeze,
            "share_param": share_param,
            "use_cont": use_cont,
            "dropout_rate": self.dropout_rate
        }