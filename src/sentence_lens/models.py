import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from peft import PeftModel, LoraConfig, get_peft_model

class AutoEncoderModel(nn.Module):
    def __init__(self, tokenizer, encoder_path, latent_model_path, decoder_path, task, freeze, share_param, use_cont):
        """
        Loads the encoder, latent_model, and decoder models.
        Initializes the tokenizer from a fixed checkpoint.
        Encoder and decoder can be optionally frozen based on freeze arg.
        Adds projection layers between encoder-latent_model and latent_model-decoder.
        """
        super().__init__()
        self.task = task
        self.dropout_rate = 0.2 # TODO: Make this configurable

       # Inconsistency on target modules is a trivial mistake on our side during training. Feel free to add whichever modules you want.
        lora_config_encoder = LoraConfig(
            r=256,
            lora_alpha=1024,
            target_modules=["c_attn"],  # 적용할 모델의 특정 레이어 - removed c_proj to match saved model
            lora_dropout=0.1,  # 드롭아웃 비율
            bias="none",
            task_type="CAUSAL_LM"
        )

 
        lora_config_latent = LoraConfig(
            r=256,
            lora_alpha=1024,    
            target_modules=["c_attn","c_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        # We will load .pt file anyways later.
        DUMMY_PATH = "openai-community/gpt2-large"
        # Just set up with random values.
        # self.encoder = PeftModel.from_pretrained(DUMMY_PATH, encoder_path)
        self.encoder = AutoModelForCausalLM.from_pretrained(DUMMY_PATH)
        self.encoder = get_peft_model(self.encoder, lora_config_encoder)

        # Initialize latent_model with LoRA config
        self.latent_model = AutoModelForCausalLM.from_pretrained(DUMMY_PATH)
        self.latent_model = get_peft_model(self.latent_model, lora_config_latent)
        
        # Load decoder with LoRA
        self.decoder = AutoModelForCausalLM.from_pretrained(DUMMY_PATH)
        self.decoder = get_peft_model(self.decoder, lora_config_encoder)

        # Enable gradient checkpointing for memory efficiency
        self.encoder.gradient_checkpointing_enable()
        self.latent_model.gradient_checkpointing_enable()
        self.decoder.gradient_checkpointing_enable()

        # Freeze encoder & decoder parameters only if freeze is True.
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

        # Cache the newline token id.
        self.newline_token_id = self.tokenizer.encode("\n")[0]

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


    def pad_question(self, batch_encoder_input_ids):
        """
        Left-pads question token sequences extracted from encoder_input_ids.
        """
        device = batch_encoder_input_ids.device
        batch_size = batch_encoder_input_ids.size(0)
        questions = []
        lengths = []
        for i in range(batch_size):
            enc_ids = batch_encoder_input_ids[i]
            newline_positions = (enc_ids == self.newline_token_id).nonzero(as_tuple=False)
            first_newline = newline_positions[0].item() if newline_positions.numel() > 0 else enc_ids.size(0) - 1
            question_ids = enc_ids[:first_newline+1]
            questions.append(question_ids)
            lengths.append(question_ids.size(0))
        max_len = max(lengths)
        all_q_padded = torch.full((batch_size, max_len), 0, dtype=batch_encoder_input_ids.dtype, device=device)
        attention_mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.long)
        position_ids = torch.zeros(batch_size, max_len, device=device, dtype=torch.long)
        for i, question_ids in enumerate(questions):
            q_len = question_ids.size(0)
            pad_len = max_len - q_len
            all_q_padded[i, pad_len:] = question_ids
            attention_mask[i, pad_len:] = 1
            position_ids[i, pad_len:] = torch.arange(q_len, device=device)
        return all_q_padded, attention_mask, position_ids


    def save_model(self, save_dir):
        """
        Save the model state dictionaries and configuration to the specified directory.
        
        Args:
            save_dir (str): Directory to save the model files
        """
        import os
        import json
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save projection layers state dictionaries
        torch.save(self.encoder_to_latent_model_proj.state_dict(), os.path.join(save_dir, "encoder_to_latent_model_proj.pt"))
        torch.save(self.latent_model_to_decoder_proj.state_dict(), os.path.join(save_dir, "latent_model_to_decoder_proj.pt"))
        
        # Save model state dictionaries - save all components regardless of sharing
        torch.save(self.latent_model.state_dict(), os.path.join(save_dir, "latent_model.pt"))
        torch.save(self.encoder.state_dict(), os.path.join(save_dir, "encoder.pt"))
        torch.save(self.decoder.state_dict(), os.path.join(save_dir, "decoder.pt"))
        
        # Save configuration
        config = {
            "task": self.task,
            "share_param": self.config["share_param"],
            "use_cont": self.use_cont,
            "use_dist": self.use_dist,
            "use_mse": self.use_mse,
            "dropout_rate": self.dropout_rate,
            "encoder_path": self.encoder.config._name_or_path,
            "latent_model_path": self.latent_model.config._name_or_path,
            "decoder_path": self.decoder.config._name_or_path if not self.config["share_param"] else None,
            "tokenizer_name": self.tokenizer.name_or_path
        }
        
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f)
            
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
        
        # Also save the complete model architectures - save all regardless of sharing
        self.encoder.save_pretrained(os.path.join(save_dir, "encoder_model"))
        self.latent_model.save_pretrained(os.path.join(save_dir, "latent_model_model"))
        self.decoder.save_pretrained(os.path.join(save_dir, "decoder_model"))
        
        print(f"Model saved to {save_dir}")
        
    @classmethod
    def load_model(cls, load_dir, tokenizer=None, task=None, use_cont=None, use_dist=None, use_mse=None, freeze=False):
        """
        Load a saved model from directory.
        
        Args:
            load_dir (str): Directory containing saved model files
            tokenizer (Optional): Pre-loaded tokenizer. If None, will load from saved files
            task (str, Optional): Task name, overrides the saved task if provided
            use_cont (bool, Optional): Whether to use contrastive loss, overrides saved setting if provided
            use_dist (bool, Optional): Whether to use distance loss, overrides saved setting if provided
            use_mse (bool, Optional): Whether to use MSE loss, overrides saved setting if provided
            
        Returns:
            AutoEncoderModel: Loaded model instance
        """
        import os
        import json
        
        # Load configuration
        with open(os.path.join(load_dir, "config.json"), "r") as f:
            config = json.load(f)
        
        # Override config with provided parameters if any
        if task is not None:
            config["task"] = task
        if use_cont is not None:
            config["use_cont"] = use_cont

        # Load tokenizer if not provided
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_dir, "tokenizer"))
        
        # Create empty model instance
        model = cls(
            tokenizer=tokenizer,
            encoder_path="",
            latent_model_path="",
            decoder_path="",
            task=config["task"],
            freeze=freeze,
            share_param=config["share_param"],
            use_cont=config["use_cont"]
        )
        
        # Update dropout rate if different
        model.dropout_rate = config["dropout_rate"]
        
        print("Load dir", load_dir)

        # Load saved state dictionaries
        model.encoder_to_latent_model_proj.load_state_dict(
            torch.load(os.path.join(load_dir, "encoder_to_latent_model_proj.pt"))
        )
        model.latent_model_to_decoder_proj.load_state_dict(
            torch.load(os.path.join(load_dir, "latent_model_to_decoder_proj.pt"))
        )
        model.latent_model.load_state_dict(
            torch.load(os.path.join(load_dir, "latent_model.pt"))
        )
        model.encoder.load_state_dict(
            torch.load(os.path.join(load_dir, "encoder.pt"))
        )
        
        # Load decoder state if it's not shared with the encoder
        if not config["share_param"]:
            model.decoder.load_state_dict(
                torch.load(os.path.join(load_dir, "decoder.pt"))
            )
        
        print(f"Model loaded from {load_dir}")
        return model
