import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import pandas as pd
import wandb
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import LoraConfig, get_peft_model

def calculate_contrastive_loss(rep1, rep2, temperature=0.5, weight=0.1):
    """
    Calculate NT-Xent (Normalized Temperature-scaled Cross Entropy) loss with in-batch negatives.
    This is the contrastive loss used in SimCLR and similar frameworks.
    
    Args:
        rep1: Representation from first model (restoration) [batch_size, dim]
        rep2: Representation from second model (prediction) [batch_size, dim]
        temperature: Temperature parameter to scale the similarity scores
        weight: Weight for contrastive loss
        
    Returns:
        Tuple of (weighted_contrastive_loss, raw_contrastive_loss)
    """
    # Normalize the representations
    rep1_norm = F.normalize(rep1, p=2, dim=1)
    rep2_norm = F.normalize(rep2, p=2, dim=1)
    
    device = rep1.device
    batch_size = rep1.shape[0]
    
    # Calculate similarity matrix between rep1 and rep2
    # [batch_size, batch_size]
    similarity_matrix = torch.matmul(rep1_norm, rep2_norm.transpose(0, 1))
    
    # Apply temperature scaling
    similarity_matrix = similarity_matrix / temperature
    
    # The positive pairs are the diagonal elements
    # Create labels where each anchor rep1[i] should be most similar to rep2[i]
    labels = torch.arange(batch_size, device=device)
    
    # Apply cross-entropy loss 
    # Each row of similarity matrix represents logits for one anchor from rep1
    # The label indicates which element from rep2 is the positive pair
    loss = F.cross_entropy(similarity_matrix, labels)
    
    # Apply the weight
    weighted_loss = weight * loss
    
    return weighted_loss, loss

class ContrastiveStepPredictor(nn.Module):
    """
    Model for either restoration or prediction with option for contrastive learning.
    Contains a single encoder-decoder pair.
    
    Two separate instances of this model will be created:
    - One for restoration
    - One for prediction
    
    The contrastive functionality compares representations between the two models.
    """
    def __init__(self, tokenizer, encoder_model_name, decoder_model_name, share_param=False, task=None):
        """
        Initialize the model predictor.
        
        Args:
            tokenizer: Tokenizer to use
            encoder_model_name: Name of pretrained encoder model
            decoder_model_name: Name of pretrained decoder model
            share_param: Whether to share parameters between encoder and decoder
        """
        super().__init__()
        self.tokenizer = tokenizer
        
        # Single encoder-decoder pair
        self.encoder = AutoModelForCausalLM.from_pretrained(encoder_model_name)

        # encoder_lora_config = LoraConfig(
        #         r=256,
        #         lora_alpha=1024,
        #         lora_dropout=0.1,
        #         bias="none",
        #         task_type="CAUSAL_LM"
        #     )
        
        # self.encoder = get_peft_model(self.encoder, encoder_lora_config)
        self.decoder = self.encoder if share_param else AutoModelForCausalLM.from_pretrained(decoder_model_name)

        # Set pad token if not defined
        self._set_pad_tokens()
        
        # Storage for encoder representation (for contrastive loss)
        self.last_encoder_rep = None
        
        self.task = task

    def _set_pad_tokens(self):
        """Set pad tokens for all models if not already defined"""
        for model in [self.encoder, self.decoder]:
            # Always set pad token to eos token
            model.config.pad_token_id = model.config.eos_token_id
            print(f"Set pad_token_id to {model.config.pad_token_id} for {model.__class__.__name__}")

    def process_model(self, encoder_input_ids, encoder_attention_mask, 
                     decoder_input_ids, decoder_attention_mask):
        """
        Process inputs through encoder and decoder.
        
        Args:
            encoder_input_ids: Input IDs for encoder
            encoder_attention_mask: Attention mask for encoder
            decoder_input_ids: Input IDs for decoder
            decoder_attention_mask: Attention mask for decoder
            
        Returns:
            Tuple of (cross entropy loss, encoder representation, logits)
        """
        batch_size = encoder_input_ids.shape[0]
        device = encoder_input_ids.device
        
        # Get encoder outputs
        encoder_outputs = self.encoder.transformer(
            encoder_input_ids, attention_mask=encoder_attention_mask
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Get representation for last token
        last_token_indices = encoder_attention_mask.sum(dim=1) - 1
        batch_range = torch.arange(batch_size, device=device)
        encoder_last_token_hidden_state = encoder_hidden_states[batch_range, last_token_indices]
        
        # Create prefix for decoder (unsqueeze for sequence dimension)
        prefix = encoder_last_token_hidden_state.unsqueeze(1)
        
        # Dropout
        prefix = F.dropout(prefix, p=0.2, training=self.training)
        
        # Get decoder token embeddings
        decoder_embeds = self.decoder.transformer.wte(decoder_input_ids)
        
        # Concatenate prefix to decoder embeddings
        decoder_embeds = torch.cat([prefix, decoder_embeds], dim=1)
        
        # Create corresponding attention mask for prefix (always 1)
        prefix_attention_mask = torch.ones((decoder_attention_mask.size(0), 1), device=device)
        full_decoder_attention_mask = torch.cat([prefix_attention_mask, decoder_attention_mask], dim=1)
        
        # Run decoder with combined embeddings and mask
        decoder_outputs = self.decoder.transformer(
            inputs_embeds=decoder_embeds,
            attention_mask=full_decoder_attention_mask
        )
        sequence_output = decoder_outputs.last_hidden_state
        logits = self.decoder.lm_head(sequence_output)
        
        # Align logits and labels (remove the extra time step added by prefix)
        shift_logits = logits[:, 0:-1, :].contiguous()
        shift_labels = decoder_input_ids.contiguous()
        pad_token_id = self.decoder.config.pad_token_id
        
        # Create a mask to ignore padding tokens in loss computation
        mask = torch.cumsum((shift_labels == pad_token_id).to(torch.int), dim=1) <= 1
        
        predictions = shift_logits.view(-1, shift_logits.size(-1))
        targets = shift_labels.view(-1)
        mask_flat = mask.view(-1)
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(predictions[mask_flat], targets[mask_flat])
        
        return loss, encoder_last_token_hidden_state, logits

    def forward(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask):
        """
        Forward pass for the model.
        
        Args:
            encoder_input_ids: Input IDs for encoder
            encoder_attention_mask: Attention mask for encoder
            decoder_input_ids: Input IDs for decoder
            decoder_attention_mask: Attention mask for decoder
            
        Returns:
            Tuple of (loss, logits, loss_dict)
        """
        # Process the inputs through the model
        ce_loss, encoder_rep, logits = self.process_model(
            encoder_input_ids, encoder_attention_mask, 
            decoder_input_ids, decoder_attention_mask
        )
        
        # Store representation for contrastive learning
        self.last_encoder_rep = encoder_rep
        
        return ce_loss, logits, {
            "ce_loss": ce_loss,
            "contrastive_loss": torch.tensor(0.0, device=ce_loss.device)  # Placeholder
        }

    def test(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, step=None, sample_indices=None):
        """
        Test method for model evaluation.
        
        Args:
            encoder_input_ids: Input IDs for encoder
            encoder_attention_mask: Attention mask for encoder
            decoder_input_ids: Input IDs for decoder (ground truth)
            decoder_attention_mask: Attention mask for decoder
            step: Current step number (for logging)
            sample_indices: Indices of samples to visualize
            
        Returns:
            Dictionary of metrics and generated text
        """
        self.encoder.eval()
        self.decoder.eval()
        
        batch_size = encoder_input_ids.shape[0]
        device = encoder_input_ids.device
        smooth_fn = SmoothingFunction().method1

        with torch.no_grad():
            # Get encoder outputs
            encoder_outputs = self.encoder.transformer(
                encoder_input_ids, attention_mask=encoder_attention_mask
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
            last_token_indices = encoder_attention_mask.sum(dim=1) - 1
            batch_range = torch.arange(batch_size, device=device)
            encoder_last_token_hidden_state = encoder_hidden_states[batch_range, last_token_indices]
            
            # Use encoder hidden state directly without projection
            prefix_for_generation = encoder_last_token_hidden_state.unsqueeze(1)

            # Generate tokens
            generated_ids = self.decoder.generate(
                inputs_embeds=prefix_for_generation,
                max_new_tokens=32,
                do_sample=False,
                temperature=0.0
            )
            
            # Decode generations
            generated_texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            
            # Decode ground truth
            gt_sequences = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in decoder_input_ids]
            
            # Decode inputs (for debugging)
            input_sequences = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in encoder_input_ids]

        # Evaluate model
        exact_match = 0
        bleu_scores = []
        
        for pred, gt in zip(generated_texts, gt_sequences):            
            try:
                if pred.strip() == gt.strip():
                    exact_match += 1
                pred_tokens = pred.split()
                gt_tokens = gt.split()
                bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smooth_fn)
                bleu_scores.append(bleu)
            except Exception as e:
                print("ERROR", e)
                bleu_scores.append(0.0)
                    
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        total_steps = len(gt_sequences)

        # Pick some sample indices if not provided
        if sample_indices is None:
            sample_indices = list(range(min(5, batch_size)))

        # Build detailed sample table
        if len(sample_indices) > 0:
            sampled_data = {
                "Context (Question + Previous Steps)": [input_sequences[i] for i in sample_indices],
                "Predicted Next Step": [generated_texts[i] for i in sample_indices],
                "Ground Truth Next Step": [gt_sequences[i] for i in sample_indices],
                "BLEU": [bleu_scores[i] for i in sample_indices],
                "Exact Match": [
                    generated_texts[i].strip() == gt_sequences[i].strip() for i in sample_indices
                ],
            }

        return {
            "next_step_acc": exact_match,
            "total_steps": total_steps,
            "avg_bleu": avg_bleu,
        }