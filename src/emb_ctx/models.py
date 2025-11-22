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
    def __init__(self, tokenizer, encoder_model_name, decoder_model_name, share_param=False, task=None, use_lora=True, update=True):
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
        
        target_tok = tokenizer.encode("<|new_line|>")[0]
        src_tok = tokenizer.encode("\n")[0]
        self.encoder.model.embed_tokens.weight.data[target_tok] = self.encoder.model.embed_tokens.weight.data[src_tok].clone()
        self.encoder.lm_head.weight.data[target_tok] = self.encoder.lm_head.weight.data[src_tok].clone()


        target_tok = tokenizer.encode("<|new_line|>")[0]
        src_tok = tokenizer.encode("\n")[0]
        self.encoder.model.embed_tokens.weight.data[target_tok] = self.encoder.model.embed_tokens.weight.data[src_tok].clone()
        self.encoder.lm_head.weight.data[target_tok] = self.encoder.lm_head.weight.data[src_tok].clone()
        
        self.use_lora = use_lora
        self.update = update
    
        if use_lora:
            print("Using Lora...")
            encoder_lora_config = LoraConfig(
                    r=1024,
                    lora_alpha=2048,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    bias="none",
                    task_type="CAUSAL_LM"
                )
            
            self.encoder = get_peft_model(self.encoder, encoder_lora_config)

        self.decoder = self.encoder if share_param else AutoModelForCausalLM.from_pretrained(decoder_model_name)

        # Enable gradient checkpointing to trade compute for memory
        # Set pad token if not defined
        self._set_pad_tokens()
        
        # Storage for encoder representation (for contrastive loss)
        self.last_encoder_rep = None
        
        self.task = task
        
        # Monkey-patch generate to route to custom generator when prefix embeds are provided
        try:
            original_encoder_generate = self.encoder.generate
            original_decoder_generate = self.decoder.generate
            
            def _make_generate_wrapper(original_generate):
                def _wrapped_generate(model_self, *args, **kwargs):
                    # If caller provides inputs_embeds (prefix), use custom path
                    prefix = kwargs.get("inputs_embeds", None)
                    if prefix is not None:
                        max_new_tokens = kwargs.get("max_new_tokens", 32)
                        return self._custom_generate_from_prefix_with_model(
                            model_self, prefix, max_new_tokens=max_new_tokens
                        )
                    # Otherwise, fall back to the original generate
                    return original_generate(*args, **kwargs)
                return _wrapped_generate
            
            self.encoder.generate = _make_generate_wrapper(original_encoder_generate).__get__(self.encoder, self.encoder.__class__)
            self.decoder.generate = _make_generate_wrapper(original_decoder_generate).__get__(self.decoder, self.decoder.__class__)
        except Exception as e:
            print(f"Warning: failed to patch generate methods: {e}")

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
        encoder_outputs = self.encoder.model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            output_hidden_states=True,  # we only need the last hidden state
        )
        encoder_hidden_states = encoder_outputs.hidden_states[-1]
        
        # Get representation for last token
        last_token_indices = encoder_attention_mask.sum(dim=1) - 1
        batch_range = torch.arange(batch_size, device=device)
        encoder_last_token_hidden_state = encoder_hidden_states[batch_range, last_token_indices]
        
        # Create prefix for decoder (unsqueeze for sequence dimension)
        prefix = encoder_last_token_hidden_state.unsqueeze(1)
        
        # Dropout
        prefix = F.dropout(prefix, p=0.2, training=self.training)
        
        # Get decoder token embeddings
        if self.use_lora:
            decoder_embeds = self.decoder.model.model.embed_tokens(decoder_input_ids)
        else:
            decoder_embeds = self.decoder.model.embed_tokens(decoder_input_ids)
        
        # Concatenate prefix to decoder embeddings
        decoder_embeds = torch.cat([prefix, decoder_embeds], dim=1)
        
        # Create corresponding attention mask for prefix (always 1)
        prefix_attention_mask = torch.ones((decoder_attention_mask.size(0), 1), device=device)
        full_decoder_attention_mask = torch.cat([prefix_attention_mask, decoder_attention_mask], dim=1)
        
        # Run decoder with combined embeddings and mask
        decoder_outputs = self.decoder.model(
            inputs_embeds=decoder_embeds,
            attention_mask=full_decoder_attention_mask,
            output_hidden_states=True,  # we only need the final hidden state
        )
        sequence_output = decoder_outputs.hidden_states[-1]
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
        self.last_encoder_rep = encoder_rep.detach()
        
        return ce_loss, logits, {
            "ce_loss": ce_loss.detach(),
            "contrastive_loss": torch.tensor(0.0, device=ce_loss.device)  # Placeholder
        }

    def _custom_generate_from_prefix_with_model(self, lm, prefix, max_new_tokens=32):
        """
        Custom greedy generation that:
        1) Prefills past_key_values using the given prefix embeddings
        2) Iteratively generates tokens using past_key_values and cache_position
        Stops on EOS or if a single token repeats excessively (degeneracy guard).
        """
        device = prefix.device
        batch_size = prefix.size(0)
        prefix_len = prefix.size(1)

        eos_id = lm.config.eos_token_id
        pad_id = lm.config.pad_token_id if lm.config.pad_token_id is not None else eos_id

        # 1) Prefill using inputs_embeds to compute first-step distribution
        prefix_attn = torch.ones((batch_size, prefix_len), dtype=torch.long, device=device)
        prefill_out = lm(
            inputs_embeds=prefix,
            attention_mask=prefix_attn,
            use_cache=True,
            cache_position=torch.arange(0, prefix_len, device=device),
        )
        past_key_values = prefill_out.past_key_values
        # Next token distribution from last position of prefix
        logits = prefill_out.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)

        generated = torch.full((batch_size, max_new_tokens), fill_value=pad_id, dtype=torch.long, device=device)
        finished = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        # Track simple repetition to avoid degenerate loops
        repeat_run = torch.zeros((batch_size,), dtype=torch.long, device=device)
        last_token = next_token.clone()

        generated[:, 0] = next_token
        if eos_id is not None:
            finished = finished | (next_token == eos_id)

        # 2) Iterative continuation
        for step in range(1, max_new_tokens):
            # For finished sequences, keep feeding eos to avoid changing the cache
            feed_when_finished = torch.full_like(next_token, eos_id) if eos_id is not None else next_token
            input_step = torch.where(finished, feed_when_finished, next_token).unsqueeze(1)

            # Continue cache positions right after the prefix + previous steps
            cp = torch.arange(prefix_len + step - 1, prefix_len + step, device=device)

            out = lm(
                input_ids=input_step,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cp,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)

            # Write tokens only for unfinished sequences
            to_write = (~finished)
            if to_write.any():
                generated[to_write, step] = next_token[to_write]

            # Update repetition counters
            same_as_last = (next_token == last_token)
            repeat_run = torch.where(same_as_last, repeat_run + 1, torch.zeros_like(repeat_run))
            last_token = next_token

            # Base EOS stop
            if eos_id is not None:
                finished = finished | (next_token == eos_id)

            # Degeneracy guard: if same token repeats many times, stop
            finished = finished | (repeat_run >= 10)

            if torch.all(finished):
                break

        return generated

    def _custom_generate_from_prefix(self, prefix, max_new_tokens=32):
        # Backwards-compatible wrapper: use decoder by default
        return self._custom_generate_from_prefix_with_model(self.decoder, prefix, max_new_tokens)

    def generate_with_encoder(self, prefix, max_new_tokens=32):
        # Allow generation using the encoder LM (when sharing params or for diagnostics)
        return self._custom_generate_from_prefix_with_model(self.encoder, prefix, max_new_tokens)

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
            encoder_outputs = self.encoder.model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                output_hidden_states=True,
            )
            encoder_hidden_states = encoder_outputs.hidden_states[-1]
            last_token_indices = encoder_attention_mask.sum(dim=1) - 1
            batch_range = torch.arange(batch_size, device=device)
            encoder_last_token_hidden_state = encoder_hidden_states[batch_range, last_token_indices]
            
            # Use encoder hidden state directly without projection
            prefix_for_generation = encoder_last_token_hidden_state.unsqueeze(1)

            # Generate tokens using custom greedy loop (prefill + cache)
            generated_ids = self._custom_generate_from_prefix(prefix_for_generation, max_new_tokens=32)
            
            # Decode generations
            try:
                generated_texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            except Exception as e:
                print("ERROR", e)
                generated_texts = ["" for g in generated_ids]
            
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