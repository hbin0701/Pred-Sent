import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import pandas as pd
import wandb
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class AutoEncoderModel(nn.Module):
    def __init__(self, tokenizer, encoder_model_name, decoder_model_name, share_param=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = AutoModelForCausalLM.from_pretrained(encoder_model_name)

        print("Share Param", share_param)
        
        if share_param:
            self.decoder = self.encoder
        else:
            self.decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name) # might need more ablation.

        # Set pad token if not defined
        if self.encoder.config.pad_token_id is None:
            self.encoder.config.pad_token_id = self.encoder.config.eos_token_id
        if self.decoder.config.pad_token_id is None:
            self.decoder.config.pad_token_id = self.decoder.config.eos_token_id

    def forward(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask):
        encoder_outputs = self.encoder.transformer(
            encoder_input_ids, attention_mask=encoder_attention_mask
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        batch_size = encoder_input_ids.size(0)
        # Get the index of the last non-padded token per sample.
        last_token_indices = encoder_attention_mask.sum(dim=1) - 1
        batch_range = torch.arange(batch_size, device=encoder_input_ids.device)
        # Use the last token's hidden state as a prefix.
        prefix = encoder_hidden_states[batch_range, last_token_indices].unsqueeze(1)

        # Get decoder token embeddings.
        decoder_embeds = self.decoder.transformer.wte(decoder_input_ids)
        # Concatenate the prefix to the decoder embeddings.
        decoder_embeds = torch.cat([prefix, decoder_embeds], dim=1)
        # Create a corresponding attention mask for the prefix (always 1).
        prefix_attention_mask = torch.ones((decoder_attention_mask.size(0), 1), device=decoder_attention_mask.device)
        decoder_attention_mask = torch.cat([prefix_attention_mask, decoder_attention_mask], dim=1)

        decoder_outputs = self.decoder.transformer(
            inputs_embeds=decoder_embeds,
            attention_mask=decoder_attention_mask
        )
        sequence_output = decoder_outputs.last_hidden_state
        logits = self.decoder.lm_head(sequence_output)

        # Align logits and labels (remove the extra time step added by the prefix).
        shift_logits = logits[:, 0:-1, :].contiguous()
        shift_labels = decoder_input_ids.contiguous()
        pad_token_id = self.decoder.config.pad_token_id
        # Create a mask to ignore padding tokens in the loss computation.
        mask = torch.cumsum((shift_labels == pad_token_id).to(torch.int), dim=1) <= 1

        predictions = shift_logits.view(-1, shift_logits.size(-1))
        targets = shift_labels.view(-1)
        mask_flat = mask.view(-1)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(predictions[mask_flat], targets[mask_flat])
        return loss, logits

    def test(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, step=None, sample_indices=None):
        self.encoder.eval()
        self.decoder.eval()
        batch_size = encoder_input_ids.shape[0]
        device = encoder_input_ids.device
        smooth_fn = SmoothingFunction().method1

        with torch.no_grad():
            encoder_outputs = self.encoder.transformer(
                encoder_input_ids, attention_mask=encoder_attention_mask
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
            last_token_indices = encoder_attention_mask.sum(dim=1) - 1
            batch_range = torch.arange(batch_size, device=device)
            prefix = encoder_hidden_states[batch_range, last_token_indices].unsqueeze(1)
            
            ae_generated_ids = self.decoder.generate(
                inputs_embeds=prefix,
                max_new_tokens=32,
                do_sample=False,
                temperature=0.0
            )
            
            ae_generated_texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in ae_generated_ids]
            gt_sequences = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in decoder_input_ids]
            input_sequences = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in encoder_input_ids]

        ae_exact_match = 0
        bleu_scores = []
        import random
        
        for pred, gt in zip(ae_generated_texts, gt_sequences):
            
            # if random.random() < 0.1:
            #     print(pred, gt)
            
            try:
                if pred.strip() == gt.strip():
                    ae_exact_match += 1
                pred_tokens = pred.split()
                gt_tokens = gt.split()
                bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smooth_fn)
                bleu_scores.append(bleu)
            except Exception:
                bleu_scores.append(0.0)

        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        total_steps = len(gt_sequences)

        # Pick some sample indices if not provided
        if sample_indices is None:
            sample_indices = list(range(min(5, batch_size)))

        # Build detailed sample table
        if len(sample_indices) > 0:
            sampled_data = {
                "Prediction": [ae_generated_texts[i] for i in sample_indices],
                "Ground Truth": [gt_sequences[i] for i in sample_indices],
                "BLEU": [bleu_scores[i] for i in sample_indices],
                "Exact Match": [
                    ae_generated_texts[i].strip() == gt_sequences[i].strip() for i in sample_indices
                ],
            }

            df = pd.DataFrame(sampled_data)
            table = wandb.Table(dataframe=df)
            step_info = {"step": step} if step is not None else {}
            wandb.log({f"Sample Predictions (Step {step})": table, **step_info})

        return {
            "ae_acc": ae_exact_match,
            "total_steps": total_steps,
            "avg_bleu": avg_bleu,
        }