import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import pandas as pd
import wandb
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import types
from peft import LoraConfig, get_peft_model

class AutoEncoderModel(nn.Module):
    def __init__(self, tokenizer, encoder_model_name, decoder_model_name, share_param=False, use_lora=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = AutoModelForCausalLM.from_pretrained(encoder_model_name)
        self.use_lora = use_lora

        print("Share Param", share_param)
        
        if self.use_lora:
            print("USING LORA...")
            encoder_lora_config = LoraConfig(
                r=1024,
                lora_alpha=2048,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.encoder = get_peft_model(self.encoder, encoder_lora_config)

        if share_param:
            self.decoder = self.encoder
        else:
            self.decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name) # might need more ablation.

        # Set pad token if not defined
        if self.encoder.config.pad_token_id is None:
            self.encoder.config.pad_token_id = self.encoder.config.eos_token_id
        if self.decoder.config.pad_token_id is None:
            self.decoder.config.pad_token_id = self.decoder.config.eos_token_id

        # Resize embeddings to accommodate any newly added special tokens (e.g., <|new_line|>)
        try:
            vocab_size = len(self.tokenizer)
            self.encoder.resize_token_embeddings(vocab_size)
            if not share_param:
                self.decoder.resize_token_embeddings(vocab_size)
        except Exception:
            pass

        # Initialize <|new_line|> embedding/LM-head rows to match newline token for BOTH encoder and decoder
        try:
            new_line_id = self.tokenizer.convert_tokens_to_ids("<|new_line|>")
            nl_ids = self.tokenizer.encode("\n", add_special_tokens=False)
            if new_line_id is not None and len(nl_ids) > 0:
                src_id = nl_ids[0]
                # Encoder
                enc_emb = self.encoder.get_input_embeddings()
                with torch.no_grad():
                    enc_emb.weight.data[new_line_id] = enc_emb.weight.data[src_id].clone()
                    if hasattr(self.encoder, "lm_head") and self.encoder.lm_head.weight.shape[0] > new_line_id:
                        self.encoder.lm_head.weight.data[new_line_id] = self.encoder.lm_head.weight.data[src_id].clone()
                # Decoder (if not shared, else already covered)
                dec_emb = self.decoder.get_input_embeddings()
                with torch.no_grad():
                    dec_emb.weight.data[new_line_id] = dec_emb.weight.data[src_id].clone()
                    if hasattr(self.decoder, "lm_head") and self.decoder.lm_head.weight.shape[0] > new_line_id:
                        self.decoder.lm_head.weight.data[new_line_id] = self.decoder.lm_head.weight.data[src_id].clone()
        except Exception:
            pass

        # Monkey-patch Qwen-family generation to use inputs_embeds only on the first step,
        # then continue with input_ids + cache, aligning with HF PR #35890 behavior.
        try:
            model_type = getattr(self.decoder.config, "model_type", "") or ""
            if "qwen" in model_type:
                original_prepare = self.decoder.prepare_inputs_for_generation

                def patched_prepare_inputs_for_generation(self_model,
                                                         input_ids=None,
                                                         past_key_values=None,
                                                         inputs_embeds=None,
                                                         attention_mask=None,
                                                         cache_position=None,
                                                         **kwargs):
                    # Call the original first to let it compute defaults (e.g., cache_position/position_ids)
                    model_inputs = original_prepare(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        cache_position=cache_position,
                        **kwargs,
                    )

                    # Mirror upstream PR logic:
                    # 1) If inputs_embeds are passed and input_ids is an empty sequence, slice embeds with cache_position
                    mi_input_ids = model_inputs.get("input_ids", None)
                    cp = model_inputs.get("cache_position", None)
                    if inputs_embeds is not None and mi_input_ids is not None:
                        try:
                            if mi_input_ids.shape[1] == 0 and cp is not None:
                                # Keep only the unprocessed tokens
                                tail = cp.shape[0]
                                inputs_embeds = inputs_embeds[:, -tail:]
                        except Exception:
                            pass

                    # 2) Use inputs_embeds only on the (remaining) first-step; otherwise continue with input_ids
                    use_embeds = False
                    if inputs_embeds is not None and cp is not None:
                        try:
                            use_embeds = (len(cp) == inputs_embeds.shape[1])
                        except Exception:
                            use_embeds = False

                    if use_embeds:
                        model_inputs["inputs_embeds"] = inputs_embeds
                        model_inputs["input_ids"] = None
                    else:
                        model_inputs["inputs_embeds"] = None
                        # Do not inject dummy tokens; rely on generate to pass proper continuation ids

                    return model_inputs

                self.decoder.prepare_inputs_for_generation = types.MethodType(
                    patched_prepare_inputs_for_generation, self.decoder
                )
                print("[Info] Applied Qwen prepare_inputs_for_generation override for inputs_embeds continuation.")
        except Exception as e:
            print(f"[Warn] Failed to apply Qwen override: {e}")

    def forward(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask):
        encoder_outputs = self.encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )
        encoder_hidden_states = encoder_outputs.hidden_states[-1]

        batch_size = encoder_input_ids.size(0)
        # Get the index of the last non-padded token per sample.
        last_token_indices = encoder_attention_mask.sum(dim=1) - 1
        batch_range = torch.arange(batch_size, device=encoder_input_ids.device)
        # Use the last token's hidden state as a prefix.
        prefix = encoder_hidden_states[batch_range, last_token_indices].unsqueeze(1)

        # Get decoder token embeddings.
        decoder_input_embeddings = self.decoder.get_input_embeddings()
        decoder_embeds = decoder_input_embeddings(decoder_input_ids)
        # Concatenate the prefix to the decoder embeddings.
        decoder_embeds = torch.cat([prefix, decoder_embeds], dim=1)
        # Create a corresponding attention mask for the prefix (always 1).
        prefix_attention_mask = torch.ones((decoder_attention_mask.size(0), 1), device=decoder_attention_mask.device)
        decoder_attention_mask = torch.cat([prefix_attention_mask, decoder_attention_mask], dim=1)

        decoder_outputs = self.decoder(
            inputs_embeds=decoder_embeds,
            attention_mask=decoder_attention_mask
        )
        logits = decoder_outputs.logits

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

    def _custom_generate_from_prefix(self, prefix, max_new_tokens=32):
        """
        Custom greedy generation that:
        1) Prefills past_key_values using the given prefix embeddings
        2) Iteratively generates tokens using past_key_values and cache_position
        """
        device = prefix.device
        batch_size = prefix.size(0)
        prefix_len = prefix.size(1)

        eos_id = self.decoder.config.eos_token_id
        pad_id = self.decoder.config.pad_token_id if self.decoder.config.pad_token_id is not None else eos_id

        # 1) Prefill using inputs_embeds to compute first-step distribution
        prefix_attn = torch.ones((batch_size, prefix_len), dtype=torch.long, device=device)
        prefill_out = self.decoder(
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

            # Continue cache positions right after prefix + previous steps
            cp = torch.arange(prefix_len + step - 1, prefix_len + step, device=device)

            out = self.decoder(
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

            # No custom stop tokens: rely only on EOS and degeneracy guard

            # Degeneracy guard: if same token repeats many times, stop
            finished = finished | (repeat_run >= 10)

            if torch.all(finished):
                break

        return generated

    def test(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, step=None, sample_indices=None):
        self.encoder.eval()
        self.decoder.eval()
        batch_size = encoder_input_ids.shape[0]
        device = encoder_input_ids.device
        smooth_fn = SmoothingFunction().method1

        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                output_hidden_states=True,
            )
            encoder_hidden_states = encoder_outputs.hidden_states[-1]
            last_token_indices = encoder_attention_mask.sum(dim=1) - 1
            batch_range = torch.arange(batch_size, device=device)
            prefix = encoder_hidden_states[batch_range, last_token_indices].unsqueeze(1)

            ae_generated_ids = self._custom_generate_from_prefix(prefix, max_new_tokens=32)
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
            sample_indices = list(range(min(50, batch_size)))

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