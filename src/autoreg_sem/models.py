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
        Encoder and decoder can be optionally frozen based on freeze arg.
        Adds projection layers between encoder-latent_model and latent_model-decoder.
        """
        super().__init__()
        self.task = task
        self.dropout_rate = 0.2 # TODO: Make this configurable

        self.encoder = AutoModelForCausalLM.from_pretrained(encoder_path)
        self.latent_model = AutoModelForCausalLM.from_pretrained(latent_model_path)

        if share_param:
            # Note: If shared, the decoder will also follow the freeze setting below.
            self.decoder = self.encoder
        else:
            if "large" in decoder_path.lower():
                print("Loading PeftModel for Decoder...")
                self.decoder = PeftModel.from_pretrained(
                    self.latent_model,               # or self.encoder, whichever backbone
                    decoder_path,            # dir with adapter_config.json + .safetensors
                )
            else:
                self.decoder = AutoModelForCausalLM.from_pretrained(
                    decoder_path,
                )

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
        hidden_size1 = self.encoder.config.hidden_size
        hidden_size2 = self.latent_model.config.hidden_size
        hidden_size3 = self.decoder.config.hidden_size

        self.encoder_to_latent_model_proj = nn.Sequential(
            nn.Linear(hidden_size1, (hidden_size1 + hidden_size2) // 2),
            nn.GELU(),
            nn.Linear((hidden_size1 + hidden_size2) // 2, hidden_size2)
        )
        self.latent_model_to_decoder_proj = nn.Sequential(
            nn.Linear(hidden_size2, (hidden_size2 + hidden_size3) // 2),
            nn.GELU(),
            nn.Linear((hidden_size2 + hidden_size3) // 2, hidden_size3)
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

    def build_decoder_inputs(self, batch_encoder_input_ids, valid_steps_hidden, steps_attention_mask, steps_valid_mask):
        """
        Builds decoder inputs and targets by concatenating question embeddings
        with latent embeddings extracted from steps.
        """
        device = batch_encoder_input_ids.device
        batch_size = batch_encoder_input_ids.size(0)
        total_steps = valid_steps_hidden.size(0)
        step_token_lengths = steps_attention_mask.sum(dim=1)
        last_token_idx = (step_token_lengths - 1).clamp(min=0)
        step_indices = torch.arange(total_steps, device=device)
        latent_step_embeds_all = valid_steps_hidden[step_indices, last_token_idx, :]

        latent_inputs_list = []
        latent_targets_list = []
        attn_mask_list = []
        latent_labels_list = []
        pointer = 0

        for i in range(batch_size):
            enc_ids = batch_encoder_input_ids[i]
            newline_positions = (enc_ids == self.newline_token_id).nonzero(as_tuple=False)
            first_newline = newline_positions[0].item() if newline_positions.numel() > 0 else enc_ids.size(0) - 1
            question_embeds = self.latent_model.get_input_embeddings()(enc_ids[:first_newline+1])
            num_valid_steps = int(steps_valid_mask[i].sum().item())
            
            if num_valid_steps > 0:
                latent_embeds = latent_step_embeds_all[pointer:pointer+num_valid_steps]
                pointer += num_valid_steps
            else:
                latent_embeds = torch.empty((0, question_embeds.size(1)), device=device)
            
            decoder_input = torch.cat([question_embeds, latent_embeds], dim=0)
            question_zeros = torch.zeros_like(question_embeds)
            combined_targets = torch.cat([question_zeros, latent_embeds], dim=0)
            
            decoder_target = torch.cat([
                combined_targets[1:], 
                torch.zeros(1, question_embeds.size(1), device=device, dtype=question_embeds.dtype)
            ], dim=0)
            
            latent_inputs_list.append(decoder_input)
            latent_targets_list.append(decoder_target)
            attn_mask = torch.ones(decoder_target.size(0), dtype=torch.long, device=device)
            attn_mask_list.append(attn_mask)
            q_len = question_embeds.size(0)
            l_len = latent_embeds.size(0)
            T = decoder_target.size(0)
            latent_labels = torch.full((T,), -1, dtype=torch.long, device=device)
            if l_len > 0 and T - q_len > 1:
                latent_labels[q_len-1:T-2] = 0
                latent_labels[T-2] = 1
            latent_labels_list.append(latent_labels)
        
        all_latents_padded = pad_sequence(latent_inputs_list, batch_first=True, padding_value=0.0)
        all_latent_targets_padded = pad_sequence(latent_targets_list, batch_first=True, padding_value=0.0)
        attn_mask_padded = pad_sequence(attn_mask_list, batch_first=True, padding_value=0)
        latent_labels_padded = pad_sequence(latent_labels_list, batch_first=True, padding_value=-1)
        return all_latents_padded, all_latent_targets_padded, attn_mask_padded, latent_labels_padded

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
        
    def calculate_distance_loss(self, embeddings):
        """
        This function is no longer used and can be removed
        """
        pass

    def forward(self, encoder_input_ids, encoder_attention_mask, steps_input_ids, steps_attention_mask, steps_valid_mask, accelerator=None):
        """
        Computes the forward pass:
          - Uses the encoder to extract latent representations (frozen if freeze=True).
          - Projects encoder outputs and builds decoder inputs.
          - Computes predictions via the decoder.
          - Projects decoder outputs.
          - Computes a cross-entropy loss via the decoder branch (frozen if freeze=True).
        """
        device = encoder_input_ids.device
        valid_steps = steps_input_ids[steps_valid_mask]
        valid_attention = steps_attention_mask[steps_valid_mask]

        # Use no_grad context only if the encoder is frozen
        if next(self.encoder.parameters()).requires_grad == False:  # Check if encoder is frozen
            with torch.no_grad():  # Encoder is frozen
                enc_outputs = self.encoder.transformer(
                    input_ids=valid_steps,
                    attention_mask=valid_attention,
                    return_dict=True
                )
                encoder_hidden_states = enc_outputs.last_hidden_state
        else:  # Encoder is trainable
            enc_outputs = self.encoder.transformer(
                input_ids=valid_steps,
                attention_mask=valid_attention,
                return_dict=True
            )
            encoder_hidden_states = enc_outputs.last_hidden_state

        # Project encoder hidden states before passing to decoder
        projected_encoder_hidden = self.encoder_to_latent_model_proj(encoder_hidden_states)

        # Apply dropout AFTER projection
        projected_encoder_hidden = F.dropout(projected_encoder_hidden, p=self.dropout_rate, training=self.training)

        all_latent_inputs, all_latent_targets, decoder_attention_mask, latent_ce_labels = self.build_decoder_inputs(
            encoder_input_ids, projected_encoder_hidden, valid_attention, steps_valid_mask
        )
        decoder_outputs = self.latent_model.transformer(
            inputs_embeds=all_latent_inputs,
            return_dict=True # Ensure return_dict is True if accessing by name
        )
        # last_hidden_state already has ln_f applied by the transformer model
        decoder_hidden = decoder_outputs.last_hidden_state

        # Apply dropout to decoder_hidden.
        decoder_hidden_dropped = F.dropout(decoder_hidden, p=self.dropout_rate, training=self.training)

        target_mask = (all_latent_targets.abs().sum(dim=-1) != 0)

        # Project the relevant decoder hidden states (after dropout)
        # No detach needed before projection; no_grad context for decoder handles gradients
        decoder_prefix_to_project = decoder_hidden_dropped[target_mask]
        decoder_prefix_projected = self.latent_model_to_decoder_proj(decoder_prefix_to_project)
        decoder_prefix = decoder_prefix_projected.unsqueeze(1) # Add sequence dimension

        valid_steps_input_ids = steps_input_ids[steps_valid_mask]
        valid_steps_attention_mask = steps_attention_mask[steps_valid_mask]
        
        # with torch.no_grad(): # Decoder is frozen
        if True:
            decoder_embeds = self.decoder.transformer.wte(valid_steps_input_ids)
            # Concatenate the *projected* decoder prefix with the *original* target token embeddings
            decoder_embeds_combined = torch.cat([decoder_prefix, decoder_embeds], dim=1)
            decoder_attention_mask = torch.cat([torch.ones((valid_steps_attention_mask.size(0), 1), device=device, dtype=torch.long), valid_steps_attention_mask], dim=1)

            # torch.cuda.empty_cache()

            decoder_outputs = self.decoder(
                inputs_embeds=decoder_embeds_combined,
                attention_mask=decoder_attention_mask,
                return_dict=True
            )
            decoder_logits = decoder_outputs.logits

        shift_logits = decoder_logits.contiguous()
        shift_labels = valid_steps_input_ids.contiguous()
        shift_labels = torch.cat([shift_labels, torch.full((shift_labels.size(0), 1), self.tokenizer.eos_token_id, device=device, dtype=shift_labels.dtype)], dim=1)

        pad_token_id = self.latent_model.config.pad_token_id
        ce_mask = torch.cumsum((shift_labels == pad_token_id).to(torch.int), dim=1) <= 1

        predictions = shift_logits.view(-1, shift_logits.size(-1))
        targets = shift_labels.view(-1)

        ce_mask_flat = ce_mask.view(-1)
        loss_fct = nn.CrossEntropyLoss()

        ce_loss = loss_fct(predictions[ce_mask_flat], targets[ce_mask_flat])
        cont_mask = latent_ce_labels >= 0

        # Use original (non-dropped) decoder_hidden for contrastive/MSE loss calculations if needed
        if self.use_cont and cont_mask.sum() > 0:
            pred_latents = decoder_hidden[cont_mask] # Use original decoder_hidden here
            target_latents = all_latent_targets[cont_mask]

            # normalize
            pred_latents_norm = F.normalize(pred_latents, p=2, dim=-1)
            target_latents_norm = F.normalize(target_latents, p=2, dim=-1)

            similarity_matrix = torch.matmul(pred_latents_norm, target_latents_norm.T)
            temperature = 0.1
            local_batch_size = pred_latents.size(0)
            labels = torch.arange(local_batch_size).to(device)

            contrastive_loss = F.cross_entropy(similarity_matrix / temperature, labels)
        else:
            contrastive_loss = torch.tensor(0.0, device=device, dtype=decoder_hidden.dtype)

        # Add all applicable losses to the total loss
        total_loss = ce_loss + contrastive_loss

        return {
            "ce_loss": ce_loss,
            "cont_loss": contrastive_loss,
            "total_loss": total_loss,
        }

    def test(
        self,
        encoder_input_ids,
        encoder_attention_mask,
        steps_input_ids,
        steps_attention_mask,
        steps_valid_mask,
        step,
        mode,
    ):
        """
        Evaluates the model on a test batch and returns accuracy metrics.
        """
        # Set all modules to evaluation mode.
        self.encoder.eval()
        self.latent_model.eval()
        self.decoder.eval()

        gt_labels = self._extract_ground_truth(encoder_input_ids)
        device = encoder_input_ids.device

        # Part I: Measure continuous accuracy.
        cont, cont_pos, grouped_outputs = self._measure_cont_acc(encoder_input_ids, gt_labels, device)

        # Part II: Measure discretized accuracy.
        disc_acc, disc_pos_acc, disc_outputs = self._measure_disc_acc(encoder_input_ids, gt_labels, device, mode=mode)

        # Log sample predictions if in test mode.
        if mode == "test":
            self._log_sample_predictions(step, encoder_input_ids, gt_labels, disc_outputs)

        return {
            "cont": cont,
            "cont_pos": cont_pos,
            "disc_acc": disc_acc,
            "disc_pos_acc": disc_pos_acc
        }

    def _extract_ground_truth(self, encoder_input_ids):
        """
        Extracts the ground truth labels from the encoder input IDs.
        Assumes the answer is the part after the first newline.
        """
        gt_labels = [
            self.tokenizer.decode(encoder_input_ids[i], skip_special_tokens=True)
            for i in range(encoder_input_ids.size(0))
        ]
        gt_labels = [x[x.index("\n"):] if "\n" in x else "" for x in gt_labels]
        return gt_labels

    def _measure_cont_acc(self, encoder_input_ids, gt_labels, device):
        """
        Part I: Generate latent representations using the decoder,
        translate them, and calculate continuous accuracy.
        """
        # Build initial decoder inputs.
        decoder_inputs, decoder_att_mask, decoder_position_ids = self.pad_question(encoder_input_ids)
        decoder_inputs_embeds = self.latent_model.transformer.wte(decoder_inputs)
        N = 10

        # Generate latent tokens over N iterations.
        for _ in range(N):
            with torch.no_grad():
                decoder_out = self.latent_model.transformer(
                    inputs_embeds=decoder_inputs_embeds,
                    attention_mask=decoder_att_mask,
                    position_ids=decoder_position_ids,
                    return_dict=True # Use return_dict
                )
                # Use last_hidden_state directly, it already includes ln_f
                decoder_hidden = decoder_out.last_hidden_state
                last_hidden = decoder_hidden[:, -1, :].unsqueeze(1)

                decoder_inputs_embeds = torch.cat([decoder_inputs_embeds, last_hidden], dim=1)
                decoder_att_mask = torch.cat(
                    [decoder_att_mask, torch.ones((decoder_att_mask.size(0), 1), device=device)],
                    dim=1,
                )
                new_position_ids = decoder_position_ids[:, -1] + 1
                decoder_position_ids = torch.cat(
                    [decoder_position_ids, new_position_ids.unsqueeze(1)], dim=1
                )

        # Extract the generated latent representations.
        target_out = decoder_inputs_embeds[:, -N:].reshape(-1, decoder_inputs_embeds.size(-1))
        target_out_projected = self.latent_model_to_decoder_proj(target_out)
        
        with torch.no_grad():
            translator_out = self.decoder.generate(
                inputs_embeds=target_out_projected.unsqueeze(1),
                do_sample=False,
                temperature=0,
            )

        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in translator_out]
        grouped_outputs = [
            "".join(decoded_outputs[i : i + N]) for i in range(0, len(decoded_outputs), N)
        ]

        # Calculate continuous accuracy metrics.
        cont = 0
        cont_pos = 0
        for a, b in zip(grouped_outputs, gt_labels):
            if self.task == "gsm8k":
                if compare_last_formula(a) == extract_final_answer(b, self.task):
                    cont_pos += 1
            if extract_final_answer(a, self.task) == extract_final_answer(b, self.task):
                cont += 1

        if self.task != "gsm8k":
            cont_pos = cont

        return cont, cont_pos, grouped_outputs

    def _measure_disc_acc(self, encoder_input_ids, gt_labels, device, mode="test"):
        """
        Part II: Iteratively generate outputs, project, re-encode, project, and calculate discretized accuracy.
        """
        # Create directory structure for saving embeddings
        embedding_dir = os.path.join("embeddings", mode)
        os.makedirs(embedding_dir, exist_ok=True)

        # Get next file index
        existing_files = [f for f in os.listdir(embedding_dir) if f.startswith('tensors') and f.endswith('.pkl')]
        next_index = len(existing_files) + 1

        decoder_inputs, decoder_att_mask, decoder_position_ids = self.pad_question(encoder_input_ids)
        decoder_inputs_embeds = self.latent_model.transformer.wte(decoder_inputs)
        N = 10
        results = ["" for _ in range(encoder_input_ids.size(0))]
        disc_acc = 0
        disc_pos_acc = 0

        for _ in range(N):
            with torch.no_grad():
                # 1. DECODER STEP
                decoder_out = self.latent_model.transformer(
                    inputs_embeds=decoder_inputs_embeds,
                    attention_mask=decoder_att_mask,
                    position_ids=decoder_position_ids,
                    return_dict=True # Use return_dict
                )
                # Use last_hidden_state directly, it already includes ln_f
                decoder_hidden = decoder_out.last_hidden_state
                last_hidden = decoder_hidden[:, -1, :].unsqueeze(1) # Unprojected latent [B, 1, H]

                # Update attention mask and position IDs.
                decoder_att_mask = torch.cat(
                    [decoder_att_mask, torch.ones((decoder_att_mask.size(0), 1), device=device)],
                    dim=1,
                )
                new_position_ids = decoder_position_ids[:, -1] + 1
                decoder_position_ids = torch.cat(
                    [decoder_position_ids, new_position_ids.unsqueeze(1)], dim=1
                )

                # 2. DECODER -> TRANSLATOR STEP
                target_out_unprojected = last_hidden.reshape(-1, last_hidden.size(-1)) # Unprojected latent [B, H]

                # Apply decoder-to-translator projection
                target_out_projected = self.latent_model_to_decoder_proj(target_out_unprojected)

                with torch.no_grad():
                    translator_out = self.decoder.generate(
                        inputs_embeds=target_out_projected.unsqueeze(1), # Use projected
                        max_new_tokens=128, # Add max_new_tokens
                        do_sample=False,
                        temperature=0,
                    )
                    decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in translator_out]
                    decoded_outputs = [x.strip() + "\n" for x in decoded_outputs]

                # 3. ENCODER STEP (Re-encoding projected translated output)
                enc_input_ids = self.tokenizer(
                    decoded_outputs, return_tensors="pt", padding=True, truncation=True
                ).input_ids.to(device)

                enc_attention = self.tokenizer(
                    decoded_outputs, return_tensors="pt", padding=True, truncation=True
                ).attention_mask.to(device)

                with torch.no_grad(): # Encoder is frozen
                    enc_outputs = self.encoder.transformer(
                        input_ids=enc_input_ids,
                        attention_mask=enc_attention,
                        return_dict=True,
                    )
                    # Use last_hidden_state directly, it already includes ln_f
                    encoder_hidden_states_full = enc_outputs.last_hidden_state # [B, SeqLen, H]

                # Extract relevant encoder hidden state (e.g., last non-padding token)
                batch_size, seq_len, _ = encoder_hidden_states_full.size()
                first_newlines = torch.full((batch_size,), seq_len - 1, dtype=torch.long, device=device)
                # Find the position of the first newline for each sample.
                for i in range(batch_size):
                    newline_idx = (enc_input_ids[i] == self.newline_token_id).nonzero(as_tuple=False)
                    if newline_idx.numel() > 0:
                        first_newlines[i] = newline_idx[0].item()
                    results[i] += decoded_outputs[i] # Append result for final comparison

                batch_range = torch.arange(batch_size, device=device)
                # Get the specific hidden state to feed back to the decoder
                encoder_hidden_states_unprojected = encoder_hidden_states_full[batch_range, first_newlines, :] # [B, H]

                # Apply encoder-to-decoder projection
                encoder_hidden_states_projected = self.encoder_to_latent_model_proj(encoder_hidden_states_unprojected)

                # Append projected encoder state to decoder inputs for next step
                decoder_inputs_embeds = torch.cat(
                    [decoder_inputs_embeds, encoder_hidden_states_projected.unsqueeze(1)], # Use projected
                    dim=1
                )

        # Calculate discretized accuracy.
        for a, b in zip(results, gt_labels):
            if self.task == "gsm8k":
                if compare_last_formula(a) == extract_final_answer(b, self.task):
                    disc_pos_acc += 1
            if extract_final_answer(a, self.task) == extract_final_answer(b, self.task):
                disc_acc += 1

        if self.task != "gsm8k":
            disc_pos_acc = disc_acc

        return disc_acc, disc_pos_acc, results

    def _log_sample_predictions(self, step, encoder_input_ids, gt_labels, predictions):
        """
        Logs a sample of predictions for visualization.
        """
        import random
        import pandas as pd
        import wandb

        # sample_indices = random.sample(range(len(predictions)), min(10, len(predictions)))
        # Instead, let's fix it!
        sample_indices = [i for i in range(len(predictions)) if i % 10 == 0][:10]

        if sample_indices:
            sampled_data = {
                "Input": [self.tokenizer.decode(encoder_input_ids[i]) for i in sample_indices],
                "Predictions": [predictions[i] for i in sample_indices],
                "Ground Truth": [gt_labels[i] for i in sample_indices],
                "Cont_Pos": [
                    compare_last_formula(predictions[i]) == compare_last_formula(gt_labels[i])
                    for i in sample_indices
                ],
                "Cont": [
                    extract_final_answer(predictions[i], self.task)
                    == extract_final_answer(gt_labels[i], self.task)
                    for i in sample_indices
                ],
            }
            # df = pd.DataFrame(sampled_data)
            # table = wandb.Table(dataframe=df)
            # wandb.log({f"Saved Predictions (Step {step})": table, "step": step})

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
        self.encoder.save_pretrained(os.path.join(save_dir, "encoder"))
        self.latent_model.save_pretrained(os.path.join(save_dir, "latent_model"))
        self.decoder.save_pretrained(os.path.join(save_dir, "decoder"))
        
        print(f"Model saved to {save_dir}")
        
    @classmethod
    def load_model(cls, load_dir, tokenizer=None, task=None, use_cont=None, freeze=False):
        """
        Load a saved model from directory.
        
        Args:
            load_dir (str): Directory containing saved model files
            tokenizer (Optional): Pre-loaded tokenizer. If None, will load from saved files
            task (str, Optional): Task name, overrides the saved task if provided
            use_cont (bool, Optional): Whether to use contrastive loss, overrides saved setting if provided
            
        Returns:
            AutoRegressiveModel: Loaded model instance
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
        
        # Create model instance
        model = cls(
            tokenizer=tokenizer,
            encoder_path=config["encoder_path"],
            latent_model_path=config["latent_model_path"],
            decoder_path=config["decoder_path"],
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
