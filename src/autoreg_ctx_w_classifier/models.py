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
from torch.profiler import profile, record_function, ProfilerActivity
from collections import defaultdict


class AutoEncoderModel(nn.Module):
    def __init__(self, tokenizer, encoder_path, decoder_path, translator_path, task, freeze, share_param, use_cont, use_dist=False, use_mse=False):
        """
        Loads the encoder, decoder, and translator models.
        Initializes the tokenizer from a fixed checkpoint.
        Encoder and Translator can be optionally frozen based on freeze arg.
        Adds projection layers between encoder-decoder and decoder-translator.
        """
        super().__init__()
        self.task = task
        self.dropout_rate = 0.2 # TODO: Make this configurable

        self.encoder = AutoModelForCausalLM.from_pretrained(encoder_path)
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_path)

        if share_param:
            # Note: If shared, the translator will also follow the freeze setting below.
            self.translator = self.encoder
        else:
            self.translator = AutoModelForCausalLM.from_pretrained(translator_path)

        # Freeze encoder & translator parameters only if freeze is True.
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.translator.parameters():
                param.requires_grad = False
        else:
            # If not freezing, ensure they are trainable
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.translator.parameters():
                param.requires_grad = True

        # Initialize the common tokenizer.
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache the newline token id.
        self.newline_token_id = self.tokenizer.encode("\n")[0]

        # Set pad token IDs.
        self.encoder.config.pad_token_id = self.tokenizer.eos_token_id
        self.decoder.config.pad_token_id = self.tokenizer.eos_token_id
        self.translator.config.pad_token_id = self.tokenizer.eos_token_id

        # Define projection layers
        hidden_size = self.encoder.config.hidden_size # Assuming encoder, decoder, translator have same hidden size
        self.encoder_to_decoder_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.decoder_to_translator_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Move models and projections to bfloat16.
        self.encoder.to(torch.bfloat16)
        self.decoder.to(torch.bfloat16)
        self.translator.to(torch.bfloat16)
        self.encoder_to_decoder_proj.to(torch.bfloat16)
        self.decoder_to_translator_proj.to(torch.bfloat16)

        self.use_cont = use_cont
        self.use_dist = use_dist
        self.use_mse = use_mse
        
        # Store configuration for saving/loading
        self.config = {
            "task": task,
            "freeze": freeze,
            "share_param": share_param,
            "use_cont": use_cont,
            "use_dist": use_dist,
            "use_mse": use_mse,
            "dropout_rate": self.dropout_rate
        }

    def build_decoder_inputs(self, batch_encoder_input_ids, valid_steps_hidden, steps_attention_mask, steps_valid_mask):
        """
        Builds decoder inputs and targets by concatenating question embeddings
        with latent embeddings extracted from steps.
        """
        device = batch_encoder_input_ids.device
        batch_size = batch_encoder_input_ids.size(0)
        # total_steps = valid_steps_hidden.size(0)
        step_token_lengths = steps_attention_mask.sum(dim=1)
        last_token_idx = (step_token_lengths - 1).clamp(min=0)
        # step_indices = torch.arange(total_steps, device=device)
        # latent_step_embeds_all = valid_steps_hidden[step_indices, last_token_idx, :]

        latent_inputs_list = []
        latent_targets_list = []
        attn_mask_list = []
        latent_labels_list = []
        pointer = 0

        for i in range(batch_size):
            enc_ids = batch_encoder_input_ids[i]
            newline_positions = (enc_ids == self.newline_token_id).nonzero(as_tuple=False)
            first_newline = newline_positions[0].item() if newline_positions.numel() > 0 else enc_ids.size(0) - 1
            question_embeds = self.decoder.get_input_embeddings()(enc_ids[:first_newline+1])
            num_valid_steps = int(steps_valid_mask[i].sum().item())
            
            # if num_valid_steps > 0:
            #     latent_embeds = latent_step_embeds_all[pointer:pointer+num_valid_steps]
            #     pointer += num_valid_steps
            # else:
            #     latent_embeds = torch.empty((0, question_embeds.size(1)), device=device)
            
            ## get latent embeds from valid_steps_hidden (newline_positions)            
            latent_embeds = valid_steps_hidden[i][newline_positions[:-1]].squeeze(1)
                                                                          
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
        Calculate distance loss to encourage maximum distance between embeddings.
        
        Args:
            embeddings: Tensor of shape [batch_size, hidden_size]
            
        Returns:
            distance_loss: Scalar tensor representing the distance loss
        """
        eps = 1e-8
        # Use F.normalize for normalization
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1, eps=eps)
        
        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
        
        # Clamp similarity values to avoid numerical issues
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0 + eps, max=1.0 - eps)
        
        # Remove diagonal (self-similarity)
        mask = ~torch.eye(similarity_matrix.shape[0], dtype=bool, device=similarity_matrix.device)
        
        # Convert to positive distance (1 - abs(similarity))
        # This creates a loss that pushes all embeddings apart
        # regardless of whether they are similar or dissimilar
        distance_loss = -torch.mean((1.0 - torch.abs(similarity_matrix[mask])))
        
        return distance_loss

    def forward(self, encoder_input_ids, encoder_attention_mask, steps_input_ids, steps_attention_mask, steps_valid_mask, accelerator=None):
        """
        Computes the forward pass:
          - Uses the encoder to extract latent representations (frozen if freeze=True).
          - Projects encoder outputs and builds decoder inputs.
          - Computes predictions via the decoder.
          - Projects decoder outputs.
          - Computes a cross-entropy loss via the translator branch (frozen if freeze=True).
        """
        device = encoder_input_ids.device
        valid_steps = steps_input_ids[steps_valid_mask]
        valid_attention = steps_attention_mask[steps_valid_mask]

        # Use no_grad context only if the encoder is frozen        
        # if next(self.encoder.parameters()).requires_grad == False:  # Check if encoder is frozen
        #     with torch.no_grad():  # Encoder is frozen
        #         enc_outputs = self.encoder.transformer(
        #             input_ids=valid_steps,
        #             attention_mask=valid_attention,
        #             return_dict=True
        #         )
        #         encoder_hidden_states = enc_outputs.last_hidden_state
        # else:  # Encoder is trainable
        #     enc_outputs = self.encoder.transformer(
        #         input_ids=valid_steps,
        #         attention_mask=valid_attention,
        #         return_dict=True
        #     )
        #     encoder_hidden_states = enc_outputs.last_hidden_state
        
        if next(self.encoder.parameters()).requires_grad == False:  # Check if encoder is frozen
            with torch.no_grad():  # Encoder is frozen
                encoder_outputs = self.encoder.transformer(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    return_dict=True
                )
                encoder_hidden_states = encoder_outputs.last_hidden_state
        else:  # Encoder is trainable
            encoder_outputs = self.encoder.transformer(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                return_dict=True
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
        

        # Project encoder hidden states before passing to decoder
        projected_encoder_hidden = self.encoder_to_decoder_proj(encoder_hidden_states)

        # Apply dropout AFTER projection
        projected_encoder_hidden = F.dropout(projected_encoder_hidden, p=self.dropout_rate, training=self.training)

        all_latent_inputs, all_latent_targets, decoder_attention_mask, latent_ce_labels = self.build_decoder_inputs(
            encoder_input_ids, projected_encoder_hidden, valid_attention, steps_valid_mask
        )
        decoder_outputs = self.decoder.transformer(
            inputs_embeds=all_latent_inputs,
            return_dict=True # Ensure return_dict is True if accessing by name
        )
        # last_hidden_state already has ln_f applied by the transformer model
        decoder_hidden = decoder_outputs.last_hidden_state

        # Apply dropout to decoder_hidden.
        decoder_hidden_dropped = F.dropout(decoder_hidden, p=self.dropout_rate, training=self.training)

        target_mask = (all_latent_targets.abs().sum(dim=-1) != 0)

        # Project the relevant decoder hidden states (after dropout)
        # No detach needed before projection; no_grad context for translator handles gradients
        translator_prefix_to_project = decoder_hidden_dropped[target_mask]
        translator_prefix_projected = self.decoder_to_translator_proj(translator_prefix_to_project)
        translator_prefix = translator_prefix_projected.unsqueeze(1) # Add sequence dimension

        valid_steps_input_ids = steps_input_ids[steps_valid_mask]
        valid_steps_attention_mask = steps_attention_mask[steps_valid_mask]
        
        # with torch.no_grad(): # Translator is frozen
        if True:
            translator_embeds = self.translator.transformer.wte(valid_steps_input_ids)
            # Concatenate the *projected* decoder prefix with the *original* target token embeddings
            translator_embeds_combined = torch.cat([translator_prefix, translator_embeds], dim=1)
            translator_attention_mask = torch.cat([torch.ones((valid_steps_attention_mask.size(0), 1), device=device, dtype=torch.long), valid_steps_attention_mask], dim=1)

            # torch.cuda.empty_cache()

            translator_outputs = self.translator(
                inputs_embeds=translator_embeds_combined,
                attention_mask=translator_attention_mask,
                return_dict=True
            )
            translator_logits = translator_outputs.logits

        shift_logits = translator_logits.contiguous()
        shift_labels = valid_steps_input_ids.contiguous()
        shift_labels = torch.cat([shift_labels, torch.full((shift_labels.size(0), 1), self.tokenizer.eos_token_id, device=device, dtype=shift_labels.dtype)], dim=1)

        pad_token_id = self.decoder.config.pad_token_id
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

            # Calculate distance loss only if use_dist is enabled
            if self.use_dist:
                # Calculate distance loss to encourage maximum distance between pred_latents
                distance_loss1 = self.calculate_distance_loss(pred_latents) # Use original decoder_hidden
                distance_loss2 = self.calculate_distance_loss(target_latents)
                distance_loss = distance_loss1 + distance_loss2
            else:
                distance_loss = torch.tensor(0.0, device=device, dtype=decoder_hidden.dtype)
                distance_loss1 = torch.tensor(0.0, device=device, dtype=decoder_hidden.dtype)
                distance_loss2 = torch.tensor(0.0, device=device, dtype=decoder_hidden.dtype)

            # Calculate MSE loss if enabled
            if self.use_mse:
                # MSE loss between predicted latents and target latents
                mse_loss = F.mse_loss(pred_latents, target_latents) # Use original decoder_hidden
            else:
                mse_loss = torch.tensor(0.0, device=device, dtype=decoder_hidden.dtype)

        else:
            contrastive_loss = torch.tensor(0.0, device=device, dtype=decoder_hidden.dtype)
            distance_loss = torch.tensor(0.0, device=device, dtype=decoder_hidden.dtype)
            distance_loss1 = torch.tensor(0.0, device=device, dtype=decoder_hidden.dtype)
            distance_loss2 = torch.tensor(0.0, device=device, dtype=decoder_hidden.dtype)
            mse_loss = torch.tensor(0.0, device=device, dtype=decoder_hidden.dtype)

        # Add all applicable losses to the total loss
        # Note: Gradients will flow back through decoder and projection layers, but stop at encoder and translator.
        total_loss = ce_loss + contrastive_loss
        if self.use_dist:
            total_loss = total_loss + distance_loss
        if self.use_mse:
            total_loss = total_loss + mse_loss

        return {
            "ce_loss": ce_loss,
            "cont_loss": contrastive_loss,
            "distance_loss": distance_loss,
            "distance_loss1": distance_loss1,
            "distance_loss2": distance_loss2,
            "mse_loss": mse_loss,
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
        self.decoder.eval()
        self.translator.eval()

        gt_labels = self._extract_ground_truth(encoder_input_ids)
        device = encoder_input_ids.device

        # Part I: Measure standard accuracy.
        acc, pos_acc, prev_pos_acc, grouped_outputs, acc_flops = self._measure_acc(encoder_input_ids, gt_labels, device)

        # Part II: Measure semi-accuracy.
        # semi_acc, semi_pos_acc, semi_prev_pos_acc, semi_outputs, semi_flops = self._measure_semi_acc(encoder_input_ids, gt_labels, device, mode=mode)
        
        semi_acc, semi_pos_acc, semi_prev_pos_acc, semi_outputs, semi_flops = acc, pos_acc, prev_pos_acc, grouped_outputs, acc_flops

        # Part III: Measure GT accuracy (if applicable to task).
        
        # Comment this out.
        if False: 
        # if self.task in ["gsm8k", "prosqa", "fw-edu", "csqa"]:
            stage_acc_dict, gt_ls_acc, gt_non_ls_acc, total_steps = self._measure_gt_acc(
                encoder_input_ids,
                encoder_attention_mask,
                steps_input_ids,
                steps_attention_mask,
                steps_valid_mask,
                device,
            )
        else:
            stage_acc_dict, gt_ls_acc, gt_non_ls_acc, total_steps = {}, 0, 0, 0

        # Log sample predictions if in test mode.
        if mode == "test":
            self._log_sample_predictions(step, encoder_input_ids, gt_labels, semi_outputs)

        return {
            "acc": acc,
            "pos_acc": pos_acc,
            "prev_pos_acc": acc + prev_pos_acc,
            "semi_acc": semi_acc,
            "semi_pos_acc": semi_pos_acc,
            "semi_prev_pos_acc": semi_acc + semi_prev_pos_acc,
            "total_steps": total_steps,
            "gt_ls_acc": gt_ls_acc,
            "gt_non_ls_acc": gt_non_ls_acc,
            "acc_flops": acc_flops,
            "semi_flops": semi_flops,
            **stage_acc_dict,  # e.g., "stage0_acc", "stage1_acc", etc.
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


    def _measure_acc(self, encoder_input_ids, gt_labels, device):
        """
        Part I: Generate latent representations using the decoder,
        translate them, and calculate exact/near-match accuracy.
        """
        # Build initial decoder inputs.
        decoder_inputs, decoder_att_mask, decoder_position_ids = self.pad_question(encoder_input_ids)
        N = 7
        
        # Initialize profiler
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        total_flops = 0
        
        # Generate latent tokens over N iterations.
        with profile(activities=activities, record_shapes=True, profile_memory=True, with_flops=True) as prof:
            decoder_inputs_embeds = self.decoder.transformer.wte(decoder_inputs)

            past_key_values = None
            for _ in range(N):
                with torch.no_grad():
                    with record_function("decoder_step"):
                        if past_key_values is None:
                            decoder_out = self.decoder.transformer(
                                inputs_embeds=decoder_inputs_embeds,
                                attention_mask=decoder_att_mask,
                                position_ids=decoder_position_ids,
                                return_dict=True # Use return_dict
                            )
                        else:
                            # Subsequent iterations: only process the new token.
                            # We assume the new token is the last token in decoder_inputs_embeds.
                            new_token_embed = decoder_inputs_embeds[:, -1:, :]  # shape: (B, 1, H)
                            new_token_pos = decoder_position_ids[:, -1:]         # shape: (B, 1)
                            # (Optionally, you can pass only the new token's attention mask if your model supports it.)
                            new_token_att = decoder_att_mask[:, -1:]               # shape: (B, 1)

                            decoder_out = self.decoder.transformer(
                                inputs_embeds=new_token_embed,
                                attention_mask=new_token_att,
                                position_ids=new_token_pos,
                                past_key_values=past_key_values,  # Use the cached keys/values.
                                use_cache=True,
                            )
                        past_key_values = (
                            decoder_out.past_key_values if hasattr(decoder_out, "past_key_values") 
                            else decoder_out[1]
                        )
                        # Use last_hidden_state directly, it already includes ln_f
                        decoder_hidden = decoder_out.last_hidden_state
                        last_hidden = decoder_hidden[:, -1, :].unsqueeze(1)
                        
                        ## classifier 
                        # 2. DECODER -> TRANSLATOR STEP
                        target_out_unprojected = last_hidden.reshape(-1, last_hidden.size(-1)) # Unprojected latent [B, H]

                        with record_function("projection_step"):
                            # Apply decoder-to-translator projection
                            target_out_projected = self.decoder_to_translator_proj(target_out_unprojected)

                        with record_function("translator_step"):
                            ## generate one token
                            translator_out = self.translator.generate(
                                inputs_embeds=target_out_projected.unsqueeze(1), # Use projected
                                max_new_tokens=1, # Add max_new_tokens
                                do_sample=False,
                                temperature=0,
                            )
                            
                            decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in translator_out]
                            
                            
                            # print(decoded_outputs)
                            # print(f"Decoded outputs: {decoded_outputs}")
                            
                            text = decoded_outputs[0]
                            if text == "###":
                                inputs_embeds = target_out_projected.unsqueeze(1)
                                outputs = self.translator(
                                    inputs_embeds=inputs_embeds,
                                    use_cache=True,
                                )
                                past_key_values_translator = outputs.past_key_values if hasattr(outputs, "past_key_values") else outputs[1]

                                next_token_logits = outputs.logits[:, -1, :]
                                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                                
                                answer = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                                
                                # decoded_outputs_str += self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                                for i in range(128):
                                    outputs = self.translator(
                                        input_ids=next_token,
                                        past_key_values=past_key_values_translator,
                                        use_cache=True,
                                    )
                                    past_key_values_translator = outputs.past_key_values if hasattr(outputs, "past_key_values") else outputs[1]
                                    
                                    next_token_logits = outputs.logits[:, -1, :]
                                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)                                        
                                    
                                    if next_token[0] == self.tokenizer.eos_token_id:
                                        break
                                    
                                    answer += self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                                
                                    
                                # Stop generating if the model generates "###"
                                break


                        decoder_inputs_embeds = torch.cat([decoder_inputs_embeds, last_hidden], dim=1)
                        decoder_att_mask = torch.cat(
                            [decoder_att_mask, torch.ones((decoder_att_mask.size(0), 1), device=device)],
                            dim=1,
                        )
                        new_position_ids = decoder_position_ids[:, -1] + 1
                        decoder_position_ids = torch.cat(
                            [decoder_position_ids, new_position_ids.unsqueeze(1)], dim=1
                        )
                        
                        decoder_inputs_embeds = last_hidden
                        decoder_att_mask = torch.ones((decoder_att_mask.size(0), 1), device=device)
                        new_position_ids = decoder_position_ids[:, -1] + 1
                        decoder_position_ids = new_position_ids.unsqueeze(1)
                        
       
        print(f"answer: {answer}")
        # Calculate total FLOPs from profiler
        total_flops = sum(event.flops for event in prof.key_averages() if event.flops > 0)
        
        print(f"Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")
        
        # print(f"Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")
        
        # # Calculate FLOPs by record_function
        # flops_by_record_function = {}
        # for event in prof.key_averages():
        #     if event.key in ["decoder_step", "projection_step", "translator_step"] and event.flops > 0:
        #         flops_by_record_function[event.key] = event.flops / 1e9  # Convert to GFLOPs

        # # Print FLOPs breakdown by record_function
        # print("FLOPs breakdown by record_function:")
        # for key, flops in flops_by_record_function.items():
        #     print(f"  - {key}: {flops:.2f} GFLOPs")


        # decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in translator_out]
        # grouped_outputs = [
        #     "".join(decoded_outputs[i : i + N]) for i in range(0, len(decoded_outputs), N)
        # ]

        # # Calculate accuracy metrics.
        # acc = 0
        # pos_acc = 0
        # prev_pos_acc = 0
        # for a, b in zip(grouped_outputs, gt_labels):
        #     if self.task == "gsm8k":
        #         if compare_last_formula(a) == extract_final_answer(b, self.task):
        #             pos_acc += 1
        #     if extract_final_answer(a, self.task) == extract_final_answer(b, self.task):
        #         acc += 1
        #     else:
        #         if compare_last_formula(a) == compare_last_formula(b):
        #             prev_pos_acc += 1

        # if self.task != "gsm8k":
        #     pos_acc = acc
        acc = 1
        pos_acc = 1
        prev_pos_acc = 1
        grouped_outputs = [""] * encoder_input_ids.size(0)  # Placeholder for grouped outputs

        return acc, pos_acc, prev_pos_acc, grouped_outputs, total_flops


    def _measure_semi_acc(self, encoder_input_ids, gt_labels, device, mode="test"):
        """
        Part II: Iteratively generate outputs, project, re-encode, project, and calculate semi-accuracy.
        """
        # Create directory structure for saving embeddings
        embedding_dir = os.path.join("embeddings", mode)
        os.makedirs(embedding_dir, exist_ok=True)

        # Get next file index
        existing_files = [f for f in os.listdir(embedding_dir) if f.startswith('tensors') and f.endswith('.pkl')]
        next_index = len(existing_files) + 1

        decoder_inputs, decoder_att_mask, decoder_position_ids = self.pad_question(encoder_input_ids)
        decoder_inputs_embeds = self.decoder.transformer.wte(decoder_inputs)
        N = 10
        results = ["" for _ in range(encoder_input_ids.size(0))]
        semi_acc = 0
        semi_pos_acc = 0
        semi_prev_pos_acc = 0
        
        cleaned = [ seq[mask.bool()].tolist() for seq, mask in zip(decoder_inputs, decoder_att_mask) ]   
        context = self.tokenizer.batch_decode(cleaned, skip_special_tokens=True)

        # Initialize profiler
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        total_flops = 0

        with profile(activities=activities, record_shapes=True, profile_memory=True, with_flops=True) as prof:
            past_key_values = None
            
            
            for _ in range(N):
                with torch.no_grad():
                    # 1. DECODER STEP
                    with record_function("decoder_step"):
                       
                        if past_key_values is None:
                            decoder_out = self.decoder.transformer(
                                inputs_embeds=decoder_inputs_embeds,
                                attention_mask=decoder_att_mask,
                                position_ids=decoder_position_ids,
                                return_dict=True # Use return_dict
                            )
                        else:
                            # Subsequent iterations: only process the new token.
                            # We assume the new token is the last token in decoder_inputs_embeds.
                            new_token_embed = decoder_inputs_embeds[:, -1:, :]  # shape: (B, 1, H)
                            new_token_pos = decoder_position_ids[:, -1:]         # shape: (B, 1)
                            # (Optionally, you can pass only the new token's attention mask if your model supports it.)
                            new_token_att = decoder_att_mask[:, -1:]               # shape: (B, 1)

                            decoder_out = self.decoder.transformer(
                                inputs_embeds=new_token_embed,
                                attention_mask=new_token_att,
                                position_ids=new_token_pos,
                                past_key_values=past_key_values,  # Use the cached keys/values.
                                use_cache=True,
                            )
                        past_key_values = (
                            decoder_out.past_key_values if hasattr(decoder_out, "past_key_values") 
                            else decoder_out[1]
                        )
                        # Use last_hidden_state directly, it already includes ln_f
                        decoder_hidden = decoder_out.last_hidden_state
                        last_hidden = decoder_hidden[:, -1, :].unsqueeze(1) # Unprojected latent [B, 1, H]

                        # # Update attention mask and position IDs.
                        # decoder_att_mask = torch.cat(
                        #     [decoder_att_mask, torch.ones((decoder_att_mask.size(0), 1), device=device)],
                        #     dim=1,
                        # )
                        decoder_att_mask = torch.ones((decoder_att_mask.size(0), 1), device=device)
                        
                        new_position_ids = decoder_position_ids[:, -1] + 1
                        # decoder_position_ids = torch.cat(
                        #     [decoder_position_ids, new_position_ids.unsqueeze(1)], dim=1
                        # )
                        decoder_position_ids = new_position_ids.unsqueeze(1)

                    # 2. DECODER -> TRANSLATOR STEP
                    target_out_unprojected = last_hidden.reshape(-1, last_hidden.size(-1)) # Unprojected latent [B, H]

                    with record_function("projection_step"):
                        # Apply decoder-to-translator projection
                        target_out_projected = self.decoder_to_translator_proj(target_out_unprojected)

                    with torch.no_grad():
                        with record_function("translator_step"):
                            
                            past_key_values_translator = None
                            decoded_outputs_str = ""
                            with torch.no_grad():
                                inputs_embeds = target_out_projected.unsqueeze(1)
                                outputs = self.translator(
                                    inputs_embeds=inputs_embeds,
                                    use_cache=True,
                                )
                                past_key_values_translator = outputs.past_key_values if hasattr(outputs, "past_key_values") else outputs[1]

                                next_token_logits = outputs.logits[:, -1, :]
                                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                                
                                decoded_outputs_str += self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                                # print(f"Decoded outputs: {decoded_outputs}")
                                for i in range(128):
                                    outputs = self.translator(
                                        input_ids=next_token,
                                        past_key_values=past_key_values_translator,
                                        use_cache=True,
                                    )
                                    past_key_values_translator = outputs.past_key_values if hasattr(outputs, "past_key_values") else outputs[1]
                                    
                                    next_token_logits = outputs.logits[:, -1, :]
                                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                                    
                                    
                                    if next_token[0] == self.tokenizer.eos_token_id:
                                        break
                                    
                                    decoded_outputs_str += self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                                
                            decoded_outputs = [decoded_outputs_str]
                            
                            if "###" in decoded_outputs[0]:
                                # Stop generating if the model generates "###"
                                break
                            decoded_outputs = [x.strip() + "\n" for x in decoded_outputs]         
                            
                            # context = [x + y for x, y in zip(context, decoded_outputs)]
                    
                    # 3. ENCODER STEP (Re-encoding projected translated output)
                    enc_input_ids = self.tokenizer(
                        decoded_outputs, return_tensors="pt", padding=True, truncation=True
                    ).input_ids.to(device)
                    
                    enc_attention = self.tokenizer(
                        decoded_outputs, return_tensors="pt", padding=True, truncation=True
                    ).attention_mask.to(device)

                    with torch.no_grad(): # Encoder is frozen
                        with record_function("encoder_step"):
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
                    encoder_hidden_states_projected = self.encoder_to_decoder_proj(encoder_hidden_states_unprojected)

                    # # Append projected encoder state to decoder inputs for next step
                    # decoder_inputs_embeds = torch.cat(
                    #     [decoder_inputs_embeds, encoder_hidden_states_projected.unsqueeze(1)], # Use projected
                    #     dim=1
                    # )
                    
                    decoder_inputs_embeds = encoder_hidden_states_projected.unsqueeze(1)

        # Calculate total FLOPs from profiler
        total_flops = sum(event.flops for event in prof.key_averages() if event.flops > 0)
        print(f"Semi Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")

        # results = context
        # # Calculate semi-accuracy.
        
        # for a, b in zip(results, gt_labels):
        #     if self.task == "gsm8k":
        #         if compare_last_formula(a) == extract_final_answer(b, self.task):
        #             semi_pos_acc += 1
        #     if extract_final_answer(a, self.task) == extract_final_answer(b, self.task):
        #         semi_acc += 1
        #     else:
        #         if compare_last_formula(a) == compare_last_formula(b):
        #             semi_prev_pos_acc += 1

        # if self.task != "gsm8k":
        #     semi_pos_acc = semi_acc

        return semi_acc, semi_pos_acc, semi_prev_pos_acc, results, total_flops


    def _measure_gt_acc(
        self, encoder_input_ids, encoder_attention_mask, steps_input_ids, steps_attention_mask, steps_valid_mask, device
    ):
        """
        Part III: Compute GT accuracy by re-encoding valid steps and grouping predictions by stage.
        Uses projection layers and detaching.
        """
        with torch.no_grad(): # Everything should be no_grad in test mode
            valid_steps = steps_input_ids[steps_valid_mask]
            valid_attention = steps_attention_mask[steps_valid_mask]
            enc_outputs = self.encoder.transformer(
                input_ids=valid_steps,
                attention_mask=valid_attention,
                return_dict=True,
            )
            # last_hidden_state already has ln_f applied
            encoder_hidden_states = enc_outputs.last_hidden_state
            # No detach needed here as we are in no_grad context already

            # Project encoder hidden states
            projected_encoder_hidden = self.encoder_to_decoder_proj(encoder_hidden_states)
            # No dropout needed in eval mode

            # Build decoder inputs for valid steps using projected encoder states.
            all_latent_inputs, all_latent_targets, decoder_attention_mask, latent_ce_labels = self.build_decoder_inputs(
                encoder_input_ids, projected_encoder_hidden, valid_attention, steps_valid_mask
            )
            decoder_outputs = self.decoder.transformer(
                inputs_embeds=all_latent_inputs,
                return_dict=True # Ensure return_dict is True
            )
            # last_hidden_state already has ln_f applied
            decoder_hidden = decoder_outputs.last_hidden_state
            # No detach needed here as we are in no_grad context already

            target_mask = (all_latent_targets.abs().sum(dim=-1) != 0)
            B, L_dec = target_mask.shape

            # Assign stage indices for valid tokens.
            stage_indices = torch.zeros_like(all_latent_targets, dtype=torch.long)
            for b in range(B):
                counter = 0
                for i in range(L_dec):
                    if target_mask[b, i]:
                        stage_indices[b, i] = counter
                        counter += 1

            # Project the relevant decoder hidden states
            translator_prefix_unprojected = decoder_hidden[target_mask]
            translator_prefix_projected = self.decoder_to_translator_proj(translator_prefix_unprojected)
            translator_prefix = translator_prefix_projected.unsqueeze(1) # Add sequence dimension

            valid_stage_indices = stage_indices[target_mask]
            valid_steps_input_ids = steps_input_ids[steps_valid_mask]
            valid_steps_attention_mask = steps_attention_mask[steps_valid_mask]


            translator_embeds = self.translator.transformer.wte(valid_steps_input_ids)
            # Concatenate the *projected* decoder prefix
            translator_embeds_combined = torch.cat([translator_prefix, translator_embeds], dim=1)
            translator_attention_mask = torch.cat(
                [torch.ones((valid_steps_attention_mask.size(0), 1), device=device, dtype=torch.long), valid_steps_attention_mask],
                dim=1,
            )

            valid_gen_indices = (translator_attention_mask[:, 0] == 1).nonzero(as_tuple=True)[0]
            # Generate using only the projected prefix as input_embeds
            # Note: translator.generate might need adjustment if it expects token IDs
            # Assuming it works correctly with inputs_embeds directly for the first token
            generated_translations = self.translator.generate(
                inputs_embeds=translator_prefix[valid_gen_indices], # Pass projected prefix
                max_new_tokens=128, # Add a reasonable max length
                do_sample=False,
                temperature=0,
            )
            # The rest of the logic remains similar...
            decoded_translations = [
                self.tokenizer.decode(t, skip_special_tokens=True) for t in generated_translations
            ]
            decoded_targets = [
                self.tokenizer.decode(valid_steps_input_ids[i], skip_special_tokens=True)
                for i in valid_gen_indices
            ]
            valid_stage_numbers = valid_stage_indices[valid_gen_indices].tolist()

            stage_total = {}
            stage_correct = {}
            gt_non_ls = []
            gt_ls = []

            for stage, pred, target in zip(valid_stage_numbers, decoded_translations, decoded_targets):
                # Use your check_eq (or equivalent) to determine correctness.
                stage = stage[0]
                correct = check_eq(pred, target)
                
                if self.task == "gsm8k":
                    non_ls_cond = ">>" in target
                elif self.task == "csqa":
                    non_ls_cond = "###" not in target
                else:
                    raise NotImplementedError(f"Task {self.task} not implemented.")
                
                if non_ls_cond:
                    gt_non_ls.append(correct)
                    stage_total[stage] = stage_total.get(stage, 0) + 1
                    stage_correct[stage] = stage_correct.get(stage, 0) + int(correct)
                else:
                    gt_ls.append(correct)
                    
            # import pdb;
            # pdb.set_trace()
            
            if len(gt_ls) != len(encoder_input_ids):
                print("ERROR")
                print(len(gt_ls), len(encoder_input_ids))

            gt_ls_acc = gt_ls.count(True)
            gt_non_ls_acc = gt_non_ls.count(True)
            stage_acc_dict = {}
            for stage in sorted(stage_total.keys()):
                stage_acc_dict[f"stage{stage}_acc"] = stage_correct[stage]
                stage_acc_dict[f"stage{stage}_total"] = stage_total[stage]

            total_steps = len(decoded_targets)

        return stage_acc_dict, gt_ls_acc, gt_non_ls_acc, total_steps


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
                "Pos_Acc": [
                    compare_last_formula(predictions[i]) == compare_last_formula(gt_labels[i])
                    for i in sample_indices
                ],
                "Acc": [
                    extract_final_answer(predictions[i], self.task)
                    == extract_final_answer(gt_labels[i], self.task)
                    for i in sample_indices
                ],
            }
            df = pd.DataFrame(sampled_data)
            table = wandb.Table(dataframe=df)
            wandb.log({f"Saved Predictions (Step {step})": table, "step": step})

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
        torch.save(self.encoder_to_decoder_proj.state_dict(), os.path.join(save_dir, "encoder_to_decoder_proj.pt"))
        torch.save(self.decoder_to_translator_proj.state_dict(), os.path.join(save_dir, "decoder_to_translator_proj.pt"))
        
        # Save model state dictionaries - save all components regardless of sharing
        torch.save(self.decoder.state_dict(), os.path.join(save_dir, "decoder.pt"))
        torch.save(self.encoder.state_dict(), os.path.join(save_dir, "encoder.pt"))
        torch.save(self.translator.state_dict(), os.path.join(save_dir, "translator.pt"))
        
        # Save configuration
        config = {
            "task": self.task,
            "share_param": self.config["share_param"],
            "use_cont": self.use_cont,
            "use_dist": self.use_dist,
            "use_mse": self.use_mse,
            "dropout_rate": self.dropout_rate,
            "encoder_path": self.encoder.config._name_or_path,
            "decoder_path": self.decoder.config._name_or_path,
            "translator_path": self.translator.config._name_or_path if not self.config["share_param"] else None,
            "tokenizer_name": self.tokenizer.name_or_path
        }
        
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f)
            
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
        
        # Also save the complete model architectures - save all regardless of sharing
        self.encoder.save_pretrained(os.path.join(save_dir, "encoder_model"))
        self.decoder.save_pretrained(os.path.join(save_dir, "decoder_model"))
        self.translator.save_pretrained(os.path.join(save_dir, "translator_model"))
        
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
        if use_dist is not None:
            config["use_dist"] = use_dist
        if use_mse is not None:
            config["use_mse"] = use_mse
        
        # Load tokenizer if not provided
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_dir, "tokenizer"))
        
        # Create model instance
        model = cls(
            tokenizer=tokenizer,
            encoder_path=config["encoder_path"],
            decoder_path=config["decoder_path"],
            translator_path=config["encoder_path"] if config["share_param"] else config["translator_path"],
            task=config["task"],
            freeze=freeze,
            share_param=config["share_param"],
            use_cont=config["use_cont"],
            use_dist=config["use_dist"],
            use_mse=config["use_mse"]
        )
        
        # Update dropout rate if different
        model.dropout_rate = config["dropout_rate"]
        
        print("Load dir", load_dir)

        # Load saved state dictionaries
        model.encoder_to_decoder_proj.load_state_dict(
            torch.load(os.path.join(load_dir, "encoder_to_decoder_proj.pt"))
        )
        model.decoder_to_translator_proj.load_state_dict(
            torch.load(os.path.join(load_dir, "decoder_to_translator_proj.pt"))
        )
        model.decoder.load_state_dict(
            torch.load(os.path.join(load_dir, "decoder.pt"))
        )
        model.encoder.load_state_dict(
            torch.load(os.path.join(load_dir, "encoder.pt"))
        )
        
        # Load translator state if it's not shared with the encoder
        if not config["share_param"]:
            model.translator.load_state_dict(
                torch.load(os.path.join(load_dir, "translator.pt"))
            )
        
        print(f"Model loaded from {load_dir}")
        return model
