import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, tokenizer):
    # Pad encoder sequences.
    encoder_input_ids = pad_sequence(
        [torch.LongTensor(item["encoder_input_ids"]) for item in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    encoder_attention_mask = pad_sequence(
        [torch.LongTensor(item["encoder_attention_mask"]) for item in batch],
        batch_first=True,
        padding_value=0
    )

    # Process the steps.
    max_num_steps = max(len(item["steps_enc"]) for item in batch)
    max_step_len = 0
    for item in batch:
        for step in item["steps_enc"]:
            max_step_len = max(max_step_len, len(step["input_ids"]))
    
    batch_steps_input_ids = []
    batch_steps_attention_mask = []
    steps_valid_mask_list = []
    
    for item in batch:
        steps_input_ids = []
        steps_attention_mask = []
        for step in item["steps_enc"]:
            step_ids = torch.LongTensor(step["input_ids"])
            step_mask = torch.LongTensor(step["attention_mask"])
            pad_len = max_step_len - step_ids.size(0)
            if pad_len > 0:
                step_ids = torch.cat([
                    step_ids, 
                    torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
                ])
                step_mask = torch.cat([
                    step_mask, 
                    torch.zeros(pad_len, dtype=torch.long)
                ])
            steps_input_ids.append(step_ids)
            steps_attention_mask.append(step_mask)
        
        num_real_steps = len(item["steps_enc"])
        valid_mask = torch.tensor([1] * num_real_steps + [0] * (max_num_steps - num_real_steps), dtype=torch.bool)
        steps_valid_mask_list.append(valid_mask)
        
        # Add dummy steps if needed.
        for _ in range(max_num_steps - num_real_steps):
            dummy_ids = torch.full((max_step_len,), tokenizer.pad_token_id, dtype=torch.long)
            dummy_mask = torch.zeros(max_step_len, dtype=torch.long)
            steps_input_ids.append(dummy_ids)
            steps_attention_mask.append(dummy_mask)
        
        steps_input_ids = torch.stack(steps_input_ids)
        steps_attention_mask = torch.stack(steps_attention_mask)
        batch_steps_input_ids.append(steps_input_ids)
        batch_steps_attention_mask.append(steps_attention_mask)
    
    batch_steps_input_ids = torch.stack(batch_steps_input_ids)
    batch_steps_attention_mask = torch.stack(batch_steps_attention_mask)
    steps_valid_mask = torch.stack(steps_valid_mask_list)

    return {
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "steps_input_ids": batch_steps_input_ids,
        "steps_attention_mask": batch_steps_attention_mask,
        "steps_valid_mask": steps_valid_mask,
    }
