import torch

def dual_collate_fn(batch, tokenizer):
    def pad_to_max_length(tensor_list, pad_token_id):
        max_length = max(t.size(0) for t in tensor_list)
        return torch.stack([
            torch.cat([t, torch.full((max_length - t.size(0),), pad_token_id, dtype=t.dtype)])
            for t in tensor_list
        ])
    encoder_input_ids = [torch.LongTensor(example["encoder_input_ids"]) for example in batch]
    encoder_attention_masks = [torch.LongTensor(example["encoder_attention_mask"]) for example in batch]
    decoder_input_ids = [torch.LongTensor(example["decoder_input_ids"]) for example in batch]
    decoder_attention_masks = [torch.LongTensor(example["decoder_attention_mask"]) for example in batch]
    pad_token_id = tokenizer.pad_token_id
    return {
        "encoder_input_ids": pad_to_max_length(encoder_input_ids, pad_token_id),
        "encoder_attention_mask": pad_to_max_length(encoder_attention_masks, 0),
        "decoder_input_ids": pad_to_max_length(decoder_input_ids, pad_token_id),
        "decoder_attention_mask": pad_to_max_length(decoder_attention_masks, 0),
    }
