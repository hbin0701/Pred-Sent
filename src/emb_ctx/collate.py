import torch

def pad_to_max_length(tensor_list, pad_token_id):
    """
    Pad a list of tensors to the maximum length in the list.
    
    Args:
        tensor_list: List of tensors to pad
        pad_token_id: Token ID to use for padding
        
    Returns:
        Stacked tensor with padding
    """
    max_length = max(t.size(0) for t in tensor_list)
    return torch.stack([
        torch.cat([t, torch.full((max_length - t.size(0),), pad_token_id, dtype=t.dtype)])
        for t in tensor_list
    ])


def problem_collate_fn(batch, tokenizer):
    """Collate function for problem dataset batches"""
    encoder_input_ids = [torch.LongTensor(example["encoder_input_ids"]) for example in batch]
    encoder_attention_masks = [torch.LongTensor(example["encoder_attention_mask"]) for example in batch]
    
    def pad_to_max_length(tensor_list, pad_token_id):
        max_length = max(t.size(0) for t in tensor_list)
        return torch.stack([
            torch.cat([t, torch.full((max_length - t.size(0),), pad_token_id, dtype=t.dtype)])
            for t in tensor_list
        ])
    
    pad_token_id = tokenizer.pad_token_id
    answers = [example["answer"] for example in batch]
    questions = [example["question"] for example in batch]
    
    return {
        "encoder_input_ids": pad_to_max_length(encoder_input_ids, pad_token_id),
        "encoder_attention_mask": pad_to_max_length(encoder_attention_masks, 0),
        "answer": answers,
        "question": questions
    }

def process_dual_batch(batch, tokenizer):
    """
    Process a batch for dual model training (shared between contrastive and regular dual approach).
    
    Args:
        batch: Batch of examples
        tokenizer: Tokenizer object for pad token ID
        
    Returns:
        Dictionary with padded tensors for both models
    """
    # For model 1

    if type(batch) == list:    
        encoder_input_ids1 = [torch.LongTensor(example["encoder_input_ids1"]) for example in batch]
        encoder_attention_masks1 = [torch.LongTensor(example["encoder_attention_mask1"]) for example in batch]
        decoder_input_ids1 = [torch.LongTensor(example["decoder_input_ids1"]) for example in batch]
        decoder_attention_masks1 = [torch.LongTensor(example["decoder_attention_mask1"]) for example in batch]
            
        # For model 2
        encoder_input_ids2 = [torch.LongTensor(example["encoder_input_ids2"]) for example in batch]
        encoder_attention_masks2 = [torch.LongTensor(example["encoder_attention_mask2"]) for example in batch]
        decoder_input_ids2 = [torch.LongTensor(example["decoder_input_ids2"]) for example in batch]
        decoder_attention_masks2 = [torch.LongTensor(example["decoder_attention_mask2"]) for example in batch]
        
        pad_token_id = tokenizer.pad_token_id
        

        return {
            "encoder_input_ids1": pad_to_max_length(encoder_input_ids1, pad_token_id),
            "encoder_attention_mask1": pad_to_max_length(encoder_attention_masks1, 0),
            "decoder_input_ids1": pad_to_max_length(decoder_input_ids1, pad_token_id),
            "decoder_attention_mask1": pad_to_max_length(decoder_attention_masks1, 0),
            
            # Model 2
            "encoder_input_ids2": pad_to_max_length(encoder_input_ids2, pad_token_id),
            "encoder_attention_mask2": pad_to_max_length(encoder_attention_masks2, 0),
            "decoder_input_ids2": pad_to_max_length(decoder_input_ids2, pad_token_id),
            "decoder_attention_mask2": pad_to_max_length(decoder_attention_masks2, 0),
            }
    
    elif type(batch) == dict:
        return batch


def contrastive_collate_fn(batch, tokenizer):
    """
    Collate function for contrastive learning approach.
    
    Args:
        batch: Batch of examples
        tokenizer: Tokenizer object for pad token ID
        
    Returns:
        Processed batch with padded tensors
    """
    return process_dual_batch(batch, tokenizer)

def dual_collate_fn(batch, tokenizer):
    """
    Collate function for standard dual model approach.
    
    Args:
        batch: Batch of examples
        tokenizer: Tokenizer object for pad token ID
        
    Returns:
        Processed batch with padded tensors
    """
    return process_dual_batch(batch, tokenizer) 