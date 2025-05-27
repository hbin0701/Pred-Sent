import torch
from tqdm import tqdm
import wandb

def test_model(model, dataloader, device, accelerator, step, mode="train", MAX_SAMPLES=5000):
    model.eval()
    total_ae_acc = 0
    total_steps = 0
    curr_samples = 0
    
    for batch in tqdm(dataloader, desc="Testing", leave=False):
        encoder_input_ids = batch["encoder_input_ids"].to(device)
        encoder_attention_mask = batch["encoder_attention_mask"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
    
        curr_samples += encoder_input_ids.size(0)
        if curr_samples > MAX_SAMPLES:
            break
        
        with torch.no_grad():
            metrics = model.test(
                encoder_input_ids,
                encoder_attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
            )
        total_ae_acc += metrics["ae_acc"]
        total_steps += metrics["total_steps"]

    overall_ae_acc = total_ae_acc / total_steps * 100
    wandb.log({f"{mode}_acc/ae": overall_ae_acc})
    return {f"{mode}_acc/ae": overall_ae_acc}
