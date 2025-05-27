import os
import glob
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
import wandb

from utils import clear_gpu_cache_if_needed
from evaluate import test_model
from models import calculate_contrastive_loss

from accelerate import Accelerator, DataLoaderConfiguration
from collate import contrastive_collate_fn, problem_collate_fn

def setup_training_components(args, models, optimizers, lr_schedulers, accelerators):
    """Extract and setup training components from provided dictionaries."""
    # Extract models, optimizers, schedulers and accelerators
    model1 = models["model1"]  # Restoration model
    model2 = models["model2"]  # Prediction model
    optimizer1 = optimizers["optimizer1"]  # Optimizer for restoration
    optimizer2 = optimizers["optimizer2"]  # Optimizer for prediction
    lr_scheduler1 = lr_schedulers["lr_scheduler1"]
    lr_scheduler2 = lr_schedulers["lr_scheduler2"]
    accelerator1 = accelerators["accelerator1"]
    accelerator2 = accelerators["accelerator2"]
    
    # Extract contrastive learning parameters
    contrastive_weight = args.contrastive_weight if hasattr(args, 'contrastive_weight') else 0.1
    temperature = args.temperature if hasattr(args, 'temperature') else 0.5
    use_contrastive = hasattr(args, 'use_cont') and args.use_cont
    grad_accum_steps = args.grad_accum_steps if hasattr(args, 'grad_accum_steps') else 1
    
    print(f"Contrastive learning is {'enabled' if use_contrastive else 'disabled'}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    
    return {
        "model1": model1,
        "model2": model2,
        "optimizer1": optimizer1,
        "optimizer2": optimizer2,
        "lr_scheduler1": lr_scheduler1,
        "lr_scheduler2": lr_scheduler2,
        "accelerator1": accelerator1,
        "accelerator2": accelerator2,
        "contrastive_weight": contrastive_weight,
        "temperature": temperature,
        "use_contrastive": use_contrastive,
        "grad_accum_steps": grad_accum_steps
    }


def train_batch(batch, components, device, global_step):
    """Process a single training batch."""
    model1 = components["model1"]
    model2 = components["model2"]
    optimizer2 = components["optimizer2"]
    lr_scheduler2 = components["lr_scheduler2"]
    accelerator2 = components["accelerator2"]
    use_contrastive = components["use_contrastive"]
    contrastive_weight = components["contrastive_weight"]
    temperature = components["temperature"]
    grad_accum_steps = components.get("grad_accum_steps", 1)
    should_update = (global_step + 1) % grad_accum_steps == 0 or grad_accum_steps == 1
    
    # Process restoration task with model1 if using contrastive learning
    model1.eval()
    
    if use_contrastive:
        with torch.no_grad():
            encoder_input_ids1 = batch["encoder_input_ids1"].to(device)
            encoder_attention_mask1 = batch["encoder_attention_mask1"].to(device)
            decoder_input_ids1 = batch["decoder_input_ids1"].to(device)
            decoder_attention_mask1 = batch["decoder_attention_mask1"].to(device)
            
            restoration_loss, _, restoration_loss_dict = model1(
                encoder_input_ids1, 
                encoder_attention_mask1, 
                decoder_input_ids1, 
                decoder_attention_mask1
            )
    else:
        # Set placeholder values when not using model1
        restoration_loss = torch.tensor(0.0, device=device)
        restoration_loss_dict = {"ce_loss": torch.tensor(0.0, device=device)}
        
    # Process prediction task with model2
    encoder_input_ids2 = batch["encoder_input_ids2"].to(device)
    encoder_attention_mask2 = batch["encoder_attention_mask2"].to(device)
    decoder_input_ids2 = batch["decoder_input_ids2"].to(device)
    decoder_attention_mask2 = batch["decoder_attention_mask2"].to(device)
    
    prediction_loss, _, prediction_loss_dict = model2(
        encoder_input_ids2, 
        encoder_attention_mask2, 
        decoder_input_ids2, 
        decoder_attention_mask2
    )
    
    # Calculate contrastive loss between model representations if enabled
    if use_contrastive:
        weighted_contrastive_loss, raw_contrastive_loss = calculate_contrastive_loss(
            model1.last_encoder_rep, 
            model2.last_encoder_rep, 
            weight=contrastive_weight, 
            temperature=temperature
        )
        prediction_loss_with_contrastive = prediction_loss + weighted_contrastive_loss
    else:
        weighted_contrastive_loss = torch.tensor(0.0, device=device)
        raw_contrastive_loss = torch.tensor(0.0, device=device)
        prediction_loss_with_contrastive = prediction_loss

    # Backward pass and optimization for prediction model
    # Scale the loss by grad_accum_steps for gradient accumulation
    scaled_loss = prediction_loss_with_contrastive / grad_accum_steps
    accelerator2.backward(scaled_loss)
    
    # Only update weights after accumulating enough gradients
    if should_update:
        optimizer2.step()
        lr_scheduler2.step()
        optimizer2.zero_grad()
    
    # Combined loss for tracking
    combined_loss = prediction_loss + weighted_contrastive_loss
    if use_contrastive:
        combined_loss += restoration_loss
    
    # Create metrics log dictionary
    log_dict = {
        "train_combined_loss": combined_loss.item(),
        "train_prediction_loss": prediction_loss.item(),
        "train_prediction_ce_loss": prediction_loss_dict["ce_loss"].item(),
        "train_contrastive_loss": raw_contrastive_loss.item(),
        "train_weighted_contrastive_loss": weighted_contrastive_loss.item(),
        "learning_rate2": lr_scheduler2.get_last_lr()[0],
        "global_step": global_step,
    }
    
    # Only include restoration metrics if using contrastive learning
    if use_contrastive:
        log_dict.update({
            "train_restoration_loss": restoration_loss.item(),
            "train_restoration_ce_loss": restoration_loss_dict["ce_loss"].item(),
            "learning_rate1": components["lr_scheduler1"].get_last_lr()[0],
        })
    
    return {
        "combined_loss": combined_loss.item(),
        "restoration_loss": restoration_loss.item(),
        "prediction_loss": prediction_loss.item(),
        "contrastive_loss": raw_contrastive_loss.item(),
        "log_dict": log_dict,
        "should_update": should_update
    }


def validate_model(valid_dataloader, components, device):
    """Run validation on the model."""
    model1 = components["model1"]
    model2 = components["model2"]
    use_contrastive = components["use_contrastive"]
    contrastive_weight = components["contrastive_weight"]
    temperature = components["temperature"]
    
    total_eval_loss = 0.0
    total_eval_restoration_loss = 0.0
    total_eval_prediction_loss = 0.0
    total_eval_contrastive_loss = 0.0
    eval_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc="Validating"):
            # Evaluate restoration task with model1 only if using contrastive learning
            if use_contrastive:
                encoder_input_ids1 = batch["encoder_input_ids1"].to(device)
                encoder_attention_mask1 = batch["encoder_attention_mask1"].to(device)
                decoder_input_ids1 = batch["decoder_input_ids1"].to(device)
                decoder_attention_mask1 = batch["decoder_attention_mask1"].to(device)
                
                restoration_loss, _, restoration_loss_dict = model1(
                    encoder_input_ids1, 
                    encoder_attention_mask1, 
                    decoder_input_ids1, 
                    decoder_attention_mask1
                )
            else:
                # Set placeholder values when not using model1
                restoration_loss = torch.tensor(0.0, device=device)
                restoration_loss_dict = {"ce_loss": torch.tensor(0.0, device=device)}
            
            # Evaluate prediction task with model2
            encoder_input_ids2 = batch["encoder_input_ids2"].to(device)
            encoder_attention_mask2 = batch["encoder_attention_mask2"].to(device)
            decoder_input_ids2 = batch["decoder_input_ids2"].to(device)
            decoder_attention_mask2 = batch["decoder_attention_mask2"].to(device)
            
            prediction_loss, _, prediction_loss_dict = model2(
                encoder_input_ids2, 
                encoder_attention_mask2, 
                decoder_input_ids2, 
                decoder_attention_mask2
            )
            
            # Calculate contrastive loss if using contrastive learning
            if use_contrastive:
                weighted_contrastive_loss, raw_contrastive_loss = calculate_contrastive_loss(
                    model1.last_encoder_rep, 
                    model2.last_encoder_rep, 
                    weight=contrastive_weight, 
                    temperature=temperature
                )
                total_eval_contrastive_loss += raw_contrastive_loss.item()
            else:
                weighted_contrastive_loss = torch.tensor(0.0, device=device)
                raw_contrastive_loss = torch.tensor(0.0, device=device)
            
            # Combined loss for tracking
            combined_loss = prediction_loss + weighted_contrastive_loss
            if use_contrastive:
                combined_loss += restoration_loss
            
            # Track validation losses
            total_eval_loss += combined_loss.item()
            total_eval_restoration_loss += restoration_loss.item()
            total_eval_prediction_loss += prediction_loss.item()
            eval_steps += 1

    # Calculate average validation losses
    avg_eval_loss = total_eval_loss / eval_steps if eval_steps > 0 else 0.0
    avg_eval_restoration_loss = total_eval_restoration_loss / eval_steps if eval_steps > 0 else 0.0
    avg_eval_prediction_loss = total_eval_prediction_loss / eval_steps if eval_steps > 0 else 0.0
    avg_eval_contrastive_loss = total_eval_contrastive_loss / eval_steps if eval_steps > 0 else 0.0
    
    return {
        "avg_eval_loss": avg_eval_loss,
        "avg_eval_restoration_loss": avg_eval_restoration_loss,
        "avg_eval_prediction_loss": avg_eval_prediction_loss,
        "avg_eval_contrastive_loss": avg_eval_contrastive_loss,
        "eval_steps": eval_steps
    }


def save_checkpoints(args, epoch, components):
    """Save model checkpoints."""
    accelerator1 = components["accelerator1"]
    accelerator2 = components["accelerator2"]
    model1 = components["model1"]
    model2 = components["model2"]
    use_contrastive = components["use_contrastive"]
    
    # Create checkpoint directory
    checkpoint_dir = f"{args.save_dir}/{epoch+1}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save encoder/decoder for both models
    if use_contrastive:
        # Unwrap model1
        unwrapped_model1 = accelerator1.unwrap_model(model1)
        unwrapped_model1.encoder.save_pretrained(
            f"{checkpoint_dir}/encoder1", 
            save_function=accelerator1.save
        )
        unwrapped_model1.decoder.save_pretrained(
            f"{checkpoint_dir}/decoder1", 
            save_function=accelerator1.save
        )
    
    # Always save model2
    unwrapped_model2 = accelerator2.unwrap_model(model2)
    unwrapped_model2.encoder.save_pretrained(
        f"{checkpoint_dir}/encoder2", 
        save_function=accelerator2.save
    )
    unwrapped_model2.decoder.save_pretrained(
        f"{checkpoint_dir}/decoder2", 
        save_function=accelerator2.save
    )


def run_detailed_evaluation(args, epoch, global_step, components, dataloaders, device, num_generations, eval_accelerator):
    """Run detailed evaluation on train, validation, and test datasets."""
    model1 = components["model1"]
    model2 = components["model2"]
    accelerator1 = components["accelerator1"]
    accelerator2 = components["accelerator2"]
    use_contrastive = components["use_contrastive"]
    
    train_gen_dataloader = dataloaders["train_gen"]
    valid_dataloader = dataloaders["valid"]
    test_dataloader = dataloaders["test"]
    train_problem_dataloader = dataloaders.get("train_problem")
    valid_problem_dataloader = dataloaders.get("valid_problem")
    test_problem_dataloader = dataloaders.get("test_problem")
    
    
    def make_new_dataloader(dataset, tokenizer, args, eval_accelerator, is_problem=False):

        dl = DataLoader(
            dataset,
            batch_size=64,
            collate_fn=lambda x: contrastive_collate_fn(x, tokenizer) if not is_problem else problem_collate_fn(x, tokenizer),
            drop_last=False,  # Don't drop uneven batches
        )
        
        return eval_accelerator.prepare(dl)

    new_test_dataloader = make_new_dataloader(test_dataloader.dataset, model1.tokenizer, args, eval_accelerator)
    new_test_problem_dataloader = make_new_dataloader(test_problem_dataloader.dataset, model1.tokenizer, args, eval_accelerator, is_problem=True)
    
    MAX_SAMPLES = len(test_problem_dataloader.dataset)
    
    # Re-create dataloaders with consistent settings
    train_gen_dataloader = DataLoader(
        train_gen_dataloader.dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=train_gen_dataloader.collate_fn
    )                    
    
    valid_dataloader = DataLoader(
        valid_dataloader.dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=valid_dataloader.collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataloader.dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=test_dataloader.collate_fn
    )
    
    print("Evaluating train dataset...")
    
    # if use_contrastive:
    #     print("Restoration metrics:")
    #     metrics_restoration = test_model(
    #         model1, 
    #         train_gen_dataloader, 
    #         device, 
    #         accelerator1, 
    #         global_step, 
    #         mode="train", 
    #         problem_dataloader=None,
    #         num_generations=num_generations,
    #         MAX_PROB_SAMPLES=MAX_SAMPLES
    #     )
    #     print(metrics_restoration)

    # print("Prediction metrics:")
    # metrics_prediction = test_model(
    #     model2, 
    #     train_gen_dataloader, 
    #     device, 
    #     accelerator2, 
    #     global_step, 
    #     mode="train", 
    #     problem_dataloader=train_problem_dataloader,
    #     num_generations=num_generations,
    #     MAX_PROB_SAMPLES=MAX_SAMPLES
    # )
    # print(metrics_prediction)
    
    # print("Evaluating validation dataset...")
    
    # # Evaluate restoration task with model1 only if using contrastive learning
    # if use_contrastive:
    #     print("Restoration metrics:")
    #     metrics_restoration = test_model(
    #         model1, 
    #         valid_dataloader, 
    #         device, 
    #         accelerator1, 
    #         global_step, 
    #         mode="eval", 
    #         problem_dataloader=None,
    #         num_generations=num_generations,
    #         MAX_PROB_SAMPLES=MAX_SAMPLES
    #     )
    #     print(metrics_restoration)
    
    # # Always evaluate prediction task with model2
    # print("Prediction metrics:")
    # metrics_prediction = test_model(
    #     model2, 
    #     valid_dataloader, 
    #     device, 
    #     accelerator2, 
    #     global_step, 
    #     mode="eval", 
    #     problem_dataloader=valid_problem_dataloader,
    #     num_generations=num_generations,
    #     MAX_PROB_SAMPLES=MAX_SAMPLES
    # )
    # print(metrics_prediction)
    
    print("Evaluating test dataset...")
    
    # Evaluate restoration task with model1 only if using contrastive learning
    # if use_contrastive:
    #     print("Restoration metrics:")
    #     metrics_restoration = test_model(
    #         model1, 
    #         test_dataloader, 
    #         device, 
    #         accelerator1, 
    #         global_step, 
    #         mode="test", 
    #         problem_dataloader=None, 
    #         num_generations=num_generations,
    #         MAX_PROB_SAMPLES=MAX_SAMPLES
    #     )
    #     print(metrics_restoration)
    
    # Always evaluate prediction task with model2
    print("Prediction metrics:")
    metrics_prediction = test_model(
        model2, 
        new_test_dataloader, 
        device, 
        eval_accelerator, 
        global_step, 
        mode="test", 
        problem_dataloader=new_test_problem_dataloader, 
        num_generations=num_generations,
        MAX_PROB_SAMPLES=MAX_SAMPLES
    )

    if eval_accelerator.is_main_process:
        print(metrics_prediction)
    
    # Free up CUDA memory after evaluation
    torch.cuda.empty_cache()


def print_epoch_summary(epoch, args, train_stats, eval_stats, components):
    """Print a summary of the epoch's results."""
    use_contrastive = components["use_contrastive"]
    
    print_str = f"Epoch {epoch+1}/{args.num_epochs} - " 
    print_str += f"Train Loss: {train_stats['avg_train_loss']:.4f} "
    
    if use_contrastive:
        print_str += f"(Restore: {train_stats['avg_train_restoration_loss']:.4f}, "
        print_str += f"Predict: {train_stats['avg_train_prediction_loss']:.4f}, "
        print_str += f"Contrastive: {train_stats['avg_train_contrastive_loss']:.4f}), "
    else:
        print_str += f"(Predict: {train_stats['avg_train_prediction_loss']:.4f}), "
        
    print_str += f"Eval Loss: {eval_stats['avg_eval_loss']:.4f} "
    
    if use_contrastive:
        print_str += f"(Restore: {eval_stats['avg_eval_restoration_loss']:.4f}, "
        print_str += f"Predict: {eval_stats['avg_eval_prediction_loss']:.4f}, "
        print_str += f"Contrastive: {eval_stats['avg_eval_contrastive_loss']:.4f})"
    else:
        print_str += f"(Predict: {eval_stats['avg_eval_prediction_loss']:.4f})"
        
    print(print_str)


def train_contrastive(args, accelerators, models, optimizers, device, lr_schedulers,
                 train_dataloader, train_gen_dataloader, valid_dataloader, test_dataloader,
                 train_problem_dataloader=None, valid_problem_dataloader=None, test_problem_dataloader=None, 
                 num_generations=10):
    """
    Training function for contrastive learning with two separate models:
    - model1: dedicated to restoration task
    - model2: dedicated to prediction task
    
    Uses separate optimizers for each model, with contrastive loss between their representations.

    Args:
        args: Command line arguments
        accelerators: Dictionary containing accelerator1 and accelerator2
        models: Dictionary containing model1 (restoration) and model2 (prediction)
        optimizers: Dictionary containing optimizer1 (restoration) and optimizer2 (prediction)
        device: Device to train on
        lr_schedulers: Dictionary containing lr_scheduler1 and lr_scheduler2
        train_dataloader: DataLoader for training data
        train_gen_dataloader: DataLoader for evaluation on training data
        valid_dataloader: DataLoader for validation data
        test_dataloader: DataLoader for test data
        train_problem_dataloader: DataLoader for problem evaluation on training data
        valid_problem_dataloader: DataLoader for problem evaluation on validation data
        test_problem_dataloader: DataLoader for problem evaluation on test data
        num_generations: Number of steps to generate for problem evaluation
    """
    global_step = 0
    
    # Setup training components
    components = setup_training_components(args, models, optimizers, lr_schedulers, accelerators)

    dataloader_config = DataLoaderConfiguration(even_batches=False)
    eval_accelerator = Accelerator(dataloader_config=dataloader_config)  

    # Recalculate total_training_steps with gradient accumulation
    grad_accum_steps = components.get("grad_accum_steps", 1)
    num_updates_per_epoch = len(train_dataloader) // grad_accum_steps
    total_training_steps = args.num_epochs * num_updates_per_epoch
    
    # Update learning rate schedulers with new total steps
    optimizers["optimizer1"].param_groups[0]["lr"] = args.lr
    optimizers["optimizer2"].param_groups[0]["lr"] = args.lr
    lr_schedulers["lr_scheduler1"] = get_scheduler(
        name="linear",
        optimizer=optimizers["optimizer1"],
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=total_training_steps,
    )
    lr_schedulers["lr_scheduler2"] = get_scheduler(
        name="linear",
        optimizer=optimizers["optimizer2"],
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=total_training_steps,
    )
    components["lr_scheduler1"] = lr_schedulers["lr_scheduler1"]
    components["lr_scheduler2"] = lr_schedulers["lr_scheduler2"]
    
    # Create a dictionary of dataloaders for evaluation
    dataloaders = {
        "train": train_dataloader,
        "train_gen": train_gen_dataloader,
        "valid": valid_dataloader,
        "test": test_dataloader,
        "train_problem": train_problem_dataloader,
        "valid_problem": valid_problem_dataloader,
        "test_problem": test_problem_dataloader
    }

    
    for epoch in range(args.num_epochs):
        # Training phase
        
        components["model2"].train()
        
        # Initialize tracking variables
        total_train_loss = 0.0
        total_train_restoration_loss = 0.0
        total_train_prediction_loss = 0.0
        total_train_contrastive_loss = 0.0
        train_steps = 0

        for idx, batch in tqdm(enumerate(train_dataloader), 
                               desc=f"Epoch {epoch+1}/{args.num_epochs}", 
                               total=len(train_dataloader)):
            
            # Skip training if there's a break statement here
            # Process batch
            batch_results = train_batch(batch, components, device, global_step)
            
            # Track losses
            total_train_loss += batch_results["combined_loss"]
            total_train_restoration_loss += batch_results["restoration_loss"]
            total_train_prediction_loss += batch_results["prediction_loss"]
            total_train_contrastive_loss += batch_results["contrastive_loss"]
            train_steps += 1
            global_step += 1

            # Log metrics
            if components["accelerator2"].is_main_process:
                wandb.log(batch_results["log_dict"])
                
            # Clear GPU cache periodically to avoid OOM
            if (idx + 1) % 10 == 0:
                clear_gpu_cache_if_needed(device, threshold=0.7)

        # Calculate average losses for the epoch
        avg_train_loss = total_train_loss / train_steps if train_steps > 0 else 0.0
        avg_train_restoration_loss = total_train_restoration_loss / train_steps if train_steps > 0 else 0.0
        avg_train_prediction_loss = total_train_prediction_loss / train_steps if train_steps > 0 else 0.0
        avg_train_contrastive_loss = total_train_contrastive_loss / train_steps if train_steps > 0 else 0.0
        
        train_stats = {
            "avg_train_loss": avg_train_loss,
            "avg_train_restoration_loss": avg_train_restoration_loss,
            "avg_train_prediction_loss": avg_train_prediction_loss,
            "avg_train_contrastive_loss": avg_train_contrastive_loss
        }

        # Validation phase
        if components["use_contrastive"]:
            components["model1"].eval()
        components["model2"].eval()

        # Evaluate restoration task on training data if using contrastive learning
        # if components["accelerator1"].is_main_process and components["use_contrastive"]:
        #     # Define dataloader again
        #     train_gen_dataloader = DataLoader(
        #         train_gen_dataloader.dataset,
        #         batch_size=args.per_device_batch_size,
        #         shuffle=False,
        #         num_workers=16,
        #         collate_fn=train_gen_dataloader.collate_fn
        #     )                    
                
            # Evaluate restoration task with model1
            # print("Restoration metrics:")
            # metrics_restoration = test_model(
            #     components["model1"], 
            #     train_gen_dataloader, 
            #     device, 
            #     components["accelerator1"], 
            #     global_step, 
            #     mode="train", 
            #     problem_dataloader=None,
            #     num_generations=num_generations,
            #     MAX_PROB_SAMPLES=len(test_problem_dataloader.dataset)
            # )
            # print(metrics_restoration)
        
        # components["accelerator1"].wait_for_everyone()
        
        # Run validation on accelerator2
        if components["accelerator2"].is_main_process:
            # Run validation
            eval_stats = validate_model(valid_dataloader, components, device)
            
            # Print epoch summary
            print_epoch_summary(epoch, args, train_stats, eval_stats, components)
            
            # Log validation metrics
            log_dict = {
                "eval_combined_loss": eval_stats["avg_eval_loss"],
                "eval_prediction_loss": eval_stats["avg_eval_prediction_loss"],
                "eval_contrastive_loss": eval_stats["avg_eval_contrastive_loss"],
                "epoch": epoch + 1,
            }
            
            # Only include restoration metrics if using contrastive learning
            if components["use_contrastive"]:
                log_dict.update({
                    "eval_restoration_loss": eval_stats["avg_eval_restoration_loss"],
                })
                
            wandb.log(log_dict)
        
            # Save model checkpoints and run detailed evaluation
            if (epoch + 1) % 1 == 0 or epoch == args.num_epochs - 1:
                # Save checkpoints
                save_checkpoints(args, epoch, components)
                

        # Run detailed evaluation
        run_detailed_evaluation(
            args, 
            epoch, 
            global_step, 
            components, 
            dataloaders, 
            device, 
            num_generations,
            eval_accelerator
        )

        # Make sure accelerator is synced before continuing to next epoch
        if components["use_contrastive"]:
            components["accelerator1"].wait_for_everyone()
        components["accelerator2"].wait_for_everyone()


