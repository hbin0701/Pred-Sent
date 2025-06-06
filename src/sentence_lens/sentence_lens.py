import torch
import argparse
import os
import numpy as np
from transformers import AutoTokenizer
from models import AutoEncoderModel
from dataset import StepsDataset
import warnings
from transformers import logging as transformers_logging
import json
from tqdm.auto import tqdm

# Suppress the specific warning
transformers_logging.set_verbosity_error()

def load_model_and_data(model_dir, data_path, num_examples=None):
    """Load the model and data."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir + "/tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoEncoderModel.load_model(
        model_dir,
        tokenizer=tokenizer,
        freeze=True  # Freeze model parameters for visualization
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load dataset using StepsDataset
    print(f"Loading dataset from {data_path}")
    dataset = StepsDataset(
        file_path=data_path,
        tokenizer=tokenizer,
        max_length=512
    )
    
    # Limit the number of examples if specified
    if num_examples is not None:
        dataset.processed_data = dataset.processed_data.select(range(min(num_examples, len(dataset))))
    
    return model, tokenizer, dataset, device

def iterative_sentence_lens(model, tokenizer, encoder_input_ids, device, example_idx=0, num_iterations=6, target_layers=None):
    """
    Implement iterative Sentence Lens: 
    1. Generate hidden states over multiple iterations
    2. For each iteration, map each layer's hidden state to the decoder
    
    Args:
        model: The model to analyze
        tokenizer: Tokenizer for decoding outputs
        encoder_input_ids: Input IDs for the encoder
        device: Device to run computation on
        example_idx: Which example to analyze
        num_iterations: Number of latent_model iterations to perform
        target_layers: List of target latent_model layers to extract from (None for all)
    
    Returns:
        Dictionary containing input details and nested results for each iteration and layer
    """
    model.eval()
    
    # Get input details
    input_ids = encoder_input_ids[example_idx:example_idx+1]
    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    # Extract the question part (before the first newline)
    newline_positions = (input_ids[0] == model.newline_token_id).nonzero(as_tuple=False)
    first_newline = newline_positions[0].item() if newline_positions.numel() > 0 else input_ids.size(1) - 1
    question_ids = input_ids[0, :first_newline+1]
    question_text = tokenizer.decode(question_ids, skip_special_tokens=True)
    
    # Setup initial latent_model inputs
    latent_model_inputs, latent_model_att_mask, latent_model_position_ids = model.pad_question(input_ids)
    latent_model_inputs_embeds = model.latent_model.transformer.wte(latent_model_inputs)
    
    # If no target layers specified, use all latent_model layers
    if target_layers is None:
        num_layers = model.latent_model.transformer.config.n_layer
        target_layers = list(range(num_layers))
    
    # Initialize results structure
    results = {
        'input': {
            'full_text': input_text,
            'question': question_text,
            'token_ids': input_ids[0].cpu().tolist(),
            'question_token_ids': question_ids.cpu().tolist()
        },
        'iterations': {},
        'final_generations': {f"layer_{layer_idx}": [] for layer_idx in target_layers}
    }
    
    # Run iterative decoding process
    for iter_idx in tqdm(range(num_iterations)):
        print(f"Processing iteration {iter_idx+1}/{num_iterations}...")
        
        # Initialize iteration results
        iter_results = {'layer_outputs': {}}
        
        with torch.no_grad():
            # Get all hidden states from latent_model
            latent_model_outputs = model.latent_model.transformer(
                inputs_embeds=latent_model_inputs_embeds,
                attention_mask=latent_model_att_mask,
                position_ids=latent_model_position_ids,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get hidden states from each specified layer
            hidden_states = latent_model_outputs.hidden_states
            
            # Store the last hidden state for the next iteration
            latent_model_hidden = latent_model_outputs.last_hidden_state
            last_hidden = latent_model_hidden[:, -1, :].unsqueeze(1)
            
            # Collect all layer hidden states for batch processing
            layer_hidden_states = []
            for layer_idx in target_layers:
                layer_hidden = hidden_states[layer_idx + 1]  # +1 because first element is input embeddings
                layer_last_hidden = layer_hidden[:, -1, :].unsqueeze(1)
                layer_hidden_states.append(layer_last_hidden)
            
            # Stack all layer hidden states
            batched_hidden = torch.cat(layer_hidden_states, dim=0)  # [num_layers, 1, hidden_size]
            
            # Project to decoder space
            target_out_unprojected = batched_hidden.reshape(-1, batched_hidden.size(-1))
            target_out_projected = model.latent_model_to_decoder_proj(target_out_unprojected)
            
            # Generate text with decoder for all layers at once
            decoder_out = model.decoder.generate(
                inputs_embeds=target_out_projected.unsqueeze(1),
                max_new_tokens=50,
                do_sample=False,
                temperature=0,
            )
            
            # Process outputs for each layer
            for layer_idx, layer_out in zip(target_layers, decoder_out):
                decoded_text = tokenizer.decode(layer_out, skip_special_tokens=True)
                iter_results['layer_outputs'][f"layer_{layer_idx}"] = decoded_text
                results['final_generations'][f"layer_{layer_idx}"].append(decoded_text)
            
            # Save iteration results
            results['iterations'][f"iteration_{iter_idx}"] = iter_results
            
            # Update latent_model inputs for next iteration
            latent_model_inputs_embeds = torch.cat([latent_model_inputs_embeds, last_hidden], dim=1)
            latent_model_att_mask = torch.cat(
                [latent_model_att_mask, torch.ones((latent_model_att_mask.size(0), 1), device=device)],
                dim=1,
            )
            new_position_ids = latent_model_position_ids[:, -1] + 1
            latent_model_position_ids = torch.cat(
                [latent_model_position_ids, new_position_ids.unsqueeze(1)], dim=1
            )
    
    # Join all generations for each layer with newlines
    for layer_idx in target_layers:
        results['final_generations'][f"layer_{layer_idx}"] = "\n".join(results['final_generations'][f"layer_{layer_idx}"])
    
    return results

def write_results_to_json(results, output_path):
    """
    Write the iterative Sentence Lens results to a JSON file with nice formatting.
    
    Args:
        results: Dictionary containing the results from iterative_sentence_lens
        output_path: Path where to save the JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓  Saved results to JSON file → {output_path}")

def process_dataset(model, tokenizer, dataset, device, num_iterations=6, target_layers=None, output_dir=None):
    """
    Process the entire dataset and save results for each example in separate JSON files.
    
    Args:
        model: The model to analyze
        tokenizer: Tokenizer for decoding outputs
        dataset: StepsDataset instance
        device: Device to run computation on
        num_iterations: Number of latent_model iterations to perform
        target_layers: List of target latent_model layers to extract from (None for all)
        output_dir: Directory to save individual JSON files
    """
    num_examples = len(dataset)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for example_idx in tqdm(range(num_examples), desc="Processing examples"):
        # Get sample from dataset
        sample = dataset[example_idx]
        
        # Convert to tensors and move to device
        encoder_input_ids = torch.tensor(sample["encoder_input_ids"]).unsqueeze(0).to(device)
        
        # Generate results for this example
        results = iterative_sentence_lens(
            model, tokenizer, encoder_input_ids, device,
            example_idx=0,  # Always 0 since we're passing single example
            num_iterations=num_iterations,
            target_layers=target_layers
        )
        
        # Save to individual JSON file
        output_path = os.path.join(output_dir, f"example_{example_idx}.json")
        write_results_to_json(results, output_path)

def main(args):
    # Load model and data
    model, tokenizer, dataset, device = load_model_and_data(
        args.model_dir, args.data_path, args.num_examples
    )
    
    # Parse target layers if provided
    target_layers = None
    if args.target_layers:
        target_layers = [int(layer) for layer in args.target_layers.split(',')]
    
    # Process entire dataset and save individual files
    print("Processing entire dataset...")
    process_dataset(
        model, tokenizer, dataset, device,
        num_iterations=args.num_iterations,
        target_layers=target_layers,
        output_dir=args.output_dir
    )
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model representations using iterative Sentence Lens")
    
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing the saved model")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to JSON dataset file")
    
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Number of examples to process (default: all)")
    
    parser.add_argument("--num_iterations", type=int, default=6,
                        help="Number of iterations to perform")
    
    parser.add_argument("--target_layers", type=str, default=None,
                        help="Comma-separated list of target layers to analyze (default: all layers)")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save individual JSON result files")
    
    args = parser.parse_args()
    main(args) 