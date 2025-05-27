import torch
from tqdm import tqdm
import wandb
import pandas as pd
import re
import os
import json
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# def extract_answer(generated_steps):
#     """
#     Extract the final answer from generated steps using ">>\n" as a marker.
    
#     Args:
#         generated_steps: List of strings, each representing a generated step
    
#     Returns:
#         The extracted answer or empty string if no answer marker found
#     """
#     # Combine all generated steps into a single string
#     full_text = "\n".join(generated_steps)
    
#     # Find the last occurrence of ">>\n"
#     marker = ">>\n"
#     idx = full_text.rfind(marker)
    
#     if idx == -1:
#         return ""  # Marker not found
    
#     # Extract the text after the marker up to the next newline
#     answer_start = idx + len(marker)
#     remaining_text = full_text[answer_start:]
    
#     # Get the first line (up to the next newline or end of string)
#     lines = remaining_text.split("\n")
#     if not lines:
#         return ""
        
#     return lines[0].strip()

def extract_answer(generated_steps, task):
    full_text = "\n".join(generated_steps)
    
    if task == "csqa":
        # Find the last occurrence of ">>\n"
        marker = "###"
        idx = full_text.find(marker)

        if idx == -1:
            return ""
        
        return full_text[idx:].split("\n")[0].replace("###", "").strip()
    
    elif task == "gsm8k":
        
        try:
            target = [x for x in generated_steps if ">>" not in x][0]
            return target.split("\n")[0].strip()
        except:
            return generated_steps[-1]
        
    else:
        raise NotImplementedError(f"Task {task} not implemented")

def create_html_visualization(sample_data, mode, step):
    """
    Create a beautiful HTML visualization for wandb from sample data.
    
    Args:
        sample_data: List of dictionaries containing sample results
        mode: Evaluation mode (train, eval, test)
        step: Current training step
        
    Returns:
        HTML string for wandb visualization
    """
    if not sample_data:
        return None
    
    # Start HTML
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .sample {{ border: 1px solid #ddd; padding: 15px; margin-bottom: 30px; border-radius: 5px; }}
            .sample h3 {{ margin-top: 0; color: #333; }}
            .question {{ background: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 15px; }}
            .steps {{ padding: 10px; background: #f9f9f9; border-left: 3px solid #4285f4; margin-bottom: 15px; white-space: pre-wrap; }}
            .result {{ display: flex; align-items: center; margin-bottom: 10px; }}
            .result .label {{ width: 120px; font-weight: bold; }}
            .correct {{ color: #0f9d58; }}
            .incorrect {{ color: #db4437; }}
            .step-number {{ font-weight: bold; color: #4285f4; }}
            
            /* Toggle controls */
            .toggle-controls {{ margin-bottom: 20px; display: flex; flex-wrap: wrap; }}
            .toggle-btn {{ padding: 8px 12px; margin-right: 10px; margin-bottom: 10px; 
                          background: #f1f1f1; border: 1px solid #ddd; border-radius: 4px; 
                          cursor: pointer; transition: all 0.3s; }}
            .toggle-btn.active {{ background: #4285f4; color: white; }}
            .timestamp-section {{ display: none; }}
            .timestamp-section.active {{ display: block; }}
            .summary-box {{ background: #e8f0fe; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .metrics {{ font-weight: bold; margin-bottom: 10px; }}
        </style>
        <script>
            function showTimestamp(stepId) {{
                /* Hide all sections */
                const sections = document.querySelectorAll('.timestamp-section');
                sections.forEach(section => {{
                    section.classList.remove('active');
                }});
                
                /* Deactivate all buttons */
                const buttons = document.querySelectorAll('.toggle-btn');
                buttons.forEach(button => {{
                    button.classList.remove('active');
                }});
                
                /* Show selected section and activate button */
                document.getElementById('section-' + stepId).classList.add('active');
                document.getElementById('btn-' + stepId).classList.add('active');
            }}
            
            /* Initialize with the latest timestamp visible */
            document.addEventListener('DOMContentLoaded', function() {{
                const latestStep = document.querySelector('.toggle-btn:last-child');
                if (latestStep) {{
                    latestStep.click();
                }}
            }});
        </script>
    </head>
    <body>
        <div class="container">
            <h2>Problem Solving Results - {mode.upper()}</h2>
            
            <div class="toggle-controls">
                <button id="btn-{step}" class="toggle-btn active" onclick="showTimestamp({step})">Step {step}</button>
            </div>
            
            <div id="section-{step}" class="timestamp-section active">
                <div class="summary-box">
                    <div class="metrics">Step {step} Metrics:</div>
                    <div>Total examples: {len(sample_data)}</div>
                    <div>Correct answers: {sum(1 for item in sample_data if item["Correct"])}</div>
                    <div>Accuracy: {sum(1 for item in sample_data if item["Correct"]) / len(sample_data) * 100:.2f}%</div>
                </div>
    """
    
    # Add samples
    for i, sample in enumerate(sample_data):
        is_correct = sample["Correct"]
        status_class = "correct" if is_correct else "incorrect"
        status_text = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        
        # Format the steps with step numbers
        steps_text = ""
        steps = sample["Generated Steps"].split("\n")
        for j, step_text in enumerate(steps):
            if step_text.strip():
                steps_text += f"<span class='step-number'>Step {j+1}:</span> {step_text}\n"
        
        html += f"""
                <div class="sample">
                    <h3>Example {i+1}</h3>
                    <div class="question">{sample["Question"]}</div>
                    <div class="steps">{steps_text}</div>
                    <div class="result">
                        <div class="label">Extracted Answer:</div>
                        <div>{sample["Extracted Answer"] or "(none)"}</div>
                    </div>
                    <div class="result">
                        <div class="label">Ground Truth:</div>
                        <div>{sample["Ground Truth"]}</div>
                    </div>
                    <div class="result">
                        <div class="label">Result:</div>
                        <div class="{status_class}">{status_text}</div>
                    </div>
                </div>
        """
    
    # End timestamp section and HTML
    html += """
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

# Keep track of samples across multiple timestamps
_sample_history = {}

def update_html_visualization(new_sample_data, mode, step):
    """
    Update the HTML visualization with new samples from a new timestamp.
    Creates a toggleable UI to switch between different timestamps.
    
    Args:
        new_sample_data: List of dictionaries containing new sample results
        mode: Evaluation mode (train, eval, test)
        step: Current training step
    
    Returns:
        Updated HTML string for wandb visualization
    """
    global _sample_history
    
    # Initialize history for this mode if needed
    if mode not in _sample_history:
        _sample_history[mode] = {}
    
    # Store the new samples for this step
    _sample_history[mode][step] = new_sample_data
    
    # If no samples, return None
    if not _sample_history[mode]:
        return None
    
    # Start HTML
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .sample {{ border: 1px solid #ddd; padding: 15px; margin-bottom: 30px; border-radius: 5px; }}
            .sample h3 {{ margin-top: 0; color: #333; }}
            .question {{ background: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 15px; }}
            .steps {{ padding: 10px; background: #f9f9f9; border-left: 3px solid #4285f4; margin-bottom: 15px; white-space: pre-wrap; }}
            .result {{ display: flex; align-items: center; margin-bottom: 10px; }}
            .result .label {{ width: 120px; font-weight: bold; }}
            .correct {{ color: #0f9d58; }}
            .incorrect {{ color: #db4437; }}
            .step-number {{ font-weight: bold; color: #4285f4; }}
            
            /* Toggle controls */
            .toggle-controls {{ margin-bottom: 20px; display: flex; flex-wrap: wrap; }}
            .toggle-btn {{ padding: 8px 12px; margin-right: 10px; margin-bottom: 10px; 
                          background: #f1f1f1; border: 1px solid #ddd; border-radius: 4px; 
                          cursor: pointer; transition: all 0.3s; }}
            .toggle-btn.active {{ background: #4285f4; color: white; }}
            .timestamp-section {{ display: none; }}
            .timestamp-section.active {{ display: block; }}
            .summary-box {{ background: #e8f0fe; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .metrics {{ font-weight: bold; margin-bottom: 10px; }}
        </style>
        <script>
            function showTimestamp(stepId) {{
                /* Hide all sections */
                const sections = document.querySelectorAll('.timestamp-section');
                sections.forEach(section => {{
                    section.classList.remove('active');
                }});
                
                /* Deactivate all buttons */
                const buttons = document.querySelectorAll('.toggle-btn');
                buttons.forEach(button => {{
                    button.classList.remove('active');
                }});
                
                /* Show selected section and activate button */
                document.getElementById('section-' + stepId).classList.add('active');
                document.getElementById('btn-' + stepId).classList.add('active');
            }}
            
            /* Initialize with the latest timestamp visible */
            document.addEventListener('DOMContentLoaded', function() {{
                const latestStep = document.querySelector('.toggle-btn:last-child');
                if (latestStep) {{
                    latestStep.click();
                }}
            }});
        </script>
    </head>
    <body>
        <div class="container">
            <h2>Problem Solving Results - {mode.upper()}</h2>
            
            <div class="toggle-controls">
    """
    
    # Add toggle buttons for each timestamp, sorted by step number
    sorted_steps = sorted(_sample_history[mode].keys())
    for history_step in sorted_steps:
        html += f'<button id="btn-{history_step}" class="toggle-btn" onclick="showTimestamp({history_step})">Step {history_step}</button>\n'
    
    html += """
            </div>
    """
    
    # Add content sections for each timestamp
    for history_step in sorted_steps:
        samples = _sample_history[mode][history_step]
        
        # Calculate metrics for this step
        total = len(samples)
        correct = sum(1 for item in samples if item["Correct"])
        accuracy = (correct / total * 100) if total > 0 else 0
        
        html += f"""
            <div id="section-{history_step}" class="timestamp-section">
                <div class="summary-box">
                    <div class="metrics">Step {history_step} Metrics:</div>
                    <div>Total examples: {total}</div>
                    <div>Correct answers: {correct}</div>
                    <div>Accuracy: {accuracy:.2f}%</div>
                </div>
        """
        
        # Add samples for this timestamp
        for i, sample in enumerate(samples):
            is_correct = sample["Correct"]
            status_class = "correct" if is_correct else "incorrect"
            status_text = "✓ CORRECT" if is_correct else "✗ INCORRECT"
            
            # Format the steps with step numbers
            steps_text = ""
            steps = sample["Generated Steps"].split("\n")
            for j, step_text in enumerate(steps):
                if step_text.strip():
                    steps_text += f"<span class='step-number'>Step {j+1}:</span> {step_text}\n"
            
            html += f"""
                <div class="sample">
                    <h3>Example {i+1}</h3>
                    <div class="question">{sample["Question"]}</div>
                    <div class="steps">{steps_text}</div>
                    <div class="result">
                        <div class="label">Extracted Answer:</div>
                        <div>{sample["Extracted Answer"] or "(none)"}</div>
                    </div>
                    <div class="result">
                        <div class="label">Ground Truth:</div>
                        <div>{sample["Ground Truth"]}</div>
                    </div>
                    <div class="result">
                        <div class="label">Result:</div>
                        <div class="{status_class}">{status_text}</div>
                    </div>
                </div>
            """
            
        # Close this timestamp section
        html += """
            </div>
        """
    
    # End HTML
    html += """
        </div>
    </body>
    </html>
    """
    
    return html

def generate_step(model, contexts, device):
    """Generate next step for each context in batch."""
    batch_size = len(contexts)
    
    # Tokenize current contexts for the batch
    encoded_contexts = model.tokenizer(
        contexts, 
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        # Use the model directly - we know it's the prediction model
        encoder = model.encoder
        decoder = model.decoder
            
        # Get encoder outputs
        encoder_outputs = encoder.transformer(
            encoded_contexts.input_ids, 
            attention_mask=encoded_contexts.attention_mask
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Get the last token representation
        last_token_indices = encoded_contexts.attention_mask.sum(dim=1) - 1
        batch_range = torch.arange(batch_size, device=device)
        encoder_last_token_hidden_state = encoder_hidden_states[batch_range, last_token_indices]
        
        # Prepare for generation
        prefix_for_generation = encoder_last_token_hidden_state.unsqueeze(1)
        
        # Generate next step
        generated_ids = decoder.generate(
            inputs_embeds=prefix_for_generation,
            max_new_tokens=32,
            do_sample=False,
            temperature=0.0
        )
        
        # Decode the generated steps
        generated_texts = [
            model.tokenizer.decode(generated_ids[i], skip_special_tokens=True).strip()
            for i in range(batch_size)
        ]
        
    return generated_texts

def log_generation(contexts, generated_texts, questions, batch_idx, step, mode):
    """Log generated steps to file."""
    # Create directory for generation logs if it doesn't exist
    os.makedirs("generation_logs", exist_ok=True)
    log_file = f"generation_logs/{mode}_generations_step{step}.jsonl"
    
    # Log each generation
    with open(log_file, "a") as f:
        for i, generated_text in enumerate(generated_texts):
            log_entry = {
                "step": step,
                "mode": mode,
                "batch_idx": batch_idx,
                "sample_idx": i,
                "generation_idx": contexts[i].count('\n'),  # Approximate step count
                "question": questions[i],
                "current_context": contexts[i],
                "generated_text": generated_text
            }
            f.write(json.dumps(log_entry) + "\n")

def evaluate_problem_solving(model, problem_dataloader, device, accelerator, step, mode, num_generations, max_samples=1500):
    """Evaluate problem-solving capabilities of the prediction model."""
    problem_solving_acc = 0
    total_problems = 0
    sample_data = []  # For HTML visualization
    
    print("Starting problem solving evaluation...")
    
    total_samples_evaluated = 0
    for batch_idx, batch in enumerate(tqdm(problem_dataloader, desc="Evaluating Problem Solving", leave=False)):
        # Check if we've processed max_samples
        batch_size = len(batch["question"])
        if total_samples_evaluated >= max_samples:
            break
        
        answers = batch["answer"]
        questions = batch["question"]
        
        total_samples_evaluated += batch_size
        total_problems += batch_size
        
        # Initialize contexts with questions for all samples in batch
        contexts = [q + "\n" for q in questions]
        all_generated_steps = [[] for _ in range(batch_size)]
        
        # Iterative generation for the entire batch
        for gen_step in range(num_generations):
            try:
                generated_texts = generate_step(model, contexts, device)
                
                # Update contexts and store generated steps
                for i, text in enumerate(generated_texts):
                    all_generated_steps[i].append(text)
                    contexts[i] += text.strip() + "\n"
                
                # Log generations
                log_generation(contexts, generated_texts, questions, batch_idx, step, mode)
                
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                continue
                
        # Evaluate the final answers
        for i in range(batch_size):
            try:
                # Extract answer from generated steps
                predicted_answer = extract_answer(all_generated_steps[i], model.task)
                ground_truth = answers[i].strip().lower()
                
                # Compare prediction with ground truth
                is_correct = predicted_answer.lower() == ground_truth if predicted_answer else False
                if is_correct:
                    problem_solving_acc += 1
                                    
                # Save sample for visualization
                if batch_idx < 5:  # Only collect samples from first few batches
                    sample_data.append({
                        "Question": questions[i],
                        "Generated Steps": "\n".join(all_generated_steps[i]),
                        "Extracted Answer": predicted_answer,
                        "Ground Truth": ground_truth,
                        "Correct": is_correct
                    })
                    
            except Exception as e:
                print(f"Error processing final answer: {str(e)}")
                continue

    # Create visualization if samples are available
    if sample_data:
        try:
            # Create visualizations directory if it doesn't exist
            os.makedirs("visualizations", exist_ok=True)
            
            html_content = create_html_visualization(sample_data, mode, step)
            # Save the HTML to a file
            with open(f"visualizations/{mode}_step{step}.html", "w") as f:
                f.write(html_content)
            
            # Also log to wandb if accelerator is main process
            if accelerator.is_main_process:
                wandb.log({f"{mode}_problem_samples": wandb.Html(html_content)})
                
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
    
    # Calculate final problem-solving accuracy
    problem_solving_acc = problem_solving_acc / total_problems * 100 if total_problems > 0 else 0
    
    return problem_solving_acc, total_problems

def test_standard_metrics(model, dataloader, device, step, task_type, max_samples=3000):
    # Edited to max_samples=3000. actually doesn't matter that much :p
    """Test standard next step prediction metrics (accuracy and BLEU)."""
    total_acc = 0
    total_steps = 0
    curr_samples = 0
    total_bleu = 0.0
    
    # Determine which inputs to use based on column names in the dataloader
    sample_batch = next(iter(dataloader))
    has_dual_inputs = all(k in sample_batch for k in ["encoder_input_ids1", "encoder_input_ids2"])
    
    # Regular testing
    for batch in tqdm(dataloader, desc="Testing", leave=False):
        if task_type == "restoration":
                # Testing restoration model
                encoder_input_ids = batch["encoder_input_ids1"].to(device)
                encoder_attention_mask = batch["encoder_attention_mask1"].to(device)
                decoder_input_ids = batch["decoder_input_ids1"].to(device)
                decoder_attention_mask = batch["decoder_attention_mask1"].to(device)
        else:
                # Testing prediction model
                encoder_input_ids = batch["encoder_input_ids2"].to(device)
                encoder_attention_mask = batch["encoder_attention_mask2"].to(device)
                decoder_input_ids = batch["decoder_input_ids2"].to(device)
                decoder_attention_mask = batch["decoder_attention_mask2"].to(device)
    
        curr_samples += encoder_input_ids.size(0)
        if curr_samples > max_samples:
            break
        
        with torch.no_grad():
            metrics = model.test(
                encoder_input_ids,
                encoder_attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                step=step
            )
        
        # Accumulate metrics
        total_acc += metrics["next_step_acc"]
        total_steps += metrics["total_steps"]
        total_bleu += metrics["avg_bleu"] * metrics["total_steps"]

    # Calculate overall metrics
    overall_acc = total_acc / total_steps * 100 if total_steps > 0 else 0
    overall_bleu = total_bleu / total_steps if total_steps > 0 else 0
    
    return overall_acc, overall_bleu, total_steps

def test_model(model, dataloader, device, accelerator, step, mode="eval", MAX_SAMPLES=5000, MAX_PROB_SAMPLES=1500, problem_dataloader=None, num_generations=10):
    """
    Test function for model on restoration or prediction task.
    
    Args:
        model: Model to evaluate (either restoration or prediction model)
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        accelerator: Accelerator
        step: Current step number
        mode: Evaluation mode (train, eval, test)
        MAX_SAMPLES: Maximum number of samples to evaluate
        problem_dataloader: DataLoader for problem evaluation (only used for prediction model)
        num_generations: Number of steps to generate for problem solving
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
   
    # Determine task type based on if we're doing problem evaluation
    task_type = "prediction" if problem_dataloader is not None else "restoration"
     
    # Get standard next-step prediction metrics
    overall_acc, overall_bleu, total_steps = test_standard_metrics(
        model, dataloader, device, step, task_type, max_samples=MAX_SAMPLES
    )
    
    # Convert metrics to tensors for gathering
    metrics_tensors = {
        "acc": torch.tensor([overall_acc], device=device),
        "bleu": torch.tensor([overall_bleu], device=device),
        "steps": torch.tensor([total_steps], device=device)
    }
    
    # Gather metrics from all processes
    gathered_metrics = {}
    for key, tensor in metrics_tensors.items():
        gathered_metrics[key] = accelerator.gather(tensor)
    
    # Calculate averages across all processes
    avg_acc = gathered_metrics["acc"].mean().item()
    avg_bleu = gathered_metrics["bleu"].mean().item()
    total_steps = gathered_metrics["steps"].sum().item()

    # Initialize metrics dictionary
    metrics_to_log = {
        f"{mode}_acc/{task_type}": avg_acc,
        f"{mode}_bleu/{task_type}": avg_bleu,
    }
    
    # Problem solving evaluation (only for prediction models)
    if problem_dataloader is not None:
        problem_solving_acc, total_problems = evaluate_problem_solving(
            model, problem_dataloader, device, accelerator, step, mode, 
            num_generations, max_samples=MAX_PROB_SAMPLES
        )
        
        # Convert problem solving metrics to tensors for gathering
        problem_metrics = {
            "acc": torch.tensor([problem_solving_acc], device=device),
            "problems": torch.tensor([total_problems], device=device)
        }
        
        # Gather problem solving metrics
        gathered_problem_metrics = {}
        for key, tensor in problem_metrics.items():
            gathered_problem_metrics[key] = accelerator.gather(tensor)
        
        # Calculate average problem solving accuracy across all processes
        avg_problem_acc = gathered_problem_metrics["acc"].mean().item()
        total_problems = gathered_problem_metrics["problems"].sum().item()
        
        # Add problem solving metrics to log
        metrics_to_log[f"{mode}_acc/problem_solving"] = avg_problem_acc
    
    # Log metrics to wandb only on main process
    if accelerator.is_main_process:
        wandb.log(metrics_to_log)
    
    return metrics_to_log 