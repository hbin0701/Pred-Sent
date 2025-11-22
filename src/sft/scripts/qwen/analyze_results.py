import json
import glob
import os
import re

def extract_step_from_checkpoint(checkpoint_name):
    match = re.search(r'checkpoint-(\d+)', checkpoint_name)
    return int(match.group(1)) if match else 0

def extract_answer(text):
    # Extract the last answer after "### " in the text
    answers = re.findall(r'### ([A-E])', text)
    return answers[-1] if answers else None

def analyze_results(results_dir):
    # Get all result files
    result_files = glob.glob(os.path.join(results_dir, '*_results.jsonl'))
    
    accuracies = []
    
    for file_path in result_files:
        correct = 0
        total = 0
        
        # Extract checkpoint name from file path
        checkpoint_name = os.path.basename(file_path).replace('_results.jsonl', '')
        step = extract_step_from_checkpoint(checkpoint_name)
        
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Extract the predicted answer from the chain-of-thought
                    pred_answer = extract_answer(data['preds'][-1])
                    # Extract the correct answer from the reference chain-of-thought
                    true_answer = extract_answer(data['answer'])
                    
                    if pred_answer and true_answer and pred_answer == true_answer:
                        correct += 1
                    total += 1
                except Exception as e:
                    print(f"Error processing line in {checkpoint_name}: {e}")
                    continue
        
        if total > 0:
            accuracy = (correct / total) * 100
            accuracies.append((step, checkpoint_name, accuracy, correct, total))
    
    # Sort by step number
    accuracies.sort(key=lambda x: x[0])
    
    # Print results in a nice format
    print("\n=== CSQA Evaluation Results ===")
    print(f"{'Step':>10} | {'Checkpoint':>30} | {'Accuracy':>8} | {'Correct':>7} | {'Total':>7}")
    print("-" * 70)
    
    for step, checkpoint, accuracy, correct, total in accuracies:
        print(f"{step:>10} | {checkpoint:>30} | {accuracy:>8.2f}% | {correct:>7} | {total:>7}")
    
    if accuracies:
        # Find best checkpoint
        best_accuracy = max(accuracies, key=lambda x: x[2])
        print("\n=== Best Performing Checkpoint ===")
        print(f"Checkpoint: {best_accuracy[1]}")
        print(f"Step: {best_accuracy[0]}")
        print(f"Accuracy: {best_accuracy[2]:.2f}%")
        print(f"Correct: {best_accuracy[3]}/{best_accuracy[4]}")
    else:
        print("\nNo results found.")

if __name__ == "__main__":
    results_dir = "/home/hyeonbin/Pred-Sent/results/qwen/csqa-cot-new"
    analyze_results(results_dir) 