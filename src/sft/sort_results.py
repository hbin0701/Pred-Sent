#!/usr/bin/env python3

import json
import os
import glob

def sort_results_file(file_path):
    # Read all results
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    
    # Sort by accuracy in descending order
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Write back sorted results
    with open(file_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Sorted {file_path}")

def main():
    # Find all all_results.jsonl files in results directory
    all_results_files = glob.glob('results/**/all_results.jsonl', recursive=True)
    
    for file_path in all_results_files:
        try:
            sort_results_file(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main() 