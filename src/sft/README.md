# Supervised Fine-Tuning (SFT)

This module implements supervised fine-tuning for language models with both Chain-of-Thought (CoT) and No-CoT approaches.

## Overview

The SFT module provides baseline models for comparison with the sentence-level prediction approach. It includes:

1. **CoT SFT**: Fine-tuning models to generate intermediate reasoning steps before producing an answer
2. **No-CoT SFT**: Fine-tuning models to directly generate answers without explicit reasoning steps

## Usage
- USE_COT=true  # Set to false for No-CoT training


Run the training script:
```bash
bash scripts/run.sh
```

For evaluation:
```bash
bash eval.sh
```

### Configuration Options

The model supports several training configurations:

- `--use_cot`: Enable Chain-of-Thought training (includes reasoning steps)
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients before updating weights
- `--max_length`: Maximum sequence length for training