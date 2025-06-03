# Supervised Fine-Tuning (SFT)

This module implements supervised fine-tuning for language models with both Chain-of-Thought (CoT) and No-CoT approaches.

## Overview

The SFT module provides baseline models for comparison with the sentence-level prediction approach. It includes:

1. **CoT SFT**: Fine-tuning models to generate intermediate reasoning steps before producing an answer
2. **No-CoT SFT**: Fine-tuning models to directly generate answers without explicit reasoning steps

## Usage

Run the training script:
```bash
cd scripts & bash run.sh
```

Note that: MODE="no_cot" or "cot" according to your use.

For evaluation, simply run:
```bash
bash eval.sh
```