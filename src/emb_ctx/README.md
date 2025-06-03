# Contextual Embeddings

This module implements constructing **contextual embedding** (Section 2) for sentence-level autoregressive modeling as described in the paper. 
Contextual embeddings are derived through a predictive objective, mainly next sentence prediction.

## Overview

Contextual embeddings are constructed by forming context-target pairs where:
- Context (`x`): Includes the question and preceding reasoning steps `(q, s₁, ..., sᵢ₋₁)`
- Target (`y`): The current reasoning step `sᵢ`

This approach ensures embeddings capture predictive cues essential for reasoning step generation.

## Details

The framework uses a decoder-only Transformer (GPT-2) with shared parameters for encoding and decoding. For an input sequence, the encoder produces hidden states, with the final hidden state serving as the latent representation of the entire sequence.

Two variants are implemented:
- **Contextual-Base (CTX-B)**: Standard contextual embeddings
- **Contextual-Contrastive (CTX-C)**: Adds contrastive regularization loss (InfoNCE) to align contextual embeddings closer to corresponding semantic embeddings 
## Run

The main execution script is `run.sh` which allows you to train and evaluate contextual embeddings. When using this script, pay special attention to the `--use_cont` flag:

- `--use_cont`: Enables the contrastive regularization loss (InfoNCE) that aligns contextual embeddings with semantic embeddings. (i.e. enables CTX-C)
- When you are using this flag (CTX-C mode), make sure you set `ENCODER1_MODEL` and `DECODER1_MODEL` as the Encoder/Decoder obtained after training for Semantic Embedding.

Example usage:
```bash
# Simply run in this directory:
cd scripts && bash run.sh 
```