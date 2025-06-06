# Semantic Embeddings

This module implements constructing **semantic embedding** (Section 2) for sentence-level autoregressive modeling as described in the paper. Semantic embeddings are derived through a reconstruction objective.

## Overview

Semantic embeddings are constructed by using each reasoning step independently as both input and reconstruction target:
- Input (`x`): A single reasoning step `sᵢ`
- Target (`y`): The same reasoning step `sᵢ`

This approach ensures the embedding encapsulates complete and detailed semantics of the individual reasoning step.

## Details

The framework uses a decoder-only Transformer (GPT-2) with shared parameters for encoding and decoding. For an input sequence, the encoder produces hidden states, with the final hidden state serving as the latent representation of the entire sequence.

The embedding is then used to condition the decoder, which is trained autoregressively with cross-entropy loss to reconstruct the original input.

## Run
    
Example usage:
```bash
# Simply run in this directory:
cd scripts && bash run.sh 
```