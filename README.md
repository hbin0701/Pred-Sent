# Step‑by‑Step Prediction (Pred‑Sent)

> ⚠️ **Work in Progress**  —  The codebase is under active development. Will be updated soon.

This repository contains the *official* implementation of the paper **“Let’s Predict Step by Step."**

**TL;DR: We present a framework that adapts a pre-
trained token-level LM to operate in sentence space, by autoregressively predicting.**

## Directory Structure

```text
pred-sent/
├── data/                     # Pre‑packed benchmark datasets
│   ├── blocksworld7/
│   ├── csqa/
│   ├── gsm8k/
│   └── prosqa/
│   └── *.json
└── src/                      # All training & inference pipelines
    ├── autoreg_ctx/          # Contextual‑embedding latent model
    ├── autoreg_ctx_w_classifier/   # + termination classifier
    ├── autoreg_sem/          # Semantic‑embedding latent model
    ├── emb_ctx/              # Contextual embedding training
    ├── emb_sem/              # Semantic embedding training
    └── sft/                  # CoT & No‑CoT supervised baselines
```

Each subdirectory contains its own **`train.py`**, **`main.py`**, and **`run.sh`** that can be invoked directly. See teh bash script for more details.

## Model Architecture

Our overall models consist of three principal components 

| Component      | Paper name     | Purpose                                            |
| -------------- | -------------- | -------------------------------------------------- |
| **Encoder**    | *Encoder*      | Convert input query into a vector representation   |
| **Decoder**    | *Latent Model* | Autoregressively emit intermediate reasoning steps |
| **Translator** | *Decoder*      | Map the emitted steps to the final answer          |

> *Naming note*: we are standardizing labels in code to be consistent with paper. Will be updated soon.

## Installation

### 1. Clone & Prepare Environment

```bash
# Clone repository
$ git clone https://github.com/hbin0701/pred-sent.git
$ cd pred-sent

# Create Conda environment (PyTorch 2.3 + CUDA 12 by default)
$ conda env create -f environment.yml
$ conda activate pred-sent
```

### 2. Download Checkpoints (optional)

TBD. We will enable this via:

```bash
$ bash scripts/download_data.sh  # ~5 minutes / <1 GB
```