# Autoregressive Semantic Embedding Model

This module trians Autoregressive Model using the Encoder / Decoder obtained after training for Semantic embedings.

## Model Architecture

The model consists of three main components:

1. **Encoder**: Converts input text into semantic embeddings
2. **Latent Model**: Predicts the next semantic embedding in the reasoning chain
3. **Decoder**: Converts semantic embeddings back to natural language text

### Training

To train the model:

1. Configure the parameters in `scripts/run.sh`:

2. Run the training script:
```bash
cd scripts && bash run.sh
```

3. To train `Sem -> Ctx`, make sure you use the decoder as `decoder2` obtained from when training `emb_ctx`.

### Configuration Options

The model supports several training configurations. Leave it as it is for replication.

- `--freeze`: Freeze encoder and  parameters during training
- `--use_cont`: Use contrastive loss