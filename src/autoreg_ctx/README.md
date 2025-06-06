# Autoregressive Contextual Embedding Model

This module trains Autoregressive Model using the Encoder / Decoder obtained after training for Contextual embedings.

## Model Architecture

The model consists of three main components:

1. **Encoder**: Converts input text into contextual embeddings
2. **Latent Model**: Predicts the next contextual embedding in the reasoning chain
3. **Decoder**: Converts contextual embeddings back to natural language text

### Training

To train the model:

1. Configure the parameters in `scripts/run.sh`:

2. Run the training script:
```bash
cd scripts && bash run.sh
```

3. To train `CTX-C`, make sure you use the CTX-C encoder / decoder obtained

### Configuration Options

The model supports several training configurations. Leave it as it is for replication.

- `--freeze`: Freeze encoder and  parameters during training
- `--use_cont`: Use contrastive loss