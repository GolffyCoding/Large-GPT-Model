# GPT Model Implementation with PyTorch

## Overview
This project implements a GPT-style transformer model using PyTorch. It includes self-attention, transformer blocks, and an embedding layer for token and positional embeddings. The model is designed for large-scale text generation and NLP tasks.

## Features
- Implements a **Self-Attention** mechanism with multiple heads.
- Uses **Layer Normalization** and **Dropout** for stability and regularization.
- Supports **Gradient Checkpointing** to reduce memory usage.
- Includes **GELU Activation** in the feed-forward network.
- Configurable parameters such as:
  - `vocab_size`: Number of tokens in the vocabulary.
  - `embed_size`: Dimensionality of embeddings.
  - `num_layers`: Number of transformer layers.
  - `heads`: Number of attention heads.
  - `dropout`: Dropout rate.
  - `forward_expansion`: Expansion factor in the feed-forward layer.
  - `max_length`: Maximum sequence length.

## Requirements
- Python 3.x
- PyTorch
- CUDA (if using GPU acceleration)

## Installation
1. Install Python dependencies:
   ```sh
   pip install torch torchvision torchaudio
   ```
2. Run the script:
   ```sh
   python gpt-model.py
   ```

## Model Architecture
1. **SelfAttention**: Computes attention scores and updates token embeddings.
2. **TransformerBlock**: Consists of self-attention, layer normalization, and a feed-forward network.
3. **GPT Model**: Stacks multiple transformer blocks and outputs logits over vocabulary.

## Example Usage
```python
import torch
from model import GPT

# Initialize model
model = GPT()

# Dummy input
input_ids = torch.randint(0, 50257, (1, 20))  # Batch size: 1, Sequence length: 20
output = model(input_ids)
print(output.shape)  # Expected output shape: (1, 20, 50257)
```

## Performance & Memory Usage
- The model prints the total number of parameters and estimated memory usage in GB.
- Uses **gradient checkpointing** to reduce memory consumption during training.

## License
This project is open-source under the MIT License.

