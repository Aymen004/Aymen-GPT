# Aymen-GPT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A completely from-scratch implementation of the GPT2/3 architecture trained on FineWeb-Edu dataset, with performance evaluation on the HellaSwag benchmark. [View Research Report](https://drive.google.com/file/d/1LEzPnJ3gGAv_sDc7MuuNBgVsr5K_tjpR/view?usp=sharing)

## Overview

Aymen-GPT is a PyTorch implementation of the GPT architecture that achieves performance comparable to GPT-3 Small (125M parameters) while being trained on only 40B tokens (compared to 300B tokens for GPT-3 Small).
The model training was done using 8 A100 (80 GB SXM4) with 80GB of memory each from Vast.ai, and took approximately 7 hours to complete.

Key features:
- Implementation of the transformer architecture from scratch
- Support for Flash Attention 2 for improved performance
- Distributed training with PyTorch DDP (Distributed Data Parallel)
- Training on the FineWeb-Edu dataset
- Evaluation on the HellaSwag benchmark

## Results

Aymen-GPT achieves performance close to GPT-3 Small on the HellaSwag benchmark:

| Model | HellaSwag Accuracy | Training Tokens | Parameters | Validation Loss | Perplexity |
| ----- | ----------------- | -------------- | ---------- | --------------- | ---------- |
| GPT-2 (124M) | 29.4% | 10B | 124M | 3.29 | 26.8 |
| GPT-3 Small (125M) | 33.7% | 300B | 125M | - | - |
| Aymen-GPT (124M) | ~33.4% | 40B | 124M | 2.80 | 16.4 |


## Model Architecture

Aymen-GPT follows the decoder-only transformer architecture based on the GPT-2/3 papers, with the following specifications:

- **Embedding Layer**: Token and position embeddings with dimension 768
- **Transformer Blocks**: 12 transformer blocks with:
  - Self-attention mechanism with 12 attention heads
  - Layer normalization applied before each sub-block (Pre-LN)
  - Feed-forward networks with 4× expansion
  - GELU activation function with tanh approximation
- **Attention Mechanism**: Causal self-attention with Flash Attention 2 support
- **Parameter Initialization**: Carefully tuned for stable training
  - Linear layers: Initialized with N(0, 0.02)
  - Output projection layers: Scaled initialization
  - Embedding layers: Initialized with N(0, 0.02)

## Training Details

Aymen-GPT was trained with the following hyperparameters:

- **Dataset**: FineWeb-Edu dataset (10B tokens × 4 epochs = 40B tokens total)
- **Optimizer**: AdamW with β₁ = 0.9, β₂ = 0.95, ε = 10⁻⁸
- **Learning Rate Schedule**:
  - Linear warmup for 715 steps (375M tokens)
  - Cosine decay to 10% of peak learning rate
  - Peak learning rate of 6×10⁻⁴
- **Weight Decay**: 0.1 applied only to weight matrices
- **Gradient Clipping**: L2 norm clipped at 1.0
- **Hardware**: 8× A100 (80GB) GPUs


## Setup and Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Aymen-GPT.git
cd Aymen-GPT
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Optional: Install Flash Attention 2 for improved performance:
```bash
pip install flash-attn
```

## Data Preparation

Download and prepare the FineWeb-Edu dataset:

```bash
python fineweb.py
```

This will download the sample-10BT split from the HuggingFaceFW/fineweb-edu dataset and prepare it for training.

## Training

### Single GPU Training

```bash
python pretrain.py
```

### Multi-GPU Training

```bash
torchrun --standalone --nproc_per_node=8 pretrain.py
```

Replace `8` with the number of GPUs you have available.

## Evaluation

Evaluate a trained model on the HellaSwag benchmark:

```bash
python hellaswag.py -m gpt2
```

You can replace `gpt2` with a path to your trained checkpoint.

## Jupyter Notebooks

This project includes a Jupyter notebook (`GPT-nb.ipynb`) that showcases the model's performance on Hellaswag evaluation and provides visualizations of the training and evaluation of Aymen-GPT on the huggingface fineweb-edu dataset (10B).

## Project Structure

- **Main Files:**
  - `pretrain.py`: Main training script with distributed training support
  - `fineweb.py`: Data preparation for the FineWeb-Edu dataset
  - `hellaswag.py`: Evaluation script for the HellaSwag benchmark
  - `GPT-nb.ipynb`: Jupyter notebook with visualizations and analysis
- **Documentation:**
  - `research-report.tex`: Technical report with detailed methodology and results
  - `docs/`: Directory containing images and additional documentation

## Contributing

Contributions are welcome! If you'd like to contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The GPT architecture is based on the [OpenAI GPT](https://github.com/openai/gpt-2) model
- The HellaSwag benchmark is from [HellaSwag: Can a Machine Really Finish Your Sentence?](https://rowanzellers.com/hellaswag/)
- The FineWeb-Edu dataset is from [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
