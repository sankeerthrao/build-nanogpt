# nanoGPT 2.0

[![CI](https://github.com/sankeerthrao/build-nanogpt/actions/workflows/ci.yml/badge.svg)](https://github.com/sankeerthrao/build-nanogpt/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

A **high-performance, modular GPT implementation** built from scratch. Based on Karpathy's [build-nanogpt](https://github.com/karpathy/build-nanogpt) lecture, extended with modern architectural improvements and production tooling.

## What's New in 2.0

This is a ground-up refactoring of the original single-file training script into a proper Python package with modern ML engineering practices:

### Architecture
- **Rotary Position Embeddings (RoPE)** — replacing learned positional embeddings for better length generalization
- **Grouped Query Attention (GQA)** — configurable KV-head compression for memory-efficient attention
- **SwiGLU activation** — the activation function used in LLaMA/Mistral, replacing GELU
- **RMSNorm** — faster, simpler alternative to LayerNorm
- **Gradient checkpointing** — trade compute for memory to train larger models

### Training
- **YAML config system** with CLI overrides and strict validation (typos are caught, not silently ignored)
- **3 LR schedules**: cosine, linear, warmup-stable-decay
- **Checkpoint management** with automatic cleanup and training resume
- **Multi-backend logging**: plain text, JSON-lines, Weights & Biases, TensorBoard

### Generation
- **Advanced sampling**: temperature, top-k, nucleus (top-p), and vectorized repetition penalty
- **Standalone CLI** for inference from any checkpoint

### Engineering
- **Modular package** (`nanogpt/`) with clean separation of concerns
- **78 unit tests** covering model, config, data, utils, and export
- **CI/CD** with GitHub Actions (lint, test across Python 3.9-3.12, security audit)
- **Model export** to ONNX and TorchScript
- **`pyproject.toml`** packaging with optional dependency groups

## Background

We start from the [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (124M) architecture and reproduce it from scratch. With modern hardware, this takes ~1hr and ~$10. The code can also scale to [GPT-3](https://arxiv.org/pdf/2005.14165) sizes.

The accompanying [YouTube lecture](https://youtu.be/l8pRSuU81PU) walks through the original implementation step by step.

> **Note:** This trains a base language model. It does not include chat/instruction finetuning — the model generates ("dreams") internet-style text, not conversational responses.

Sample output after 10B tokens of training:
```
Hello, I'm a language model, and my goal is to make English as easy and fun as possible for everyone
Hello, I'm a language model, so the next time I go, I'll just say, I like this stuff.
Hello, I'm a language model, and the question is, what should I do if I want to be a teacher?
```

## Installation

```bash
# Clone and install
git clone https://github.com/sankeerthrao/build-nanogpt.git
cd build-nanogpt
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# Or specific groups
pip install -e ".[train]"      # datasets, transformers
pip install -e ".[logging]"    # wandb, tensorboard
pip install -e ".[export]"     # onnx, onnxruntime
pip install -e ".[dev]"        # pytest, ruff, mypy
```

## Quick Start

### Train with default config (GPT-2 124M)
```bash
# Download and tokenize FineWeb-Edu dataset
python -m nanogpt.data.fineweb

# Train
python -m nanogpt.train --config configs/gpt2_124m.yaml

# Or with CLI overrides
python -m nanogpt.train --config configs/gpt2_124m.yaml --learning_rate 3e-4 --use_wandb true
```

### Train with modern architecture (LLaMA-style)
```bash
python -m nanogpt.train --config configs/gpt2_modern.yaml
```

### Multi-GPU training
```bash
torchrun --standalone --nproc_per_node=8 -m nanogpt.train --config configs/gpt2_124m.yaml
```

### Generate text from a checkpoint
```bash
python -m nanogpt.generate \
    --checkpoint log/model_19073.pt \
    --prompt "The meaning of life is" \
    --temperature 0.8 \
    --top_p 0.95 \
    --num_samples 3
```

### Resume training from a checkpoint
```bash
python -m nanogpt.train --config configs/gpt2_124m.yaml --resume log/model_05000.pt
```

## Configuration

Training is configured via YAML files with CLI overrides. Unknown keys raise errors to catch typos.

```yaml
# configs/gpt2_124m.yaml - classic GPT-2 reproduction
block_size: 1024
vocab_size: 50304
n_layer: 12
n_head: 12
n_embd: 768
learning_rate: 6.0e-4
lr_schedule: cosine
dtype: bfloat16
```

```yaml
# configs/gpt2_modern.yaml - modern LLaMA-style architecture
use_rope: true          # Rotary Position Embeddings
use_swiglu: true        # SwiGLU activation
use_rmsnorm: true       # RMSNorm
n_kv_head: 4            # Grouped Query Attention (4 KV heads)
lr_schedule: warmup_stable_decay
```

See `nanogpt/config.py` for the full list of configuration options with defaults and descriptions.

## Project Structure

```
build-nanogpt/
├── nanogpt/                  # Main package
│   ├── model.py              # GPT model (MHA/GQA, RoPE, SwiGLU, RMSNorm)
│   ├── config.py             # YAML + CLI configuration system
│   ├── train.py              # Training loop with DDP support
│   ├── generate.py           # Standalone generation script
│   ├── data/
│   │   ├── dataloader.py     # Streaming shard-based data loader
│   │   └── fineweb.py        # FineWeb-Edu download & tokenization
│   ├── eval/
│   │   └── hellaswag.py      # HellaSwag benchmark evaluation
│   ├── utils/
│   │   ├── lr_schedule.py    # LR schedules (cosine, linear, WSD)
│   │   ├── logging.py        # Multi-backend logger (txt, json, wandb, tb)
│   │   └── checkpoint.py     # Save/load/cleanup checkpoints
│   └── export/
│       └── exporter.py       # ONNX and TorchScript export
├── configs/                  # Pre-built YAML configurations
│   ├── gpt2_124m.yaml        # Classic GPT-2 124M reproduction
│   └── gpt2_modern.yaml      # Modern architecture (RoPE+SwiGLU+GQA)
├── tests/                    # 78 unit tests
├── train_gpt2.py             # Original single-file training script
├── fineweb.py                # Original data download script
├── hellaswag.py              # Original evaluation script
└── pyproject.toml            # Package configuration
```

## Architecture Options

| Feature | Classic (GPT-2) | Modern (LLaMA-style) |
|---------|-----------------|---------------------|
| Position embeddings | Learned | RoPE |
| Attention | Multi-Head (MHA) | Grouped Query (GQA) |
| Activation | GELU | SwiGLU |
| Normalization | LayerNorm | RMSNorm |
| Projection bias | Yes | No |

## Model Export

```python
from nanogpt.model import GPT, GPTConfig
from nanogpt.export import export_torchscript, export_onnx

model = GPT(GPTConfig(vocab_size=50304))

# TorchScript
export_torchscript(model, "model.torchscript")

# ONNX (requires pip install nanogpt[export])
export_onnx(model, "model.onnx")
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check nanogpt/ tests/

# Type check
mypy nanogpt/
```

## Original Resources

- [Video lecture: Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU)
- [Discussions](https://github.com/karpathy/build-nanogpt/discussions)
- [Zero To Hero Discord](https://discord.gg/3zy8kqD9Cp) — channel **#nanoGPT**

For production-grade training at scale, see:
- [litGPT](https://github.com/Lightning-AI/litgpt)
- [TinyLlama](https://github.com/jzhang38/TinyLlama)

## License

MIT
