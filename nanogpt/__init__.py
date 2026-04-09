"""
nanoGPT - A high-performance, modular GPT implementation.

Built from scratch following Karpathy's build-nanogpt lecture,
then extended with modern architectural improvements and production tooling.
"""

__version__ = "2.0.0"

from nanogpt.model import GPT, GPTConfig
from nanogpt.config import TrainConfig, load_config

__all__ = ["GPT", "GPTConfig", "TrainConfig", "load_config", "__version__"]
