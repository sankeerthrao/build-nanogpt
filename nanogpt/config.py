"""
YAML-based configuration system with CLI overrides and validation.

Supports hierarchical configs and strict type checking via dataclasses.
Unknown YAML keys are rejected to catch typos early.
"""

import os
import argparse
from dataclasses import dataclass, fields, asdict
from typing import Optional, get_type_hints

import yaml


@dataclass
class TrainConfig:
    """Complete training configuration with sensible defaults."""

    # --- Model Architecture ---
    block_size: int = 1024
    vocab_size: int = 50304  # padded for efficiency (divisible by 128)
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: Optional[int] = None  # for GQA; None = same as n_head (MHA)
    n_embd: int = 768
    use_rope: bool = False  # Rotary Position Embeddings
    use_swiglu: bool = False  # SwiGLU activation (replaces GELU)
    use_rmsnorm: bool = False  # RMSNorm (replaces LayerNorm)
    rope_theta: float = 10000.0  # RoPE base frequency
    dropout: float = 0.0

    # --- Training ---
    max_steps: int = 19073
    total_batch_size: int = 524288  # 2**19 ~0.5M tokens
    micro_batch_size: int = 64
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 715
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    lr_schedule: str = "cosine"  # cosine, linear, warmup_stable_decay
    stable_fraction: float = 0.8  # for warmup_stable_decay schedule

    # --- Mixed Precision ---
    dtype: str = "bfloat16"  # bfloat16, float16, float32
    use_compile: bool = False

    # --- Gradient Checkpointing ---
    gradient_checkpointing: bool = False

    # --- Data ---
    data_root: str = "edu_fineweb10B"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_split: str = "sample-10BT"
    shard_size: int = 100_000_000  # 100M tokens per shard

    # --- Evaluation ---
    val_interval: int = 250
    val_steps: int = 20
    hellaswag_eval: bool = True
    generate_interval: int = 250
    generate_samples: int = 4
    generate_max_length: int = 32

    # --- Checkpointing ---
    log_dir: str = "log"
    checkpoint_interval: int = 5000
    keep_last_n_checkpoints: int = 3
    save_optimizer: bool = True

    # --- Logging ---
    use_wandb: bool = False
    wandb_project: str = "nanogpt"
    wandb_run_name: Optional[str] = None
    use_tensorboard: bool = False
    log_interval: int = 1

    # --- Generation ---
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0

    # --- Distributed ---
    seed: int = 1337

    def validate(self):
        """Validate configuration consistency."""
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )
        if self.n_kv_head is not None:
            assert self.n_head % self.n_kv_head == 0, (
                f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
            )
        assert self.dtype in ("bfloat16", "float16", "float32"), (
            f"dtype must be bfloat16, float16, or float32, got {self.dtype}"
        )
        assert self.lr_schedule in ("cosine", "linear", "warmup_stable_decay"), (
            f"lr_schedule must be cosine, linear, or warmup_stable_decay, got {self.lr_schedule}"
        )
        assert self.total_batch_size > 0, "total_batch_size must be positive"
        assert self.micro_batch_size > 0, "micro_batch_size must be positive"
        assert self.block_size > 0, "block_size must be positive"
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"

    def to_dict(self):
        return asdict(self)


def load_config(config_path: Optional[str] = None, cli_args: Optional[list] = None) -> TrainConfig:
    """
    Load config from YAML file, then override with CLI arguments.

    Priority: CLI args > YAML file > defaults
    """
    config = TrainConfig()

    # Load from YAML if provided
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f) or {}
        # Apply YAML values (error on unknown keys to catch typos)
        valid_keys = {f.name for f in fields(config)}
        for key, value in yaml_config.items():
            if key not in valid_keys:
                raise ValueError(
                    f"Unknown config key '{key}' in {config_path}. "
                    f"Valid keys: {sorted(valid_keys)}"
                )
            setattr(config, key, value)

    # Parse CLI overrides
    if cli_args is not None:
        parser = _build_arg_parser(config)
        parsed, _ = parser.parse_known_args(cli_args)
        for key, value in vars(parsed).items():
            if value is not None:
                setattr(config, key, value)

    config.validate()
    return config


def _build_arg_parser(config: TrainConfig) -> argparse.ArgumentParser:
    """Dynamically build argparse from TrainConfig fields."""
    parser = argparse.ArgumentParser(description="nanoGPT Training")
    hints = get_type_hints(TrainConfig)
    for f in fields(config):
        name = f"--{f.name}"
        hint = hints.get(f.name, f.type)
        # Resolve Optional[X] -> X, bool -> _str_to_bool
        args = getattr(hint, "__args__", ())
        if hint is bool:
            parser.add_argument(name, type=_str_to_bool, default=None)
        elif args and type(None) in args:
            # Optional[X] — extract the non-None type
            inner = next((a for a in args if a is not type(None)), str)
            parser.add_argument(name, type=inner, default=None)
        else:
            parser.add_argument(name, type=hint, default=None)
    return parser


def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")
