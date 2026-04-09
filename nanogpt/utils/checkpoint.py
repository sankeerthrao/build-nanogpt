"""
Checkpoint management for nanoGPT training.

Provides save/load utilities with automatic cleanup of old checkpoints.
"""

import os
import re
import glob
from typing import Optional, Tuple

import torch


def save_checkpoint(
    model,
    optimizer,
    step: int,
    val_loss: float,
    config,
    log_dir: str,
    keep_last_n: int = 3,
) -> str:
    """Save a training checkpoint and clean up old ones.

    Checkpoint files are named ``model_{step:05d}.pt`` and stored in
    *log_dir*.  After saving, the oldest checkpoints are removed so that
    at most *keep_last_n* remain.

    Args:
        model: The model (raw, unwrapped from DDP if applicable).
        optimizer: The optimizer.
        step: Current training step.
        val_loss: Validation loss at this step.
        config: Model/training configuration (stored in the checkpoint
            for reproducibility).
        log_dir: Directory to write the checkpoint file.
        keep_last_n: Maximum number of checkpoint files to keep.
            Set to ``0`` or negative to keep all.

    Returns:
        Path to the saved checkpoint file.
    """
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "step": step,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, checkpoint_path)

    # Clean up old checkpoints
    if keep_last_n > 0:
        _cleanup_old_checkpoints(log_dir, keep_last_n)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
) -> Tuple:
    """Load a training checkpoint.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        model: Model instance to load weights into.
        optimizer: Optional optimizer instance to restore state into.

    Returns:
        Tuple of ``(model, optimizer, step, val_loss)``.
        If *optimizer* was ``None``, the returned optimizer is also ``None``
        (even if the checkpoint contained optimizer state).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    step = checkpoint.get("step", 0)
    val_loss = checkpoint.get("val_loss", float("inf"))

    return model, optimizer, step, val_loss


def _cleanup_old_checkpoints(log_dir: str, keep_last_n: int) -> None:
    """Remove oldest checkpoint files, keeping only the most recent *keep_last_n*.

    Checkpoint filenames are expected to match ``model_NNNNN.pt`` where
    ``NNNNN`` is the zero-padded step number.
    """
    pattern = os.path.join(log_dir, "model_*.pt")
    checkpoint_files = glob.glob(pattern)

    # Extract (step_number, filepath) pairs
    step_re = re.compile(r"model_(\d+)\.pt$")
    step_file_pairs = []
    for f in checkpoint_files:
        match = step_re.search(os.path.basename(f))
        if match:
            step_file_pairs.append((int(match.group(1)), f))

    # Sort by step number (ascending)
    step_file_pairs.sort(key=lambda x: x[0])

    # Remove oldest if we have more than keep_last_n
    while len(step_file_pairs) > keep_last_n:
        _, old_path = step_file_pairs.pop(0)
        try:
            os.remove(old_path)
        except OSError:
            pass  # best-effort cleanup
