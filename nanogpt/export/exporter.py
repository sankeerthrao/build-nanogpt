"""Model export utilities for ONNX and TorchScript formats."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def export_onnx(
    model: nn.Module,
    output_path: str | os.PathLike,
    block_size: int = 1024,
    opset_version: int = 17,
) -> None:
    """Export model to ONNX format.

    Args:
        model: The GPT model to export.
        output_path: Path where the ONNX file will be saved.
        block_size: Maximum sequence length the model supports.
        opset_version: ONNX opset version to target.
    """
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise ImportError(
            "ONNX export requires the 'onnx' package. "
            "Install it with: pip install nanogpt[export]"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    seq_len = min(block_size, 64)
    dummy_input = torch.zeros(1, seq_len, dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            opset_version=opset_version,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
        )

    file_size = output_path.stat().st_size
    print(f"ONNX model exported to {output_path} ({_format_size(file_size)})")


class _LogitsOnlyWrapper(nn.Module):
    """Wrapper that returns only logits (no None loss) for tracing compatibility."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx):
        logits, _ = self.model(idx)
        return logits


def export_torchscript(
    model: nn.Module,
    output_path: str | os.PathLike,
    block_size: int = 1024,
) -> None:
    """Export model to TorchScript format via tracing.

    The model is wrapped so that only logits are returned (no None loss),
    which is required for TorchScript tracing compatibility.

    Args:
        model: The GPT model to export.
        output_path: Path where the TorchScript file will be saved.
        block_size: Maximum sequence length the model supports.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    wrapper = _LogitsOnlyWrapper(model)
    wrapper.eval()
    seq_len = min(block_size, 64)
    dummy_input = torch.zeros(1, seq_len, dtype=torch.long)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_input,))
        traced.save(str(output_path))

    file_size = output_path.stat().st_size
    print(f"TorchScript model exported to {output_path} ({_format_size(file_size)})")
