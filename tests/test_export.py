"""Tests for model export utilities."""

import os
import tempfile

import pytest
import torch

from nanogpt.model import GPT, GPTConfig
from nanogpt.export.exporter import export_torchscript


class TestExportTorchScript:
    @pytest.fixture
    def small_model(self):
        config = GPTConfig(block_size=32, vocab_size=100, n_layer=2, n_head=4, n_embd=64)
        torch.manual_seed(42)
        return GPT(config)

    def test_export_torchscript(self, small_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            export_torchscript(small_model, path, block_size=32)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_torchscript_produces_output(self, small_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            export_torchscript(small_model, path, block_size=32)
            loaded = torch.jit.load(path)
            dummy = torch.zeros(1, 16, dtype=torch.long)
            output = loaded(dummy)
            # Output is logits only (wrapper strips None loss for tracing)
            assert output.shape == (1, 16, 100)

    def test_export_creates_parent_dirs(self, small_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "deep", "model.pt")
            export_torchscript(small_model, path, block_size=32)
            assert os.path.exists(path)
