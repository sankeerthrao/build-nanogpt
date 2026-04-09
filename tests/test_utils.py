"""Tests for utility modules: lr_schedule, checkpoint, logging."""

import json
import os
import tempfile

import pytest
import torch

from nanogpt.model import GPT, GPTConfig
from nanogpt.utils.lr_schedule import get_lr
from nanogpt.utils.checkpoint import save_checkpoint, load_checkpoint
from nanogpt.utils.logging import TrainingLogger


class TestLRSchedule:
    def test_warmup_start(self):
        lr = get_lr(0, max_lr=6e-4, min_lr=6e-5, warmup_steps=100, max_steps=1000)
        assert lr == pytest.approx(6e-4 * 1 / 100)

    def test_warmup_end(self):
        lr = get_lr(99, max_lr=6e-4, min_lr=6e-5, warmup_steps=100, max_steps=1000)
        assert lr == pytest.approx(6e-4 * 100 / 100)

    def test_after_max_steps(self):
        lr = get_lr(2000, max_lr=6e-4, min_lr=6e-5, warmup_steps=100, max_steps=1000)
        assert lr == pytest.approx(6e-5)

    def test_cosine_midpoint(self):
        warmup = 100
        max_steps = 1000
        mid = warmup + (max_steps - warmup) // 2
        lr = get_lr(mid, max_lr=6e-4, min_lr=6e-5, warmup_steps=warmup, max_steps=max_steps)
        expected_mid = (6e-4 + 6e-5) / 2
        assert abs(lr - expected_mid) < 1e-5

    def test_cosine_at_max_steps(self):
        lr = get_lr(1000, max_lr=6e-4, min_lr=6e-5, warmup_steps=100, max_steps=1000)
        assert lr == pytest.approx(6e-5, abs=1e-7)

    def test_linear_schedule(self):
        lr = get_lr(
            550, max_lr=6e-4, min_lr=6e-5, warmup_steps=100,
            max_steps=1000, schedule="linear",
        )
        assert 6e-5 < lr < 6e-4

    def test_linear_at_max_steps(self):
        lr = get_lr(
            1000, max_lr=6e-4, min_lr=6e-5, warmup_steps=100,
            max_steps=1000, schedule="linear",
        )
        assert lr == pytest.approx(6e-5, abs=1e-7)

    def test_warmup_stable_decay_stable_phase(self):
        lr = get_lr(
            200, max_lr=6e-4, min_lr=6e-5, warmup_steps=100,
            max_steps=1000, schedule="warmup_stable_decay", stable_fraction=0.8,
        )
        assert lr == pytest.approx(6e-4)

    def test_warmup_stable_decay_decay_phase(self):
        lr = get_lr(
            900, max_lr=6e-4, min_lr=6e-5, warmup_steps=100,
            max_steps=1000, schedule="warmup_stable_decay", stable_fraction=0.8,
        )
        assert 6e-5 < lr < 6e-4

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="Unknown schedule"):
            get_lr(500, max_lr=6e-4, min_lr=6e-5, warmup_steps=100,
                   max_steps=1000, schedule="exponential")

    def test_monotonic_decrease_cosine(self):
        """LR should monotonically decrease after warmup for cosine schedule."""
        lrs = [
            get_lr(s, max_lr=6e-4, min_lr=6e-5, warmup_steps=100, max_steps=1000)
            for s in range(100, 1001)
        ]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1] + 1e-10


class TestCheckpoint:
    @pytest.fixture
    def model_and_optimizer(self):
        config = GPTConfig(block_size=32, vocab_size=100, n_layer=2, n_head=4, n_embd=64)
        torch.manual_seed(42)
        model = GPT(config)
        optimizer = model.configure_optimizers(0.1, 1e-4, "cpu", master_process=False)
        return model, optimizer, config

    def test_save_and_load(self, model_and_optimizer):
        model, optimizer, config = model_and_optimizer
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model, optimizer, step=100, val_loss=2.5, config=config, log_dir=tmpdir)
            files = os.listdir(tmpdir)
            assert any("model_00100" in f for f in files)

            ckpt_path = os.path.join(tmpdir, "model_00100.pt")
            model2 = GPT(config)
            opt2 = model2.configure_optimizers(0.1, 1e-4, "cpu", master_process=False)
            model2, opt2, step, val_loss = load_checkpoint(ckpt_path, model2, opt2)
            assert step == 100
            assert val_loss == pytest.approx(2.5)

    def test_weights_match_after_load(self, model_and_optimizer):
        model, optimizer, config = model_and_optimizer
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model, optimizer, step=50, val_loss=3.0, config=config, log_dir=tmpdir)
            ckpt_path = os.path.join(tmpdir, "model_00050.pt")
            model2 = GPT(config)
            model2, _, _, _ = load_checkpoint(ckpt_path, model2)
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), model2.named_parameters()
            ):
                assert torch.allclose(p1, p2), f"Mismatch on parameter {n1}"

    def test_keep_last_n(self, model_and_optimizer):
        model, optimizer, config = model_and_optimizer
        with tempfile.TemporaryDirectory() as tmpdir:
            for step in [100, 200, 300, 400]:
                save_checkpoint(
                    model, optimizer, step=step, val_loss=2.5,
                    config=config, log_dir=tmpdir, keep_last_n=2,
                )
            pt_files = [f for f in os.listdir(tmpdir) if f.endswith(".pt")]
            assert len(pt_files) == 2
            # Should keep the two most recent
            assert any("00300" in f for f in pt_files)
            assert any("00400" in f for f in pt_files)

    def test_load_without_optimizer(self, model_and_optimizer):
        model, optimizer, config = model_and_optimizer
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model, optimizer, step=100, val_loss=2.5, config=config, log_dir=tmpdir)
            ckpt_path = os.path.join(tmpdir, "model_00100.pt")
            model2 = GPT(config)
            model2, opt_out, step, val_loss = load_checkpoint(ckpt_path, model2, optimizer=None)
            assert opt_out is None
            assert step == 100


class TestTrainingLogger:
    def test_log_train(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(log_dir=tmpdir)
            logger.log_train(step=0, loss=4.5, lr=6e-4, norm=1.0, dt=0.5, tokens_per_sec=100000)
            logger.close()

            log_path = os.path.join(tmpdir, "log.txt")
            assert os.path.exists(log_path)
            with open(log_path) as f:
                content = f.read()
            assert "train" in content

    def test_log_val(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(log_dir=tmpdir)
            logger.log_val(step=0, val_loss=5.0)
            logger.close()

            log_path = os.path.join(tmpdir, "log.txt")
            with open(log_path) as f:
                content = f.read()
            assert "val" in content

    def test_log_hellaswag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(log_dir=tmpdir)
            logger.log_hellaswag(step=250, accuracy=0.295)
            logger.close()

            log_path = os.path.join(tmpdir, "log.txt")
            with open(log_path) as f:
                content = f.read()
            assert "hella" in content

    def test_log_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(log_dir=tmpdir)
            logger.log_generation(step=250, texts=["Hello world", "Test output"])
            logger.close()

            log_path = os.path.join(tmpdir, "log.txt")
            with open(log_path) as f:
                content = f.read()
            assert "gen_0" in content

    def test_json_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(log_dir=tmpdir)
            logger.log_train(step=0, loss=4.5, lr=6e-4, norm=1.0, dt=0.5, tokens_per_sec=100000)
            logger.log_val(step=0, val_loss=5.0)
            logger.close()

            jsonl_path = os.path.join(tmpdir, "metrics.jsonl")
            assert os.path.exists(jsonl_path)
            with open(jsonl_path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            data = json.loads(lines[0])
            assert data["step"] == 0
            assert "loss" in data
            assert "timestamp" in data

    def test_creates_log_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "subdir", "logs")
            logger = TrainingLogger(log_dir=log_dir)
            logger.close()
            assert os.path.isdir(log_dir)
