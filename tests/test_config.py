"""Tests for the configuration system."""

import os
import tempfile

import pytest
import yaml

from nanogpt.config import TrainConfig, load_config


class TestTrainConfig:
    def test_defaults(self):
        config = TrainConfig()
        assert config.block_size == 1024
        assert config.n_layer == 12
        assert config.learning_rate == 6e-4
        assert config.dtype == "bfloat16"

    def test_validate_valid(self):
        config = TrainConfig()
        config.validate()  # should not raise

    def test_validate_bad_embd_head_ratio(self):
        config = TrainConfig(n_embd=100, n_head=3)
        with pytest.raises(AssertionError, match="n_embd.*must be divisible"):
            config.validate()

    def test_validate_bad_kv_head(self):
        config = TrainConfig(n_head=12, n_kv_head=5)
        with pytest.raises(AssertionError, match="n_head.*must be divisible by n_kv_head"):
            config.validate()

    def test_validate_bad_dtype(self):
        config = TrainConfig(dtype="float8")
        with pytest.raises(AssertionError, match="dtype must be"):
            config.validate()

    def test_validate_bad_lr_schedule(self):
        config = TrainConfig(lr_schedule="exponential")
        with pytest.raises(AssertionError, match="lr_schedule must be"):
            config.validate()

    def test_validate_bad_dropout(self):
        config = TrainConfig(dropout=1.0)
        with pytest.raises(AssertionError, match="dropout"):
            config.validate()

    def test_to_dict(self):
        config = TrainConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["block_size"] == 1024
        assert "n_layer" in d


class TestLoadConfig:
    def test_load_defaults(self):
        config = load_config()
        assert config.block_size == 1024

    def test_load_from_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"n_layer": 6, "n_head": 6, "n_embd": 384}, f)
            f.flush()
            config = load_config(f.name)
        os.unlink(f.name)
        assert config.n_layer == 6
        assert config.n_head == 6
        assert config.n_embd == 384

    def test_yaml_unknown_key_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"learnin_rate": 1e-3}, f)
            f.flush()
            with pytest.raises(ValueError, match="Unknown config key"):
                load_config(f.name)
        os.unlink(f.name)

    def test_cli_overrides(self):
        config = load_config(cli_args=["--n_layer", "6", "--learning_rate", "1e-3"])
        assert config.n_layer == 6
        assert config.learning_rate == 1e-3

    def test_cli_overrides_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"n_layer": 6, "n_embd": 384, "n_head": 6}, f)
            f.flush()
            config = load_config(f.name, cli_args=["--n_layer", "24"])
        os.unlink(f.name)
        assert config.n_layer == 24  # CLI overrides YAML
        assert config.n_embd == 384  # YAML value preserved

    def test_nonexistent_yaml_uses_defaults(self):
        config = load_config("/nonexistent/path.yaml")
        assert config.block_size == 1024

    def test_bool_cli_override(self):
        config = load_config(cli_args=["--use_wandb", "true"])
        assert config.use_wandb is True

    def test_optional_int_cli_override(self):
        config = load_config(cli_args=["--n_kv_head", "4"])
        assert config.n_kv_head == 4
