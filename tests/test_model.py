"""Tests for the GPT model and its components."""

import pytest
import torch

from nanogpt.model import (
    GPT,
    GPTConfig,
    RMSNorm,
    CausalSelfAttention,
    MLP,
    SwiGLU,
    Block,
    apply_rope,
    precompute_rope_freqs,
)


class TestGPTConfig:
    def test_default_config(self):
        config = GPTConfig()
        assert config.block_size == 1024
        assert config.vocab_size == 50257
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_embd == 768

    def test_custom_config(self):
        config = GPTConfig(n_layer=6, n_head=6, n_embd=384)
        assert config.n_layer == 6
        assert config.n_head == 6
        assert config.n_embd == 384


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(768)
        x = torch.randn(2, 10, 768)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self):
        norm = RMSNorm(64)
        x = torch.randn(1, 5, 64) * 100
        out = norm(x)
        rms = out.float().pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.2)


class TestRoPE:
    def test_precompute_shape(self):
        cos, sin = precompute_rope_freqs(64, 1024)
        assert cos.shape == (1024, 32)
        assert sin.shape == (1024, 32)

    def test_apply_rope_shape(self):
        cos, sin = precompute_rope_freqs(64, 1024)
        x = torch.randn(2, 12, 128, 64)
        out = apply_rope(x, cos, sin)
        assert out.shape == x.shape

    def test_rope_deterministic(self):
        cos, sin = precompute_rope_freqs(64, 1024)
        x = torch.randn(1, 1, 1, 64)
        x_dup = x.clone()
        out1 = apply_rope(x, cos, sin)
        out2 = apply_rope(x_dup, cos, sin)
        assert torch.allclose(out1, out2)


class TestCausalSelfAttention:
    def test_mha_output_shape(self):
        config = GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=100, block_size=32)
        attn = CausalSelfAttention(config)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == (2, 16, 64)

    def test_gqa_output_shape(self):
        config = GPTConfig(
            n_layer=2, n_head=8, n_kv_head=2, n_embd=64,
            vocab_size=100, block_size=32,
            use_rope=True, use_swiglu=True, use_rmsnorm=True,
        )
        attn = CausalSelfAttention(config)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == (2, 16, 64)

    def test_rope_attention(self):
        config = GPTConfig(
            n_layer=2, n_head=4, n_embd=64, vocab_size=100,
            block_size=32, use_rope=True, use_swiglu=True, use_rmsnorm=True,
        )
        attn = CausalSelfAttention(config)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == (2, 16, 64)

    def test_standard_has_bias(self):
        """Standard (non-modern) architecture should have biases on projections."""
        config = GPTConfig(
            n_layer=2, n_head=4, n_embd=64, vocab_size=100, block_size=32,
            use_rope=False, use_swiglu=False, use_rmsnorm=False,
        )
        attn = CausalSelfAttention(config)
        assert attn.q_proj.bias is not None
        assert attn.k_proj.bias is not None
        assert attn.c_proj.bias is not None

    def test_modern_no_bias(self):
        """Modern architecture (RoPE/SwiGLU/RMSNorm) should not have biases."""
        config = GPTConfig(
            n_layer=2, n_head=4, n_embd=64, vocab_size=100, block_size=32,
            use_rope=True, use_swiglu=True, use_rmsnorm=True,
        )
        attn = CausalSelfAttention(config)
        assert attn.q_proj.bias is None
        assert attn.k_proj.bias is None
        assert attn.c_proj.bias is None


class TestMLP:
    def test_output_shape(self):
        config = GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=100, block_size=32)
        mlp = MLP(config)
        x = torch.randn(2, 16, 64)
        out = mlp(x)
        assert out.shape == (2, 16, 64)


class TestSwiGLU:
    def test_output_shape(self):
        config = GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=100, block_size=32)
        swiglu = SwiGLU(config)
        x = torch.randn(2, 16, 64)
        out = swiglu(x)
        assert out.shape == (2, 16, 64)


class TestBlock:
    def test_standard_block(self):
        config = GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=100, block_size=32)
        block = Block(config)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 16, 64)

    def test_modern_block(self):
        config = GPTConfig(
            n_layer=2, n_head=4, n_embd=64, vocab_size=100, block_size=32,
            use_rope=True, use_swiglu=True, use_rmsnorm=True,
        )
        block = Block(config)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 16, 64)


class TestGPT:
    @pytest.fixture
    def small_config(self):
        return GPTConfig(block_size=32, vocab_size=100, n_layer=2, n_head=4, n_embd=64)

    def test_forward_logits_only(self, small_config):
        torch.manual_seed(42)
        model = GPT(small_config)
        idx = torch.randint(0, 100, (2, 16))
        logits, loss = model(idx)
        assert logits.shape == (2, 16, 100)
        assert loss is None

    def test_forward_with_targets(self, small_config):
        torch.manual_seed(42)
        model = GPT(small_config)
        idx = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))
        logits, loss = model(idx, targets)
        assert logits.shape == (2, 16, 100)
        assert loss is not None
        assert loss.item() > 0

    def test_sequence_length_assertion(self, small_config):
        model = GPT(small_config)
        idx = torch.randint(0, 100, (1, 64))  # exceeds block_size=32
        with pytest.raises(AssertionError):
            model(idx)

    def test_weight_tying(self, small_config):
        model = GPT(small_config)
        assert model.transformer.wte.weight is model.lm_head.weight

    def test_modern_architecture(self):
        config = GPTConfig(
            block_size=32, vocab_size=100, n_layer=2, n_head=4,
            n_kv_head=2, n_embd=64,
            use_rope=True, use_swiglu=True, use_rmsnorm=True,
        )
        torch.manual_seed(42)
        model = GPT(config)
        idx = torch.randint(0, 100, (2, 16))
        logits, loss = model(idx)
        assert logits.shape == (2, 16, 100)

    def test_no_wpe_with_rope(self):
        config = GPTConfig(
            block_size=32, vocab_size=100, n_layer=2, n_head=4, n_embd=64,
            use_rope=True, use_swiglu=True, use_rmsnorm=True,
        )
        model = GPT(config)
        assert 'wpe' not in model.transformer

    def test_has_wpe_without_rope(self):
        config = GPTConfig(
            block_size=32, vocab_size=100, n_layer=2, n_head=4, n_embd=64,
            use_rope=False,
        )
        model = GPT(config)
        assert 'wpe' in model.transformer

    def test_gradient_checkpointing(self):
        config = GPTConfig(
            block_size=32, vocab_size=100, n_layer=2, n_head=4,
            n_embd=64, gradient_checkpointing=True,
        )
        torch.manual_seed(42)
        model = GPT(config)
        model.train()
        idx = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))
        logits, loss = model(idx, targets)
        loss.backward()
        assert loss.item() > 0

    def test_generate(self, small_config):
        torch.manual_seed(42)
        model = GPT(small_config)
        model.eval()
        idx = torch.randint(0, 100, (1, 5))
        output = model.generate(idx, max_new_tokens=10)
        assert output.shape == (1, 15)

    def test_generate_with_sampling_params(self, small_config):
        torch.manual_seed(42)
        model = GPT(small_config)
        model.eval()
        idx = torch.randint(0, 100, (1, 5))
        output = model.generate(
            idx, max_new_tokens=10, temperature=0.8,
            top_k=10, top_p=0.9, repetition_penalty=1.2,
        )
        assert output.shape == (1, 15)

    def test_generate_batch(self, small_config):
        torch.manual_seed(42)
        model = GPT(small_config)
        model.eval()
        idx = torch.randint(0, 100, (3, 5))
        output = model.generate(idx, max_new_tokens=10)
        assert output.shape == (3, 15)

    def test_count_parameters(self, small_config):
        model = GPT(small_config)
        count = model.count_parameters()
        assert count > 0
        assert isinstance(count, int)

    def test_configure_optimizers(self, small_config):
        model = GPT(small_config)
        optimizer = model.configure_optimizers(
            weight_decay=0.1, learning_rate=1e-4,
            device_type="cpu", master_process=False,
        )
        assert optimizer is not None
        assert len(optimizer.param_groups) == 2

    def test_loss_decreases(self, small_config):
        """Verify that a few optimization steps reduce the loss (sanity check)."""
        torch.manual_seed(42)
        model = GPT(small_config)
        model.train()
        optimizer = model.configure_optimizers(0.1, 1e-3, "cpu", master_process=False)
        idx = torch.randint(0, 100, (4, 16))
        targets = torch.randint(0, 100, (4, 16))

        _, loss0 = model(idx, targets)
        for _ in range(20):
            optimizer.zero_grad()
            _, loss = model(idx, targets)
            loss.backward()
            optimizer.step()

        assert loss.item() < loss0.item()
