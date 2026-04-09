"""
GPT model with modern architectural improvements.

Supports:
- Standard Multi-Head Attention (MHA) or Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE) or learned positional embeddings
- SwiGLU or GELU activation
- RMSNorm or LayerNorm
- Gradient checkpointing
- Weight tying
"""

import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: Optional[int] = None  # None = MHA, else GQA
    n_embd: int = 768
    use_rope: bool = False
    use_swiglu: bool = False
    use_rmsnorm: bool = False
    rope_theta: float = 10000.0
    dropout: float = 0.0
    gradient_checkpointing: bool = False


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute the cos and sin frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos_freqs = freqs.cos()
    sin_freqs = freqs.sin()
    return cos_freqs, sin_freqs


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor."""
    # x: (B, n_head, T, head_dim)
    B, H, T, D = x.shape
    # Reshape x into pairs
    x_reshape = x.float().reshape(B, H, T, D // 2, 2)
    x0 = x_reshape[..., 0]
    x1 = x_reshape[..., 1]
    # Apply rotation
    cos_t = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
    sin_t = sin[:T].unsqueeze(0).unsqueeze(0)
    out0 = x0 * cos_t - x1 * sin_t
    out1 = x0 * sin_t + x1 * cos_t
    # Interleave back
    out = torch.stack([out0, out1], dim=-1).reshape(B, H, T, D)
    return out.type_as(x)


class CausalSelfAttention(nn.Module):
    """Multi-Head or Grouped Query Attention with optional RoPE."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.use_rope = config.use_rope
        self.dropout = config.dropout

        assert config.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.n_rep = config.n_head // self.n_kv_head

        # Use bias for GPT-2 compatibility; modern architectures (RoPE etc.) skip bias
        use_bias = not (config.use_rope or config.use_swiglu or config.use_rmsnorm)

        # Projections
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=use_bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=use_bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=use_bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=use_bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # RoPE
        if self.use_rope:
            cos_freqs, sin_freqs = precompute_rope_freqs(
                self.head_dim, config.block_size, config.rope_theta
            )
            self.register_buffer("rope_cos", cos_freqs, persistent=False)
            self.register_buffer("rope_sin", sin_freqs, persistent=False)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.resid_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match Q heads for GQA."""
        if self.n_rep == 1:
            return x
        B, H, T, D = x.shape
        x = x.unsqueeze(2).expand(B, H, self.n_rep, T, D)
        return x.reshape(B, H * self.n_rep, T, D)

    def forward(self, x):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

        # GQA: repeat KV heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Flash attention
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class SwiGLU(nn.Module):
    """SwiGLU activation (Shazeer, 2020). Used in LLaMA/Mistral."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # SwiGLU uses 2/3 * 4 * n_embd for hidden dim to keep param count similar
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        # Round to nearest multiple of 256 for hardware efficiency
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MLP(nn.Module):
    """Standard GELU MLP."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with configurable norm and activation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        norm_cls = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.ln_1 = norm_cls(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = norm_cls(config.n_embd)
        self.mlp = SwiGLU(config) if config.use_swiglu else MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    GPT Language Model.

    Supports both classic GPT-2 architecture and modern improvements
    (RoPE, GQA, SwiGLU, RMSNorm).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))

        # Positional embeddings: learned (GPT-2 style) if not using RoPE
        if not config.use_rope:
            self.transformer['wpe'] = nn.Embedding(config.block_size, config.n_embd)

        # Final norm
        norm_cls = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.transformer['ln_f'] = norm_cls(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Gradient checkpointing
        self._use_gradient_checkpointing = config.gradient_checkpointing

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        )

        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)

        if not self.config.use_rope:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.transformer.wpe(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        # Transformer blocks
        for block in self.transformer.h:
            if self._use_gradient_checkpointing and self.training:
                x = torch_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from HuggingFace."""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        # Use standard architecture for pretrained GPT-2
        config_args['use_rope'] = False
        config_args['use_swiglu'] = False
        config_args['use_rmsnorm'] = False

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # Map HF GPT-2 keys to our keys.
        # Key differences:
        # 1. HF fuses Q/K/V into c_attn; we split into q_proj/k_proj/v_proj
        # 2. HF uses Conv1D (transposed weights); we use nn.Linear
        # 3. Biases transfer directly (no transpose needed)
        transposed = ['attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        with torch.no_grad():
            for k_hf in sd_keys_hf:
                if 'attn.c_attn' in k_hf:
                    # Split fused QKV into separate Q, K, V
                    is_weight = k_hf.endswith('.weight')
                    param = sd_hf[k_hf]
                    if is_weight:
                        param = param.t()  # Conv1D -> Linear
                    n_embd = param.shape[0] // 3
                    q, k_param, v = param.split(n_embd, dim=0)
                    suffix = 'weight' if is_weight else 'bias'
                    # Derive prefix: "transformer.h.0." from "transformer.h.0.attn.c_attn.weight"
                    prefix = k_hf.rsplit('attn.c_attn.', 1)[0]
                    for name, tensor in [('q_proj', q), ('k_proj', k_param), ('v_proj', v)]:
                        key = f"{prefix}attn.{name}.{suffix}"
                        assert key in sd, f"Expected key {key} in model state dict"
                        sd[key].copy_(tensor)
                elif any(k_hf.endswith(w) for w in transposed):
                    # Conv1D weights need transpose
                    assert k_hf in sd, f"Expected key {k_hf} in model state dict"
                    sd[k_hf].copy_(sd_hf[k_hf].t())
                else:
                    # Direct copy (biases, embeddings, layernorms)
                    assert k_hf in sd, f"Expected key {k_hf} in model state dict"
                    sd[k_hf].copy_(sd_hf[k_hf])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=True):
        """Configure AdamW optimizer with weight decay groups."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate,
            betas=(0.9, 0.95), eps=1e-8, fused=use_fused,
        )
        return optimizer

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50,
                 top_p=1.0, repetition_penalty=1.0):
        """
        Generate tokens autoregressively with configurable sampling.

        Args:
            idx: (B, T) tensor of token indices
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling (0 = disabled)
            top_p: nucleus sampling threshold (1.0 = disabled)
            repetition_penalty: penalty for repeated tokens (1.0 = disabled)
        """
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Repetition penalty (vectorized)
            if repetition_penalty != 1.0:
                # Gather logits for tokens that appeared in the sequence
                score = torch.gather(logits, 1, idx)
                # Apply penalty: divide positive logits, multiply negative ones
                score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
                logits.scatter_(1, idx, score)

            # Temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx
