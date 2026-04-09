"""
Standalone generation / inference script for nanoGPT.

Loads a trained checkpoint and generates text from a prompt.

Usage::

    python -m nanogpt.generate --checkpoint log/model_19073.pt
    python -m nanogpt.generate --checkpoint log/model_19073.pt --prompt "Once upon a time"
    python -m nanogpt.generate --checkpoint log/model_19073.pt --num_samples 5 --temperature 0.9
"""

import argparse

import torch
import tiktoken

from nanogpt.model import GPT, GPTConfig


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained nanoGPT checkpoint."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a .pt checkpoint file.",
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello, I'm a language model,",
        help="Text prompt to condition generation on.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=200,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (higher = more random).",
    )
    parser.add_argument(
        "--top_k", type=int, default=50,
        help="Top-k sampling (0 = disabled).",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Nucleus (top-p) sampling threshold (1.0 = disabled).",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.1,
        help="Repetition penalty (1.0 = disabled).",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help='Device to use: "auto", "cuda", "mps", or "cpu".',
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1. Load checkpoint
    # -------------------------------------------------------------------------
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # -------------------------------------------------------------------------
    # 2. Create model from checkpoint config
    # -------------------------------------------------------------------------
    model_config = checkpoint["config"]
    if not isinstance(model_config, GPTConfig):
        # If config was saved as a dataclass, it should already be a GPTConfig.
        # If it was saved as a dict, reconstruct it.
        if isinstance(model_config, dict):
            model_config = GPTConfig(**model_config)
        else:
            # Assume it's a dataclass-like object with the right fields
            model_config = GPTConfig(
                block_size=getattr(model_config, "block_size", 1024),
                vocab_size=getattr(model_config, "vocab_size", 50257),
                n_layer=getattr(model_config, "n_layer", 12),
                n_head=getattr(model_config, "n_head", 12),
                n_kv_head=getattr(model_config, "n_kv_head", None),
                n_embd=getattr(model_config, "n_embd", 768),
                use_rope=getattr(model_config, "use_rope", False),
                use_swiglu=getattr(model_config, "use_swiglu", False),
                use_rmsnorm=getattr(model_config, "use_rmsnorm", False),
                rope_theta=getattr(model_config, "rope_theta", 10000.0),
                dropout=getattr(model_config, "dropout", 0.0),
                gradient_checkpointing=False,  # not needed for inference
            )

    model = GPT(model_config)

    # -------------------------------------------------------------------------
    # 3. Load state dict
    # -------------------------------------------------------------------------
    model.load_state_dict(checkpoint["model"])
    print(f"Model loaded: {model.count_parameters():,} parameters")

    # -------------------------------------------------------------------------
    # 4. Set device, move model, set eval mode
    # -------------------------------------------------------------------------
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # -------------------------------------------------------------------------
    # 5. Encode prompt
    # -------------------------------------------------------------------------
    enc = tiktoken.get_encoding("gpt2")
    prompt_tokens = enc.encode(args.prompt)
    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0).repeat(args.num_samples, 1)

    # -------------------------------------------------------------------------
    # 6. Generate
    # -------------------------------------------------------------------------
    print(f"\nPrompt: {args.prompt}")
    print(f"Generating {args.num_samples} sample(s) with max_tokens={args.max_tokens}, "
          f"temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, "
          f"repetition_penalty={args.repetition_penalty}")
    print("-" * 60)

    with torch.no_grad():
        generated = model.generate(
            tokens,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

    # -------------------------------------------------------------------------
    # 7. Decode and print each sample
    # -------------------------------------------------------------------------
    for i in range(args.num_samples):
        decoded = enc.decode(generated[i].tolist())
        print(f"\n--- Sample {i + 1} ---")
        print(decoded)

    print("-" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
