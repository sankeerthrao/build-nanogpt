"""
Main training script for nanoGPT.

A clean, modular refactoring of the original train_gpt2.py.

Usage::

    # Simple launch:
    python -m nanogpt.train

    # With config file:
    python -m nanogpt.train --config configs/gpt2_124m.yaml

    # With CLI overrides:
    python -m nanogpt.train --config configs/gpt2_124m.yaml --learning_rate 3e-4

    # DDP launch for e.g. 8 GPUs:
    torchrun --standalone --nproc_per_node=8 -m nanogpt.train

    # Resume from checkpoint:
    python -m nanogpt.train --config configs/gpt2_124m.yaml --resume log/model_05000.pt
"""

import os
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken

from nanogpt.model import GPT, GPTConfig
from nanogpt.config import TrainConfig, load_config
from nanogpt.data import DataLoaderLite
from nanogpt.eval.hellaswag import iterate_examples, render_example, get_most_likely_row
from nanogpt.utils import TrainingLogger, get_lr, save_checkpoint, load_checkpoint


# Dtype mapping from config string to torch dtype
_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def main():
    # -------------------------------------------------------------------------
    # 1. Parse config from optional YAML file + CLI overrides
    # -------------------------------------------------------------------------
    config_path = None
    resume_path = None
    cli_args = list(sys.argv[1:])

    # Extract --config and --resume before passing rest to config loader
    filtered_args = []
    i = 0
    while i < len(cli_args):
        if cli_args[i] == "--config" and i + 1 < len(cli_args):
            config_path = cli_args[i + 1]
            i += 2
        elif cli_args[i] == "--resume" and i + 1 < len(cli_args):
            resume_path = cli_args[i + 1]
            i += 2
        elif cli_args[i].startswith("--config="):
            config_path = cli_args[i].split("=", 1)[1]
            i += 1
        elif cli_args[i].startswith("--resume="):
            resume_path = cli_args[i].split("=", 1)[1]
            i += 1
        else:
            filtered_args.append(cli_args[i])
            i += 1

    # Also support positional first arg as config path
    if config_path is None and filtered_args and not filtered_args[0].startswith("--"):
        config_path = filtered_args.pop(0)

    cfg = load_config(config_path=config_path, cli_args=filtered_args)

    # -------------------------------------------------------------------------
    # 2. Set up DDP if RANK env var is set
    # -------------------------------------------------------------------------
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # Auto-detect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        if master_process:
            print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # -------------------------------------------------------------------------
    # 3. Set seeds
    # -------------------------------------------------------------------------
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    # -------------------------------------------------------------------------
    # 4. Compute grad_accum_steps
    # -------------------------------------------------------------------------
    B = cfg.micro_batch_size
    T = cfg.block_size
    assert cfg.total_batch_size % (B * T * ddp_world_size) == 0, (
        f"total_batch_size ({cfg.total_batch_size}) must be divisible by "
        f"B * T * world_size ({B} * {T} * {ddp_world_size} = {B * T * ddp_world_size})"
    )
    grad_accum_steps = cfg.total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {cfg.total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # -------------------------------------------------------------------------
    # 5. Create data loaders
    # -------------------------------------------------------------------------
    train_loader = DataLoaderLite(
        B=B, T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="train",
        data_root=cfg.data_root,
        master_process=master_process,
    )
    val_loader = DataLoaderLite(
        B=B, T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="val",
        data_root=cfg.data_root,
        master_process=master_process,
    )

    torch.set_float32_matmul_precision("high")

    # -------------------------------------------------------------------------
    # 6. Create model
    # -------------------------------------------------------------------------
    model_config = GPTConfig(
        block_size=cfg.block_size,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_kv_head,
        n_embd=cfg.n_embd,
        use_rope=cfg.use_rope,
        use_swiglu=cfg.use_swiglu,
        use_rmsnorm=cfg.use_rmsnorm,
        rope_theta=cfg.rope_theta,
        dropout=cfg.dropout,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )
    model = GPT(model_config)
    model.to(device)
    if master_process:
        print(f"model parameters: {model.count_parameters():,}")

    # -------------------------------------------------------------------------
    # 7. Optionally torch.compile
    # -------------------------------------------------------------------------
    use_compile = cfg.use_compile
    if use_compile:
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # 8. Wrap in DDP if needed
    # -------------------------------------------------------------------------
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model  # always the unwrapped model

    # -------------------------------------------------------------------------
    # 9. Create optimizer
    # -------------------------------------------------------------------------
    optimizer = raw_model.configure_optimizers(
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.learning_rate,
        device_type=device_type,
        master_process=master_process,
    )

    # -------------------------------------------------------------------------
    # 10. Create logger
    # -------------------------------------------------------------------------
    logger = None
    if master_process:
        logger = TrainingLogger(
            log_dir=cfg.log_dir,
            use_wandb=cfg.use_wandb,
            use_tensorboard=cfg.use_tensorboard,
            wandb_project=cfg.wandb_project,
            wandb_run_name=cfg.wandb_run_name,
            config=cfg.to_dict(),
        )

    # -------------------------------------------------------------------------
    # 11. Optionally resume from checkpoint
    # -------------------------------------------------------------------------
    start_step = 0
    if resume_path is not None:
        if master_process:
            print(f"resuming from checkpoint: {resume_path}")
        raw_model, optimizer, start_step, _ = load_checkpoint(
            resume_path, raw_model, optimizer
        )
        start_step += 1  # resume from the next step
        if master_process:
            print(f"resumed at step {start_step}")

    # -------------------------------------------------------------------------
    # 12. Prepare for generation during training
    # -------------------------------------------------------------------------
    enc = tiktoken.get_encoding("gpt2")
    torch_dtype = _DTYPE_MAP.get(cfg.dtype, torch.bfloat16)

    # -------------------------------------------------------------------------
    # 13. Training loop
    # -------------------------------------------------------------------------
    for step in range(start_step, cfg.max_steps):
        t0 = time.time()
        last_step = (step == cfg.max_steps - 1)

        # ----- Validation -----
        if step % cfg.val_interval == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                for _ in range(cfg.val_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch_dtype):
                        logits, loss = model(x, y)
                    loss = loss / cfg.val_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                logger.log_val(step, val_loss_accum.item())
                if step > 0 and (step % cfg.checkpoint_interval == 0 or last_step):
                    checkpoint_path = save_checkpoint(
                        model=raw_model,
                        optimizer=optimizer if cfg.save_optimizer else None,
                        step=step,
                        val_loss=val_loss_accum.item(),
                        config=raw_model.config,
                        log_dir=cfg.log_dir,
                        keep_last_n=cfg.keep_last_n_checkpoints,
                    )
                    print(f"saved checkpoint to {checkpoint_path}")

        # ----- HellaSwag evaluation -----
        if (step % cfg.val_interval == 0 or last_step) and cfg.hellaswag_eval and (not use_compile):
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                # Only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                # Render the example into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # Get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch_dtype):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # Reduce the stats across all processes
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total if num_total > 0 else 0.0
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                logger.log_hellaswag(step, acc_norm)

        # ----- Generation -----
        if ((step > 0 and step % cfg.generate_interval == 0) or last_step) and (not use_compile):
            model.eval()
            num_return_sequences = cfg.generate_samples
            max_length = cfg.generate_max_length
            prompt_tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(prompt_tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            tokens = tokens.to(device)
            max_new_tokens = max_length - tokens.size(1)
            if max_new_tokens > 0:
                with torch.autocast(device_type=device_type, dtype=torch_dtype):
                    xgen = raw_model.generate(
                        tokens,
                        max_new_tokens=max_new_tokens,
                        temperature=cfg.temperature,
                        top_k=cfg.top_k,
                        top_p=cfg.top_p,
                        repetition_penalty=cfg.repetition_penalty,
                    )
            else:
                xgen = tokens
            # Print and log the generated text
            generated_texts = []
            for i in range(num_return_sequences):
                decoded = enc.decode(xgen[i, :max_length].tolist())
                generated_texts.append(decoded)
                if master_process:
                    print(f"rank {ddp_rank} sample {i}: {decoded}")
            if master_process:
                logger.log_generation(step, generated_texts)

        # ----- Training step -----
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch_dtype):
                logits, loss = model(x, y)
            # Scale loss for gradient accumulation (SUM -> MEAN)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # Learning rate scheduling
        lr = get_lr(
            step=step,
            max_lr=cfg.learning_rate,
            min_lr=cfg.learning_rate * cfg.min_lr_ratio,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps,
            schedule=cfg.lr_schedule,
            stable_fraction=cfg.stable_fraction,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = B * T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(
                f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | "
                f"norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )
            if step % cfg.log_interval == 0:
                logger.log_train(
                    step=step,
                    loss=loss_accum.item(),
                    lr=lr,
                    norm=norm.item() if isinstance(norm, torch.Tensor) else norm,
                    dt=dt,
                    tokens_per_sec=tokens_per_sec,
                )

    # -------------------------------------------------------------------------
    # 14. Cleanup
    # -------------------------------------------------------------------------
    if ddp:
        destroy_process_group()
    if logger is not None:
        logger.close()


if __name__ == "__main__":
    main()
