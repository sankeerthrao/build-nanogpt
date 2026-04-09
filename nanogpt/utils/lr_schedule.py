"""
Learning rate schedule functions for nanoGPT training.

Supports cosine, linear, and warmup-stable-decay schedules.  All schedules
use a linear warmup phase and return ``min_lr`` after ``max_steps``.
"""

import math


def get_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
    schedule: str = "cosine",
    stable_fraction: float = 0.8,
) -> float:
    """Compute the learning rate for a given training step.

    Args:
        step: Current training step (0-indexed).
        max_lr: Peak learning rate (reached at the end of warmup).
        min_lr: Minimum learning rate (floor value after decay).
        warmup_steps: Number of linear warmup steps.
        max_steps: Total number of training steps.
        schedule: Schedule type — one of:
            - ``"cosine"``: Linear warmup then cosine decay to ``min_lr``.
            - ``"linear"``: Linear warmup then linear decay to ``min_lr``.
            - ``"warmup_stable_decay"``: Linear warmup, hold at ``max_lr``
              for ``stable_fraction`` of training, then cosine decay.
        stable_fraction: Fraction of total steps spent at ``max_lr`` in the
            ``"warmup_stable_decay"`` schedule.  Ignored for other schedules.

    Returns:
        Learning rate for the given step.

    Raises:
        ValueError: If *schedule* is not one of the supported values.
    """
    # After max_steps, always return min_lr
    if step > max_steps:
        return min_lr

    # Phase 1: linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if schedule == "cosine":
        # Phase 2: cosine decay from max_lr to min_lr
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    elif schedule == "linear":
        # Phase 2: linear decay from max_lr to min_lr
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        return max_lr - decay_ratio * (max_lr - min_lr)

    elif schedule == "warmup_stable_decay":
        # Phase 2: stable at max_lr for stable_fraction of total steps
        stable_steps = int(max_steps * stable_fraction)
        if step < stable_steps:
            return max_lr
        # Phase 3: cosine decay from max_lr to min_lr over remaining steps
        decay_steps = max_steps - stable_steps
        if decay_steps == 0:
            return min_lr
        decay_ratio = (step - stable_steps) / decay_steps
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    else:
        raise ValueError(
            f"Unknown schedule '{schedule}'. "
            f"Supported: 'cosine', 'linear', 'warmup_stable_decay'"
        )
