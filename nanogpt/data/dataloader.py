"""
Modular data loading for nanoGPT training.

Loads pre-tokenized .npy shards and serves batches for distributed training.
Supports multi-process data loading with shard-level parallelism.
"""

import os

import numpy as np
import torch


def load_tokens(filename: str) -> torch.Tensor:
    """Load a .npy token shard and return as a torch.long tensor.

    Args:
        filename: Path to the .npy file containing token IDs.

    Returns:
        1-D torch.long tensor of token IDs.
    """
    npt = np.load(filename)
    npt = npt.astype(np.int32)  # avoid overflow when converting to torch
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    """Lightweight data loader that streams pre-tokenized .npy shards.

    Designed for distributed training: each process reads from the same shards
    but at offset positions so that every token is seen exactly once across
    all processes per epoch.

    Args:
        B: Micro-batch size (number of sequences per batch).
        T: Sequence length (number of tokens per sequence).
        process_rank: Rank of the current process (0-indexed).
        num_processes: Total number of data-parallel processes.
        split: One of ``"train"`` or ``"val"``.
        data_root: Directory containing the ``.npy`` shard files.
        master_process: Whether this process should print status messages.
        shuffle_shards: If ``True``, shuffle shard order on every ``reset()``
            for better training randomness.
    """

    def __init__(
        self,
        B: int,
        T: int,
        process_rank: int,
        num_processes: int,
        split: str,
        data_root: str = "edu_fineweb10B",
        master_process: bool = True,
        shuffle_shards: bool = False,
    ):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.master_process = master_process
        self.shuffle_shards = shuffle_shards
        assert split in {"train", "val"}, f"split must be 'train' or 'val', got '{split}'"

        # Discover shard files for the requested split
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if self.master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        """Reset the loader to the beginning of the dataset.

        If ``shuffle_shards`` is enabled, the shard order is randomised
        (useful at the start of each training epoch).
        """
        if self.shuffle_shards:
            import random
            random.shuffle(self.shards)
        # State: start at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """Return the next ``(x, y)`` batch of token tensors.

        ``x`` has shape ``(B, T)`` and ``y`` has shape ``(B, T)`` where ``y``
        is the next-token target (shifted by one position).

        Returns:
            Tuple of ``(x, y)`` tensors.
        """
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets
        # Advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # If loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
