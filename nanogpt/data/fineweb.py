"""
FineWeb-Edu dataset downloader and tokenizer.

Downloads the FineWeb-Edu dataset from HuggingFace and tokenizes it into
.npy shards suitable for streaming with :class:`DataLoaderLite`.

Usage::

    python -m nanogpt.data.fineweb

Will save shards to the local directory ``edu_fineweb10B``.
"""

import os
import multiprocessing as mp

import numpy as np
import tiktoken
from tqdm import tqdm

# GPT-2 tokenizer (lazily initialised at module level for multiprocessing)
_enc = tiktoken.get_encoding("gpt2")
_eot = _enc._special_tokens["<|endoftext|>"]  # end-of-text token


def tokenize(doc: dict) -> np.ndarray:
    """Tokenize a single document and return a numpy array of uint16 tokens.

    Each document is prefixed with the ``<|endoftext|>`` special token so that
    document boundaries are preserved.

    Args:
        doc: Dictionary with at least a ``"text"`` key.

    Returns:
        1-D ``np.uint16`` array of token IDs.
    """
    tokens = [_eot]  # delimiter between documents
    tokens.extend(_enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), (
        "token dictionary too large for uint16"
    )
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename: str, tokens_np: np.ndarray) -> None:
    """Write a numpy token array to disk as a ``.npy`` file.

    Args:
        filename: Output path (without ``.npy`` extension — ``np.save`` adds it).
        tokens_np: 1-D numpy array of token IDs.
    """
    np.save(filename, tokens_np)


def download_and_tokenize(
    local_dir: str = "edu_fineweb10B",
    remote_name: str = "sample-10BT",
    shard_size: int = int(1e8),
    num_workers: int = 0,
) -> None:
    """Download the FineWeb-Edu dataset and tokenize it into .npy shards.

    Shards are written to ``local_dir`` with the naming convention
    ``edufineweb_{split}_{index:06d}.npy``.  The first shard (index 0)
    is assigned to the ``val`` split; all subsequent shards are ``train``.

    Args:
        local_dir: Local directory to store the tokenized shards.
        remote_name: HuggingFace dataset configuration name (e.g. ``"sample-10BT"``).
        shard_size: Number of tokens per shard.
        num_workers: Number of multiprocessing workers.  ``0`` means use
            ``os.cpu_count() // 2`` (at least 1).
    """
    from datasets import load_dataset  # pip install datasets

    # Create the local directory if it doesn't exist yet
    data_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), local_dir)
    if not os.path.isabs(local_dir):
        # Use cwd-relative path when local_dir is relative (matches original behaviour)
        data_cache_dir = os.path.abspath(local_dir)
    os.makedirs(data_cache_dir, exist_ok=True)

    # Download the dataset
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

    # Resolve worker count
    nprocs = num_workers if num_workers > 0 else max(1, os.cpu_count() // 2)

    # Tokenize all documents and write output shards
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # Pre-allocate buffer to hold the current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, fw, chunksize=16):
            # Is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # Simply append tokens to current shard
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # Update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                    )
                progress_bar.update(len(tokens))
            else:
                # Write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    data_cache_dir, f"edufineweb_{split}_{shard_index:06d}"
                )
                # Split the document into whatever fits in this shard; remainder goes to next
                remainder = shard_size - token_count
                if progress_bar is not None:
                    progress_bar.update(remainder)
                all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # Populate the next shard with the leftovers of the current doc
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # Write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                data_cache_dir, f"edufineweb_{split}_{shard_index:06d}"
            )
            write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    download_and_tokenize()
