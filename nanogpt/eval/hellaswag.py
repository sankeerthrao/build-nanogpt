"""
HellaSwag evaluation for language models.

Downloads and evaluates the HellaSwag benchmark using completion-style scoring.
See https://github.com/rowanz/hellaswag for dataset details.

Example HellaSwag json item::

    {
        "ind": 24,
        "activity_label": "Roof shingle removal",
        "ctx_a": "A man is sitting on a roof.",
        "ctx_b": "he",
        "ctx": "A man is sitting on a roof. he",
        "split": "val",
        "split_type": "indomain",
        "label": 3,
        "endings": [
            "is using wrap to wrap a pair of skis.",
            "is ripping level tiles off.",
            "is holding a rubik's cube.",
            "starts pulling up roofing on a roof."
        ],
        "source_id": "activitynet~v_-JhWjGDPHMY"
    }

Benchmark numbers (completion style):

- gpt2 (124M):  acc_norm ~29.55%
- gpt2-xl (1558M): acc_norm ~48.93%

The validation set has 10,042 examples.
"""

import os
import json

import requests
import tiktoken
from tqdm import tqdm
import torch
from torch.nn import functional as F

# ---------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

_hellaswag_urls = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")


def download_file(url: str, fname: str, chunk_size: int = 1024) -> None:
    """Download a file from *url* and save it to *fname* with a progress bar.

    Args:
        url: Source URL.
        fname: Destination file path.
        chunk_size: Download chunk size in bytes.
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download(split: str) -> None:
    """Download the HellaSwag data for a given split if not already cached.

    Args:
        split: One of ``"train"``, ``"val"``, or ``"test"``.
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = _hellaswag_urls[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)


def render_example(example: dict):
    """Render a HellaSwag example into tensors for evaluation.

    Given an example dictionary, produces padded token and mask tensors
    for the four candidate completions.

    Args:
        example: A single HellaSwag example (parsed JSON object).

    Returns:
        Tuple of ``(data, tokens, mask, label)`` where:
        - *data*: Dict with ``label``, ``ctx_tokens``, and ``ending_tokens``.
        - *tokens*: ``(4, max_len)`` long tensor of token IDs.
        - *mask*: ``(4, max_len)`` long tensor (1 in the completion region).
        - *label*: Integer index of the correct completion.
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # Data needed to reproduce this eval on the C side
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # Gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end)  # prepend " " for GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # Collate — row lengths may differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(tok_row)] = torch.tensor(tok_row)
        mask[i, : len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_examples(split: str):
    """Yield HellaSwag examples one at a time for the given split.

    Downloads the data on first call if it is not already cached.

    Args:
        split: One of ``"train"``, ``"val"``, or ``"test"``.

    Yields:
        Parsed JSON objects (dicts).
    """
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


def get_most_likely_row(tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor) -> int:
    """Return the index of the completion with the lowest average loss.

    This is the core scoring function for HellaSwag evaluation.  It computes
    the cross-entropy loss over only the completion region (where ``mask == 1``)
    and returns the row (completion index) with the lowest average loss.

    Args:
        tokens: ``(num_completions, seq_len)`` token IDs.
        mask: ``(num_completions, seq_len)`` binary mask (1 = completion region).
        logits: ``(num_completions, seq_len, vocab_size)`` model output logits.

    Returns:
        Integer index of the most likely completion (0-indexed).
    """
    # Evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # Get the average loss just for the completion region (where mask == 1)
    shift_mask = (mask[..., 1:]).contiguous()  # shift mask to align with loss
    masked_shift_losses = shift_losses * shift_mask
    # Sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # The completion with the lowest loss is the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


@torch.no_grad()
def evaluate(model, device: str, split: str = "val"):
    """Evaluate a model on the HellaSwag benchmark.

    Unlike the standalone script version, this accepts a model object directly
    instead of loading from HuggingFace, making it suitable for evaluating
    models during training.

    Args:
        model: A language model with a ``forward(tokens) -> (logits, loss)``
            interface (the nanoGPT convention).
        device: Device string (e.g. ``"cuda"`` or ``"cpu"``).
        split: HellaSwag split to evaluate on (default ``"val"``).

    Returns:
        Dict with ``"num_correct"``, ``"num_total"``, and ``"accuracy"`` keys.
    """
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    model.eval()
    num_correct_norm = 0
    num_total = 0

    for example in iterate_examples(split):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Get the logits from the model
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(tokens)
        pred_norm = get_most_likely_row(tokens, mask, logits)

        num_total += 1
        num_correct_norm += int(pred_norm == label)

        if num_total % 1000 == 0:
            print(
                f"HellaSwag eval: {num_total} examples, "
                f"acc_norm: {num_correct_norm}/{num_total}"
                f"={num_correct_norm / num_total:.4f}"
            )

    accuracy = num_correct_norm / num_total if num_total > 0 else 0.0
    print(
        f"HellaSwag final: {num_correct_norm}/{num_total}={accuracy:.4f}"
    )
    return {
        "num_correct": num_correct_norm,
        "num_total": num_total,
        "accuracy": accuracy,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate HellaSwag")
    parser.add_argument(
        "-m", "--model_type", type=str, default="gpt2",
        help="HuggingFace model type to evaluate (e.g. gpt2, gpt2-xl)",
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda",
        help="Device to use for evaluation",
    )
    args = parser.parse_args()

    # Standalone evaluation using HuggingFace model
    from transformers import GPT2LMHeadModel

    torch.set_float32_matmul_precision("high")
    hf_model = GPT2LMHeadModel.from_pretrained(args.model_type)
    hf_model.to(args.device)

    # Wrap HF model to match nanoGPT interface: forward(tokens) -> (logits, loss)
    class _HFWrapper(torch.nn.Module):
        def __init__(self, hf_model):
            super().__init__()
            self.hf_model = hf_model

        def forward(self, tokens, targets=None):
            out = self.hf_model(tokens)
            return out.logits, None

    wrapped = _HFWrapper(hf_model)
    wrapped.to(args.device)
    evaluate(wrapped, args.device)
