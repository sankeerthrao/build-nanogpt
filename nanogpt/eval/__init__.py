"""Evaluation benchmarks for nanoGPT."""


def __getattr__(name):
    """Lazy imports to avoid triggering tiktoken init at package import time."""
    if name in ("render_example", "iterate_examples", "get_most_likely_row"):
        from nanogpt.eval.hellaswag import render_example, iterate_examples, get_most_likely_row
        return {
            "render_example": render_example,
            "iterate_examples": iterate_examples,
            "get_most_likely_row": get_most_likely_row,
        }[name]
    raise AttributeError(f"module 'nanogpt.eval' has no attribute {name!r}")


__all__ = ["render_example", "iterate_examples", "get_most_likely_row"]
