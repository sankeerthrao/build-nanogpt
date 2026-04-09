"""Data loading and preprocessing utilities for nanoGPT."""


def __getattr__(name):
    """Lazy imports to avoid ModuleNotFoundError during partial installs."""
    if name in ("DataLoaderLite", "load_tokens"):
        from nanogpt.data.dataloader import DataLoaderLite, load_tokens
        return {"DataLoaderLite": DataLoaderLite, "load_tokens": load_tokens}[name]
    if name == "download_and_tokenize":
        from nanogpt.data.fineweb import download_and_tokenize
        return download_and_tokenize
    raise AttributeError(f"module 'nanogpt.data' has no attribute {name!r}")


__all__ = ["DataLoaderLite", "load_tokens", "download_and_tokenize"]
