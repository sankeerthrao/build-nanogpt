"""Model export utilities for nanoGPT."""


def __getattr__(name):
    """Lazy imports to avoid loading export dependencies at package import time."""
    if name in ("export_onnx", "export_torchscript"):
        from nanogpt.export.exporter import export_onnx, export_torchscript
        return {"export_onnx": export_onnx, "export_torchscript": export_torchscript}[name]
    raise AttributeError(f"module 'nanogpt.export' has no attribute {name!r}")


__all__ = ["export_onnx", "export_torchscript"]
