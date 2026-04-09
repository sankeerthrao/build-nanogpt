"""
Training logger with support for multiple backends.

Always writes plain-text and JSON logs to disk.  Optionally logs to
Weights & Biases and/or TensorBoard when enabled.
"""

import os
import json
import time
import warnings
from typing import Optional, List


class TrainingLogger:
    """Multi-backend training logger.

    Writes every metric to a plain-text log file (``log.txt``) and a
    JSON-lines file (``metrics.jsonl``) in the specified *log_dir*.
    Optionally forwards metrics to W&B and/or TensorBoard.

    Args:
        log_dir: Directory to store log files.  Created if it doesn't exist.
        use_wandb: Enable Weights & Biases logging.
        use_tensorboard: Enable TensorBoard logging.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name (optional).
        config: Training configuration dict to log as metadata.
    """

    def __init__(
        self,
        log_dir: str,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        wandb_project: str = "nanogpt",
        wandb_run_name: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Plain-text log
        self._log_path = os.path.join(log_dir, "log.txt")
        self._log_file = open(self._log_path, "a")

        # JSON-lines log
        self._jsonl_path = os.path.join(log_dir, "metrics.jsonl")
        self._jsonl_file = open(self._jsonl_path, "a")

        # --- Optional: Weights & Biases ---
        self._wandb = None
        if use_wandb:
            try:
                import wandb

                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config=config,
                    dir=log_dir,
                )
                self._wandb = wandb
            except ImportError:
                warnings.warn(
                    "wandb is not installed — W&B logging disabled. "
                    "Install with: pip install wandb"
                )
            except Exception as e:
                warnings.warn(f"Failed to initialise wandb: {e}")

        # --- Optional: TensorBoard ---
        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))
            except ImportError:
                warnings.warn(
                    "tensorboard is not installed — TensorBoard logging disabled. "
                    "Install with: pip install tensorboard"
                )
            except Exception as e:
                warnings.warn(f"Failed to initialise TensorBoard: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_text(self, step: int, metric_type: str, value) -> None:
        """Write a single line to the plain-text log."""
        self._log_file.write(f"{step} {metric_type} {value}\n")
        self._log_file.flush()

    def _write_jsonl(self, record: dict) -> None:
        """Write a single JSON object to the JSON-lines log."""
        record["timestamp"] = time.time()
        self._jsonl_file.write(json.dumps(record) + "\n")
        self._jsonl_file.flush()

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_train(
        self,
        step: int,
        loss: float,
        lr: float,
        norm: float,
        dt: float,
        tokens_per_sec: float,
    ) -> None:
        """Log training metrics for a single step.

        Args:
            step: Training step number.
            loss: Training loss (averaged across micro-batches).
            lr: Current learning rate.
            norm: Gradient norm.
            dt: Wall-clock time for this step (seconds).
            tokens_per_sec: Throughput in tokens per second.
        """
        self._write_text(step, "train_loss", f"{loss:.6f}")
        self._write_jsonl({
            "step": step,
            "type": "train",
            "loss": loss,
            "lr": lr,
            "grad_norm": norm,
            "dt_ms": dt * 1000,
            "tokens_per_sec": tokens_per_sec,
        })

        if self._wandb is not None:
            self._wandb.log(
                {
                    "train/loss": loss,
                    "train/lr": lr,
                    "train/grad_norm": norm,
                    "train/dt_ms": dt * 1000,
                    "train/tokens_per_sec": tokens_per_sec,
                },
                step=step,
            )

        if self._tb_writer is not None:
            self._tb_writer.add_scalar("train/loss", loss, step)
            self._tb_writer.add_scalar("train/lr", lr, step)
            self._tb_writer.add_scalar("train/grad_norm", norm, step)
            self._tb_writer.add_scalar("train/tokens_per_sec", tokens_per_sec, step)

    def log_val(self, step: int, val_loss: float) -> None:
        """Log validation loss.

        Args:
            step: Training step number.
            val_loss: Validation loss.
        """
        self._write_text(step, "val", f"{val_loss:.4f}")
        self._write_jsonl({
            "step": step,
            "type": "val",
            "val_loss": val_loss,
        })

        if self._wandb is not None:
            self._wandb.log({"val/loss": val_loss}, step=step)

        if self._tb_writer is not None:
            self._tb_writer.add_scalar("val/loss", val_loss, step)

    def log_hellaswag(self, step: int, accuracy: float) -> None:
        """Log HellaSwag evaluation accuracy.

        Args:
            step: Training step number.
            accuracy: HellaSwag accuracy (0.0 to 1.0).
        """
        self._write_text(step, "hella", f"{accuracy:.4f}")
        self._write_jsonl({
            "step": step,
            "type": "hellaswag",
            "accuracy": accuracy,
        })

        if self._wandb is not None:
            self._wandb.log({"eval/hellaswag_acc": accuracy}, step=step)

        if self._tb_writer is not None:
            self._tb_writer.add_scalar("eval/hellaswag_acc", accuracy, step)

    def log_generation(self, step: int, texts: List[str]) -> None:
        """Log generated text samples.

        Args:
            step: Training step number.
            texts: List of generated text strings.
        """
        for i, text in enumerate(texts):
            self._write_text(step, f"gen_{i}", repr(text))
        self._write_jsonl({
            "step": step,
            "type": "generation",
            "texts": texts,
        })

        if self._wandb is not None:
            try:
                table = self._wandb.Table(columns=["step", "sample_id", "text"])
                for i, text in enumerate(texts):
                    table.add_data(step, i, text)
                self._wandb.log({"generations": table}, step=step)
            except Exception:
                pass  # non-critical

    def close(self) -> None:
        """Flush and close all log files and backends."""
        self._log_file.close()
        self._jsonl_file.close()

        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass

        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass
