import torch
import torch.nn as nn
from torch.optim import Optimizer
from pathlib import Path
from typing import Any


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    accuracy: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a model checkpoint to local file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
    }
    if metadata is not None:
        reserved = checkpoint.keys() & metadata.keys()
        if reserved:
            raise ValueError(
                f"Checkpoint metadata cannot overwrite reserved keys: {sorted(reserved)}"
            )
        checkpoint.update(metadata)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
    expected_arch: str | None = None,
    expected_role: str | None = None,
) -> dict[str, Any]:
    """Load a model checkpoint from local file."""
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    for key, expected in (("arch", expected_arch), ("role", expected_role)):
        actual = checkpoint.get(key)
        if expected is not None and actual is not None and actual != expected:
            raise ValueError(
                f"Checkpoint {key} mismatch: expected {expected!r}, got {actual!r} "
                f"from {path}."
            )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint