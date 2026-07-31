import torch
import torch.nn as nn
from torch.optim import Optimizer
from pathlib import Path

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    accuracy: float,
) -> None:
    """Save a model checkpoint to local file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
    }, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
) -> dict[str, Any]:
    """Load a model checkpoint from local file."""
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint