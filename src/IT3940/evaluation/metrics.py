"""Classification evaluation metrics for final reporting.

Kept separate from training-loop evaluate helpers so final-protocol
metrics (Top-1, NLL, and later ECE, etc.) can grow independently.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def evaluate_classification(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate Top-1 accuracy and mean NLL on a labeled split.

    Returns:
        Dict with:
          - accuracy: Top-1 accuracy in [0, 1]
          - nll: mean negative log-likelihood of the true class
    """
    model.eval()

    total_nll = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(images)
        nll = F.cross_entropy(logits, labels, reduction="sum")

        batch_size = labels.size(0)
        total_nll += nll.item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot evaluate on an empty dataloader.")

    return {
        "accuracy": total_correct / total_samples,
        "nll": total_nll / total_samples,
    }
