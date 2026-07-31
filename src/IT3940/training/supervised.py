from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer


def _reduce_sums(
    total_loss: float,
    total_correct: float,
    total_samples: float,
    device: torch.device,
) -> tuple[float, float, float]:
    """Sum loss/correct/samples across DDP ranks when distributed is active."""
    if not (dist.is_available() and dist.is_initialized()):
        return total_loss, total_correct, total_samples

    totals = torch.tensor(
        [total_loss, total_correct, total_samples],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return totals[0].item(), totals[1].item(), totals[2].item()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Train the model 1 epoch."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad() # delete gradients from previous epoch
        logits = model(images) # forward
        loss = criterion(logits, labels) # compute loss
        loss.backward() # backpropagate the loss
        optimizer.step() # update weights

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    total_loss, total_correct, total_samples = _reduce_sums(
        total_loss, total_correct, total_samples, device
    )

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model after training 1 epoch."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    total_loss, total_correct, total_samples = _reduce_sums(
        total_loss, total_correct, total_samples, device
    )

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }
