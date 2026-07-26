from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import Optimizer
from pathlib import Path


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

    for batch in dataloader: # dataloader is a iterable
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad() # delete gradients from previous epoch
        logits = model(images) # forward
        loss = criterion(logits, labels) # compute loss
        loss.backward() # backpropagate the loss
        optimizer.step() # update weights

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
    
    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples

    return {
        'loss': epoch_loss,
        'accuracy': epoch_accuracy,
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
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)

        logits = model(images) # forward pass
        loss = criterion(logits, labels) # compute loss

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
    
    eval_loss = total_loss / total_samples
    eval_accuracy = total_correct / total_samples

    return {
        'loss': eval_loss,
        'accuracy': eval_accuracy,
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    accuracy: float,
) -> None:
    """Save the model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
    }, path)