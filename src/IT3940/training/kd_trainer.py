import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from IT3940.kd.base import KDObjective


class KDTrainer:
    """Train a student model using knowledge distillation."""
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        objective: KDObjective,
        optimizer: Optimizer,
        device: torch.device,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.objective = objective
        self.optimizer = optimizer
        self.device = device

        self.teacher.requires_grad_(False)
        self.teacher.eval()

    
    def train_one_epoch(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """Train the student model for one epoch."""
        self.teacher.eval()
        self.student.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_kd_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            images = batch["images"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            indices = batch.get("indices")
            if indices is not None:
                indices = indices.to(self.device, non_blocking=True)

            batch_size = labels.size(0)

            self.optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = self.teacher(images)

            student_logits = self.student(images)

            losses = self.objective(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                labels=labels,
                indices=indices,
            )

            loss = losses['total']

            loss.backward()
            self.optimizer.step()

            total_loss += losses['total'].item() * batch_size
            total_ce_loss += losses['ce'].item() * batch_size
            total_kd_loss += losses['kd'].item() * batch_size

            predictions = student_logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

        total_loss, total_ce_loss, total_kd_loss, total_correct, total_samples = _reduce_sums(
            total_loss, total_ce_loss, total_kd_loss, total_correct, total_samples, self.device
        )

        return {
            'loss': total_loss / total_samples,
            'ce_loss': total_ce_loss / total_samples,
            'kd_loss': total_kd_loss / total_samples,
            'accuracy': total_correct / total_samples,
        }

    @torch.inference_mode()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """Evaluate the student model."""
        self.teacher.eval()
        self.student.eval()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_kd_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            images = batch["images"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            indices = batch.get("indices")
            if indices is not None:
                indices = indices.to(self.device, non_blocking=True)

            batch_size = labels.size(0)

            with torch.no_grad():
                teacher_logits = self.teacher(images)

            student_logits = self.student(images)

            losses = self.objective(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                labels=labels,
                indices=indices,
            )

            total_loss += losses['total'].item() * batch_size
            total_ce_loss += losses['ce'].item() * batch_size
            total_kd_loss += losses['kd'].item() * batch_size

            predictions = student_logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

        total_loss, total_ce_loss, total_kd_loss, total_correct, total_samples = _reduce_sums(
            total_loss, total_ce_loss, total_kd_loss, total_correct, total_samples, self.device
        )

        return {
            'loss': total_loss / total_samples,
            'ce_loss': total_ce_loss / total_samples,
            'kd_loss': total_kd_loss / total_samples,
            'accuracy': total_correct / total_samples,
        }


def _reduce_sums(
    total_loss: float,
    total_ce_loss: float,
    total_kd_loss: float,
    total_correct: float,
    total_samples: float,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    if not (dist.is_available() and dist.is_initialized()):
        return total_loss, total_ce_loss, total_kd_loss, total_correct, total_samples

    totals = torch.tensor(
        [total_loss, total_ce_loss, total_kd_loss, total_correct, total_samples],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return totals[0].item(), totals[1].item(), totals[2].item(), totals[3].item(), totals[4].item()