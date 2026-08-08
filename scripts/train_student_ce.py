"""Train CE-only student baseline WRN-16-2 (DDP).

Protocol (shared with all student KD variants):
  global batch 128, lr 0.1, 240 epochs, SGD+Nesterov, cosine → 1e-5
  aug: RandomCrop + HFlip only
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler

from IT3940.data.cifar100 import CIFAR100
from IT3940.models.wrn import STUDENT_ARCH, STUDENT_ARCH_NAME, WideResNet
from IT3940.training.supervised import evaluate, train_one_epoch
from IT3940.utils.checkpoint import save_checkpoint
from IT3940.utils.hub import upload_checkpoint

GLOBAL_BATCH_SIZE = 128
NUM_EPOCHS = 240
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
ETA_MIN = 1e-5
SEED = 42
REPO_ID = "vohuutridung/IT3940_new"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main = local_rank == 0
    world_size = dist.get_world_size()

    if GLOBAL_BATCH_SIZE % world_size != 0:
        raise SystemExit(
            f"GLOBAL_BATCH_SIZE={GLOBAL_BATCH_SIZE} must be divisible by world_size={world_size}"
        )
    batch_size = GLOBAL_BATCH_SIZE // world_size
    depth, widen_factor = STUDENT_ARCH

    set_seed(SEED + local_rank)

    # Keep the new protocol artifact separate from the legacy student_ce.pt.
    checkpoint_path = f"checkpoints/student_ce_{STUDENT_ARCH_NAME}.pt"
    path_in_repo = f"student/student_ce_{STUDENT_ARCH_NAME}.pt"

    if is_main:
        print(f"Student arch: WRN-{depth}-{widen_factor} (CE-only)")
        print(
            f"Using {world_size} GPUs, global batch={GLOBAL_BATCH_SIZE}, "
            f"per-GPU batch={batch_size}, lr={LEARNING_RATE}, epochs={NUM_EPOCHS}"
        )

    data = CIFAR100(train_aug="student")
    train_sampler = DistributedSampler(data.train_dataset, shuffle=True)
    train_loader = data.get_train_loader(
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
    )
    val_sampler = DistributedSampler(data.val_dataset, shuffle=False)
    val_loader = data.get_val_loader(
        batch_size=batch_size,
        sampler=val_sampler,
    )

    model = WideResNet(depth=depth, widen_factor=widen_factor, num_classes=100).to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=ETA_MIN,
    )

    best_accuracy = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        scheduler.step()

        if is_main:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}/{NUM_EPOCHS}, "
                f"LR: {current_lr:.6f}, "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Accuracy: {train_metrics['accuracy']:.4f}, "
                f"Validation Loss: {val_metrics['loss']:.4f}, "
                f"Validation Accuracy: {val_metrics['accuracy']:.4f}"
            )
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                save_checkpoint(
                    path=checkpoint_path,
                    model=model.module,
                    optimizer=optimizer,
                    epoch=epoch,
                    accuracy=best_accuracy,
                    metadata={
                        "role": "student_ce",
                        "arch": STUDENT_ARCH_NAME,
                    },
                )
                print(f"Save new best checkpoint with accuracy: {best_accuracy:.4f}")

    if is_main:
        print(f"Training completed! Best validation accuracy: {best_accuracy:.4f}")
        url = upload_checkpoint(
            local_path=checkpoint_path,
            repo_id=REPO_ID,
            path_in_repo=path_in_repo,
        )
        print(f"Uploaded checkpoint to {url}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
