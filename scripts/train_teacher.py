from IT3940.data.cifar100 import CIFAR100
from IT3940.models.wrn import WideResNet
from IT3940.training.supervised import train_one_epoch, evaluate
from IT3940.utils.checkpoint import save_checkpoint
from IT3940.utils.hub import upload_checkpoint
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    # 0. Setup distributed training
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main = local_rank == 0
    

    # 1. Hyperparameters
    # batch_size is per-GPU; global batch = batch_size * world_size
    seed = 42
    batch_size = 512
    num_epochs = 240

    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 5e-4

    checkpoint_path = "checkpoints/teacher.pt"
    path_in_repo = "teacher/teacher.pt"

    set_seed(seed + local_rank)

    if is_main:
        print(f"Using {dist.get_world_size()} GPUs, device={device}, per-GPU batch={batch_size}")


    # 2. Data
    data = CIFAR100()
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


    # 3. Teacher model
    model = WideResNet(depth=40, widen_factor=2, num_classes=100).to(device)
    model = DDP(model, device_ids=[local_rank])


    # 4. Loss / optimizer / scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-5,
    )


    # 5. Training loop
    best_accuracy = 0.0

    for epoch in range(1, num_epochs + 1):
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
        current_lr = optimizer.param_groups[0]["lr"]

        if is_main:
            print(
                f"Epoch {epoch}/{num_epochs}, "
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
                )
                print(f"Save new best checkpoint with accuracy: {best_accuracy:.4f}")

    if is_main:
        print(f"Training completed! Best validation accuracy: {best_accuracy:.4f}")
        repo_id = "vohuutridung/IT3940"
        url = upload_checkpoint(
            local_path=checkpoint_path,
            repo_id=repo_id,
            path_in_repo=path_in_repo,
        )
        print(f"Uploaded checkpoint to {url}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
