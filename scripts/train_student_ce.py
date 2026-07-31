from IT3940.data.cifar100 import CIFAR100
from IT3940.models.wrn import WideResNet
from IT3940.training.supervised import train_one_epoch, evaluate
from IT3940.utils.checkpoint import save_checkpoint
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    # 1. Hyperparameters
    seed = 42
    batch_size = 512
    num_epochs = 240

    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 5e-4

    checkpoint_path = "checkpoints/student_ce.pt"
    path_in_repo = "student/student_ce.pt"


    set_seed(seed)


    # 2. Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # 3. Data
    data = CIFAR100()
    train_loader = data.get_train_loader(batch_size)
    val_loader = data.get_val_loader(batch_size)


    # 4. Teacher model
    model = WideResNet(depth=16, widen_factor=2, num_classes=100,).to(device)


    # 5. Loss
    criterion = nn.CrossEntropyLoss()


    # 6. Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )


    # 7. Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-5,
    )


    # 8. Training loop
    best_accuracy = 0.0

    for epoch in range(1, num_epochs + 1):
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
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch}/{num_epochs}, LR: {current_lr:.6f}, Train Loss: {train_metrics['loss']:.4f}, Train Accuracy: {train_metrics['accuracy']:.4f}, Validation Loss: {val_metrics['loss']:.4f}, Validation Accuracy: {val_metrics['accuracy']:.4f}")

        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                accuracy=best_accuracy,
            )
            print(f"Save new best checkpoint with accuracy: {best_accuracy:.4f}")

    print(f"Training completed! Best validation accuracy: {best_accuracy:.4f}")

    repo_id = "vohuutridung/IT3940"
    url = upload_checkpoint(
        local_path=checkpoint_path,
        repo_id=repo_id,
        path_in_repo=path_in_repo,
    )
    print(f"Uploaded checkpoint to {url}")


if __name__ == "__main__":
    main()