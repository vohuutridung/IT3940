"""Train vanilla KD or a confidence-aware KD variant (DDP).

Examples:
  # Vanilla KD baseline
  torchrun --nproc_per_node=2 scripts/train_kd.py --variant vanilla

  # Confidence-aware: MCP + weighting + confidence emphasis
  torchrun --nproc_per_node=2 scripts/train_kd.py \\
    --variant confidence \\
    --signal mcp \\
    --mechanism weighting \\
    --orientation confidence
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler

from IT3940.data.cifar100 import CIFAR100
from IT3940.kd.base import KDObjective
from IT3940.kd.confidence import ConfidenceSignal, PriorityOrientation
from IT3940.kd.confidence_aware import ConfidenceAwareKD, KDMechanism
from IT3940.kd.rank_cache import get_or_build_rank_store
from IT3940.kd.vanilla import VanillaKD
from IT3940.models.wrn import WideResNet
from IT3940.training.kd_trainer import KDTrainer
from IT3940.utils.checkpoint import load_checkpoint, save_checkpoint
from IT3940.utils.hub import download_checkpoint, upload_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vanilla or confidence-aware KD.")
    parser.add_argument(
        "--variant",
        type=str,
        choices=["vanilla", "confidence"],
        required=True,
        help="KD variant to train",
    )
    parser.add_argument(
        "--signal",
        type=str,
        choices=["mcp", "entropy", "margin", "gt_prob"],
        default=None,
        help="Required for --variant confidence",
    )
    parser.add_argument(
        "--mechanism",
        type=str,
        choices=["weighting", "mixing", "temperature"],
        default=None,
        help="Required for --variant confidence",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        choices=["confidence", "uncertainty"],
        default=None,
        help="Required for --variant confidence",
    )

    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--repo-id", type=str, default="vohuutridung/IT3940")
    parser.add_argument("--teacher-filename", type=str, default="teacher/teacher.pt")
    parser.add_argument("--teacher-name", type=str, default="wrn40-2")
    parser.add_argument("--cache-dir", type=str, default="cache/confidence_ranks")

    parser.add_argument("--alpha", type=float, default=0.5, help="Vanilla CE–KD mix λ0")
    parser.add_argument("--temperature", type=float, default=4.0, help="Vanilla temperature τ0")
    parser.add_argument("--weight-gamma", type=float, default=1.0)
    parser.add_argument("--mix-delta", type=float, default=0.2)
    parser.add_argument("--temp-beta", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1024, help="Per-GPU batch size")
    parser.add_argument("--num-epochs", type=int, default=240)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.variant == "confidence":
        missing = [
            name
            for name, value in [
                ("--signal", args.signal),
                ("--mechanism", args.mechanism),
                ("--orientation", args.orientation),
            ]
            if value is None
        ]
        if missing:
            raise SystemExit(
                f"--variant confidence requires: {', '.join(missing)}"
            )


def run_name(args: argparse.Namespace) -> str:
    if args.variant == "vanilla":
        return "vanilla"
    return f"{args.signal}_{args.mechanism}_{args.orientation}"


def resolve_teacher_checkpoint(args: argparse.Namespace) -> Path:
    if args.teacher_checkpoint is not None:
        path = Path(args.teacher_checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
        return path
    return download_checkpoint(repo_id=args.repo_id, filename=args.teacher_filename)


def build_objective(
    args: argparse.Namespace,
    teacher: torch.nn.Module,
    data: CIFAR100,
    device: torch.device,
    teacher_checkpoint: Path,
) -> KDObjective:
    if args.variant == "vanilla":
        return VanillaKD(alpha=args.alpha, temperature=args.temperature)

    assert args.signal is not None
    assert args.mechanism is not None
    assert args.orientation is not None

    rank_store = get_or_build_rank_store(
        teacher=teacher,
        dataset=data.train_eval_dataset,
        signal=args.signal,  # type: ignore[arg-type]
        device=device,
        teacher_checkpoint=teacher_checkpoint,
        teacher_name=args.teacher_name,
        dataset_id=data.id,
        val_ratio=data.val_ratio,
        seed=data.seed,
        cache_dir=args.cache_dir,
    )
    return ConfidenceAwareKD(
        rank_store=rank_store,
        mechanism=args.mechanism,  # type: ignore[arg-type]
        orientation=args.orientation,  # type: ignore[arg-type]
        alpha=args.alpha,
        temperature=args.temperature,
        weight_gamma=args.weight_gamma,
        mix_delta=args.mix_delta,
        temp_beta=args.temp_beta,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main = local_rank == 0

    set_seed(args.seed + local_rank)
    name = run_name(args)
    checkpoint_path = Path("checkpoints") / f"student_kd_{name}.pt"
    path_in_repo = f"student/{name}.pt"

    if is_main:
        print(f"Variant: {name}")
        print(f"Using {dist.get_world_size()} GPUs, per-GPU batch={args.batch_size}")

    teacher_checkpoint = resolve_teacher_checkpoint(args)
    if is_main:
        print(f"Teacher checkpoint: {teacher_checkpoint}")

    data = CIFAR100()
    train_sampler = DistributedSampler(data.train_dataset, shuffle=True)
    train_loader = data.get_train_loader(
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
    )
    val_sampler = DistributedSampler(data.val_dataset, shuffle=False)
    val_loader = data.get_val_loader(
        batch_size=args.batch_size,
        sampler=val_sampler,
    )

    teacher = WideResNet(depth=40, widen_factor=2, num_classes=100).to(device)
    load_checkpoint(teacher_checkpoint, model=teacher, device=device)
    teacher.requires_grad_(False)
    teacher.eval()

    # Build objective on every rank (rank cache is shared filesystem / recomputed if needed).
    objective = build_objective(
        args=args,
        teacher=teacher,
        data=data,
        device=device,
        teacher_checkpoint=teacher_checkpoint,
    )

    student = WideResNet(depth=16, widen_factor=2, num_classes=100).to(device)
    student = DDP(student, device_ids=[local_rank])

    optimizer = optim.SGD(
        student.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-5,
    )

    trainer = KDTrainer(
        teacher=teacher,
        student=student,
        objective=objective,
        optimizer=optimizer,
        device=device,
    )

    best_accuracy = 0.0
    for epoch in range(1, args.num_epochs + 1):
        train_sampler.set_epoch(epoch)

        train_metrics = trainer.train_one_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        scheduler.step()

        if is_main:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}/{args.num_epochs}, "
                f"LR: {current_lr:.6f}, "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"CE: {train_metrics['ce_loss']:.4f}, "
                f"KD: {train_metrics['kd_loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                save_checkpoint(
                    path=checkpoint_path,
                    model=student.module,
                    optimizer=optimizer,
                    epoch=epoch,
                    accuracy=best_accuracy,
                )
                print(f"Saved new best checkpoint: {best_accuracy:.4f}")

    if is_main:
        print(f"Training completed! Best validation accuracy: {best_accuracy:.4f}")
        url = upload_checkpoint(
            local_path=checkpoint_path,
            repo_id=args.repo_id,
            path_in_repo=path_in_repo,
        )
        print(f"Uploaded checkpoint to {url}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
