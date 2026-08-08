"""Train vanilla KD or a confidence-aware KD variant (DDP).

Student protocol (shared with CE-only):
  WRN-16-2, global batch 128, lr 0.1, 240 epochs, weak aug (crop+flip)

Teacher default: WRN-28-10 (second baseline). Use --teacher-arch wrn40-2 for the first.

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
from IT3940.kd.confidence_aware import ConfidenceAwareKD
from IT3940.kd.rank_cache import get_or_build_rank_store
from IT3940.kd.vanilla import VanillaKD
from IT3940.models.wrn import (
    STUDENT_ARCH,
    STUDENT_ARCH_NAME,
    TEACHER_ARCHS,
    WideResNet,
)
from IT3940.training.kd_trainer import KDTrainer
from IT3940.utils.checkpoint import load_checkpoint, save_checkpoint
from IT3940.utils.hub import download_checkpoint, upload_checkpoint

GLOBAL_BATCH_SIZE = 128
NUM_EPOCHS = 240
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
ETA_MIN = 1e-5
SEED = 42


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
    parser.add_argument("--repo-id", type=str, default="vohuutridung/IT3940_new")
    parser.add_argument(
        "--teacher-arch",
        type=str,
        choices=sorted(TEACHER_ARCHS),
        default="wrn28-10",
        help="Teacher architecture (must match the checkpoint)",
    )
    parser.add_argument(
        "--teacher-filename",
        type=str,
        default=None,
        help="HF path inside repo; default teacher/teacher_{arch}.pt",
    )
    parser.add_argument("--cache-dir", type=str, default="cache/confidence_ranks")

    parser.add_argument("--alpha", type=float, default=0.5, help="CE–KD mix λ0")
    parser.add_argument("--temperature", type=float, default=4.0, help="KD temperature τ0")
    parser.add_argument("--weight-gamma", type=float, default=1.0)
    parser.add_argument("--mix-delta", type=float, default=0.2)
    parser.add_argument("--temp-beta", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=SEED)
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


def resolve_teacher_filename(args: argparse.Namespace) -> str:
    if args.teacher_filename is not None:
        return args.teacher_filename
    return f"teacher/teacher_{args.teacher_arch}.pt"


def resolve_teacher_checkpoint(args: argparse.Namespace) -> Path:
    reference = args.teacher_checkpoint or resolve_teacher_filename(args)
    if Path(reference).name == "teacher.pt" and args.teacher_arch != "wrn40-2":
        raise ValueError(
            "Legacy teacher.pt is WRN-40-2; pass --teacher-arch wrn40-2 "
            "or use teacher/teacher_wrn28-10.pt."
        )
    for arch in TEACHER_ARCHS:
        if arch != args.teacher_arch and arch in Path(reference).name:
            raise ValueError(
                f"Teacher reference {reference!r} indicates {arch}, but "
                f"--teacher-arch is {args.teacher_arch}."
            )

    if args.teacher_checkpoint is not None:
        path = Path(args.teacher_checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
        return path
    return download_checkpoint(
        repo_id=args.repo_id,
        filename=resolve_teacher_filename(args),
    )


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

    rank_store = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        rank_store = get_or_build_rank_store(
            teacher=teacher,
            dataset=data.train_eval_dataset,
            signal=args.signal,  # type: ignore[arg-type]
            device=device,
            teacher_checkpoint=teacher_checkpoint,
            teacher_name=args.teacher_arch,
            dataset_id=data.id,
            val_ratio=data.val_ratio,
            seed=data.seed,
            cache_dir=args.cache_dir,
        )

    if dist.is_initialized():
        # Only rank 0 may build/write a missing cache. Other ranks wait and
        # then load the completed file, preventing concurrent torch.save calls.
        dist.barrier()
        if rank_store is None:
            rank_store = get_or_build_rank_store(
                teacher=teacher,
                dataset=data.train_eval_dataset,
                signal=args.signal,  # type: ignore[arg-type]
                device=device,
                teacher_checkpoint=teacher_checkpoint,
                teacher_name=args.teacher_arch,
                dataset_id=data.id,
                val_ratio=data.val_ratio,
                seed=data.seed,
                cache_dir=args.cache_dir,
            )

    assert rank_store is not None
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
    world_size = dist.get_world_size()

    if GLOBAL_BATCH_SIZE % world_size != 0:
        raise SystemExit(
            f"GLOBAL_BATCH_SIZE={GLOBAL_BATCH_SIZE} must be divisible by world_size={world_size}"
        )
    batch_size = GLOBAL_BATCH_SIZE // world_size

    set_seed(args.seed + local_rank)
    name = run_name(args)
    checkpoint_filename = f"student_kd_{args.teacher_arch}_{name}.pt"
    checkpoint_path = Path("checkpoints") / checkpoint_filename
    path_in_repo = f"student/{checkpoint_filename}"

    teacher_depth, teacher_widen = TEACHER_ARCHS[args.teacher_arch]
    student_depth, student_widen = STUDENT_ARCH

    if is_main:
        print(f"Variant: {name}")
        print(f"Teacher: WRN-{teacher_depth}-{teacher_widen} ({args.teacher_arch})")
        print(f"Student: WRN-{student_depth}-{student_widen}")
        print(
            f"Using {world_size} GPUs, global batch={GLOBAL_BATCH_SIZE}, "
            f"per-GPU batch={batch_size}, lr={LEARNING_RATE}, epochs={NUM_EPOCHS}"
        )

    teacher_checkpoint = resolve_teacher_checkpoint(args)
    if is_main:
        print(f"Teacher checkpoint: {teacher_checkpoint}")

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

    teacher = WideResNet(
        depth=teacher_depth,
        widen_factor=teacher_widen,
        num_classes=100,
    ).to(device)
    load_checkpoint(
        teacher_checkpoint,
        model=teacher,
        device=device,
        expected_arch=args.teacher_arch,
        expected_role="teacher",
    )
    teacher.requires_grad_(False)
    teacher.eval()

    objective = build_objective(
        args=args,
        teacher=teacher,
        data=data,
        device=device,
        teacher_checkpoint=teacher_checkpoint,
    )

    student = WideResNet(
        depth=student_depth,
        widen_factor=student_widen,
        num_classes=100,
    ).to(device)
    student = DDP(student, device_ids=[local_rank])

    optimizer = optim.SGD(
        student.parameters(),
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

    trainer = KDTrainer(
        teacher=teacher,
        student=student,
        objective=objective,
        optimizer=optimizer,
        device=device,
    )

    best_accuracy = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)

        train_metrics = trainer.train_one_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        scheduler.step()

        if is_main:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}/{NUM_EPOCHS}, "
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
                    metadata={
                        "role": "student_kd",
                        "arch": STUDENT_ARCH_NAME,
                        "teacher_arch": args.teacher_arch,
                        "variant": name,
                    },
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
