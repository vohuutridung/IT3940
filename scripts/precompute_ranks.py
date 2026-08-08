"""Precompute and cache teacher confidence percentile ranks on the training set."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from IT3940.data.cifar100 import CIFAR100
from IT3940.kd.confidence import ConfidenceSignal
from IT3940.kd.rank_cache import default_cache_path, get_or_build_rank_store
from IT3940.models.wrn import TEACHER_ARCHS, WideResNet
from IT3940.utils.checkpoint import load_checkpoint
from IT3940.utils.hub import download_checkpoint


SIGNALS: list[ConfidenceSignal] = ["mcp", "entropy", "margin", "gt_prob"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute cached percentile ranks r_i for all confidence signals.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        default=None,
        help="Local teacher checkpoint path. If omitted, download from HF.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="vohuutridung/IT3940_new",
        help="HF repo id when downloading teacher checkpoint",
    )
    parser.add_argument(
        "--teacher-arch",
        type=str,
        choices=sorted(TEACHER_ARCHS),
        default="wrn28-10",
        help="Teacher architecture (must match the checkpoint)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Checkpoint filename inside the HF repo; default teacher/teacher_{arch}.pt",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache/confidence_ranks",
        help="Root directory for cached rank tensors",
    )
    parser.add_argument(
        "--signal",
        type=str,
        choices=[*SIGNALS, "all"],
        default="all",
        help="Confidence signal to precompute",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    return parser.parse_args()


def resolve_teacher_filename(args: argparse.Namespace) -> str:
    if args.filename is not None:
        return args.filename
    return f"teacher/teacher_{args.teacher_arch}.pt"


def resolve_teacher_checkpoint(args: argparse.Namespace) -> str:
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
        return args.teacher_checkpoint
    return str(
        download_checkpoint(
            repo_id=args.repo_id,
            filename=resolve_teacher_filename(args),
        )
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth, widen_factor = TEACHER_ARCHS[args.teacher_arch]

    teacher_checkpoint = resolve_teacher_checkpoint(args)
    print(f"Using teacher checkpoint: {teacher_checkpoint}")
    print(f"Teacher arch: WRN-{depth}-{widen_factor} ({args.teacher_arch})")

    teacher = WideResNet(depth=depth, widen_factor=widen_factor, num_classes=100).to(device)
    load_checkpoint(
        teacher_checkpoint,
        model=teacher,
        device=device,
        expected_arch=args.teacher_arch,
        expected_role="teacher",
    )

    data = CIFAR100(train_aug="student")
    signals = SIGNALS if args.signal == "all" else [args.signal]

    for signal in signals:
        store = get_or_build_rank_store(
            teacher=teacher,
            dataset=data.train_eval_dataset,
            signal=signal,
            device=device,
            teacher_checkpoint=teacher_checkpoint,
            teacher_name=args.teacher_arch,
            dataset_id=data.id,
            val_ratio=data.val_ratio,
            seed=data.seed,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            force_recompute=args.force,
        )
        cache_path = default_cache_path(args.cache_dir, store.metadata)
        print(f"[{signal}] cached {store.ranks.numel()} ranks -> {cache_path}")


if __name__ == "__main__":
    main()
