"""Precompute and cache teacher confidence percentile ranks on the training set."""

from __future__ import annotations

import argparse

import torch

from IT3940.data.cifar100 import CIFAR100
from IT3940.kd.confidence import ConfidenceSignal
from IT3940.kd.rank_cache import default_cache_path, get_or_build_rank_store
from IT3940.models.wrn import WideResNet
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
        default="vohuutridung/IT3940",
        help="HF repo id when downloading teacher checkpoint",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="teacher/teacher.pt",
        help="Checkpoint filename inside the HF repo",
    )
    parser.add_argument(
        "--teacher-name",
        type=str,
        default="wrn40-2",
        help="Teacher model name used in the cache folder path",
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
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    return parser.parse_args()


def resolve_teacher_checkpoint(args: argparse.Namespace) -> str:
    if args.teacher_checkpoint is not None:
        return args.teacher_checkpoint
    return str(download_checkpoint(repo_id=args.repo_id, filename=args.filename))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_checkpoint = resolve_teacher_checkpoint(args)
    print(f"Using teacher checkpoint: {teacher_checkpoint}")
    print(f"Teacher name: {args.teacher_name}")

    teacher = WideResNet(depth=40, widen_factor=2, num_classes=100).to(device)
    load_checkpoint(teacher_checkpoint, model=teacher, device=device)

    data = CIFAR100()
    signals = SIGNALS if args.signal == "all" else [args.signal]

    for signal in signals:
        store = get_or_build_rank_store(
            teacher=teacher,
            dataset=data.train_eval_dataset,
            signal=signal,
            device=device,
            teacher_checkpoint=teacher_checkpoint,
            teacher_name=args.teacher_name,
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
