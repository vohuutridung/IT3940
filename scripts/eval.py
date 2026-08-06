"""Evaluate a checkpoint from the Hugging Face Hub on CIFAR-100 test.

Examples:
  python scripts/eval.py --filename teacher/teacher.pt
  python scripts/eval.py --filename student/student_ce.pt --arch student
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from IT3940.data.cifar100 import CIFAR100
from IT3940.models.wrn import WideResNet
from IT3940.training.supervised import evaluate
from IT3940.utils.checkpoint import load_checkpoint
from IT3940.utils.hub import download_checkpoint


ARCHS = {
    "teacher": (40, 2),
    "student": (16, 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a HF Hub checkpoint on CIFAR-100 test.")
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Path of the file inside the HF repo, e.g. teacher/teacher.pt",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="vohuutridung/IT3940",
        help="Hugging Face model repo id",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=sorted(ARCHS),
        default=None,
        help="Model architecture preset (teacher=WRN-40-2, student=WRN-16-2)",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def resolve_arch(args: argparse.Namespace) -> tuple[int, int]:
    if args.arch is not None:
        return ARCHS[args.arch]

    # Infer from common HF paths used in this project.
    if args.filename.startswith("teacher/"):
        return ARCHS["teacher"]
    if args.filename.startswith("student/"):
        return ARCHS["student"]

    raise SystemExit(
        "Could not infer architecture. Pass --arch teacher|student "
    )


def main() -> None:
    args = parse_args()
    depth, widen_factor = resolve_arch(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Repo: {args.repo_id}")
    print(f"File: {args.filename}")
    print(f"Model: WRN-{depth}-{widen_factor}")

    checkpoint_path = download_checkpoint(
        repo_id=args.repo_id,
        filename=args.filename,
    )
    print(f"Downloaded to: {checkpoint_path}")

    model = WideResNet(
        depth=depth,
        widen_factor=widen_factor,
        num_classes=100,
    ).to(device)

    checkpoint = load_checkpoint(checkpoint_path, model=model, device=device)
    if "epoch" in checkpoint or "accuracy" in checkpoint:
        epoch = checkpoint.get("epoch", "?")
        accuracy = checkpoint.get("accuracy", None)
        acc_str = f"{accuracy:.4f}" if isinstance(accuracy, float) else str(accuracy)
        print(f"Checkpoint meta: epoch={epoch}, recorded_accuracy={acc_str}")

    data = CIFAR100()
    test_loader = data.get_test_loader(batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"CIFAR-100 test — accuracy: {100.0 * metrics['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
