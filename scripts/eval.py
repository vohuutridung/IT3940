"""Evaluate one HF Hub checkpoint on CIFAR-100 (Top-1 + NLL).

Protocol (proposal): selected checkpoints are evaluated on the official
test set. Call once per model; derive ΔCE / ΔVKD offline across runs.

Examples:
  python scripts/eval.py --filename teacher/teacher_wrn28-10.pt
  python scripts/eval.py --filename teacher/teacher_wrn40-2.pt
  python scripts/eval.py --filename student/student_ce_wrn16-2.pt
  python scripts/eval.py --filename student/student_kd_wrn28-10_vanilla.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from IT3940.data.cifar100 import CIFAR100
from IT3940.evaluation.metrics import evaluate_classification
from IT3940.models.wrn import (
    STUDENT_ARCH,
    STUDENT_ARCH_NAME,
    TEACHER_ARCHS,
    WideResNet,
)
from IT3940.utils.checkpoint import load_checkpoint
from IT3940.utils.hub import download_checkpoint


ARCHS: dict[str, tuple[int, int]] = {
    **TEACHER_ARCHS,
    "student": STUDENT_ARCH,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one HF Hub checkpoint: Top-1 Accuracy and NLL.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Path inside the HF repo, e.g. teacher/teacher_wrn28-10.pt",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="vohuutridung/IT3940_new",
        help="Hugging Face model repo id",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=sorted(ARCHS),
        default=None,
        help="Architecture preset. Inferred from path if omitted.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "val"],
        default="test",
        help="Evaluation split (default: test; val for sanity checks only)",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def resolve_arch(args: argparse.Namespace) -> tuple[int, int]:
    if args.arch is not None:
        return ARCHS[args.arch]

    name = args.filename
    # A teacher slug inside a student filename records provenance only. The
    # checkpoint architecture remains WRN-16-2.
    if name.startswith("student/") or Path(name).name.startswith("student_"):
        return STUDENT_ARCH
    if "wrn28-10" in name or "wrn2810" in name:
        return TEACHER_ARCHS["wrn28-10"]
    if "wrn40-2" in name or "wrn402" in name:
        return TEACHER_ARCHS["wrn40-2"]
    # Legacy default teacher path before arch was encoded in the filename.
    if name in {"teacher/teacher.pt", "teacher.pt"}:
        return TEACHER_ARCHS["wrn40-2"]

    raise SystemExit(
        "Could not infer architecture from --filename. "
        "Pass --arch wrn28-10|wrn40-2|student."
    )


def main() -> None:
    args = parse_args()
    depth, widen_factor = resolve_arch(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Repo: {args.repo_id}")
    print(f"File: {args.filename}")
    print(f"Model: WRN-{depth}-{widen_factor}")
    print(f"Split: {args.split}")

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

    expected_arch = STUDENT_ARCH_NAME
    if (depth, widen_factor) != STUDENT_ARCH:
        expected_arch = next(
            arch for arch, spec in TEACHER_ARCHS.items()
            if spec == (depth, widen_factor)
        )
    checkpoint = load_checkpoint(
        checkpoint_path,
        model=model,
        device=device,
        expected_arch=expected_arch,
    )
    if "epoch" in checkpoint or "accuracy" in checkpoint:
        epoch = checkpoint.get("epoch", "?")
        accuracy = checkpoint.get("accuracy", None)
        acc_str = f"{accuracy:.4f}" if isinstance(accuracy, float) else str(accuracy)
        print(f"Checkpoint meta: epoch={epoch}, recorded_val_accuracy={acc_str}")
    if "role" in checkpoint or "arch" in checkpoint:
        print(
            "Model meta: "
            f"role={checkpoint.get('role', '?')}, "
            f"arch={checkpoint.get('arch', '?')}, "
            f"teacher_arch={checkpoint.get('teacher_arch', '-')}, "
            f"variant={checkpoint.get('variant', '-')}"
        )

    data = CIFAR100(train_aug="student")
    if args.split == "test":
        loader = data.get_test_loader(batch_size=args.batch_size)
    else:
        loader = data.get_val_loader(batch_size=args.batch_size)

    metrics = evaluate_classification(
        model=model,
        dataloader=loader,
        device=device,
    )

    top1 = metrics["accuracy"]
    nll = metrics["nll"]
    print(
        f"Model {args.filename} — "
        f"Top-1: {100.0 * top1:.2f}% ({top1:.4f}), "
        f"NLL: {nll:.4f}"
    )


if __name__ == "__main__":
    main()
