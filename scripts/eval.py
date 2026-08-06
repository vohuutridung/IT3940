"""Evaluate one HF Hub checkpoint on CIFAR-100 (Top-1 + NLL).

Protocol (proposal): selected checkpoints are evaluated on the official
test set. Call once per model; derive ΔCE / ΔVKD offline across runs.

Examples:
  python scripts/eval.py --filename teacher/teacher.pt
  python scripts/eval.py --filename student/student_ce.pt
  python scripts/eval.py --filename student/student_vanilla.pt
  python scripts/eval.py --filename student/student_mcp_weighting_confidence.pt
  python scripts/eval.py --filename student/student_ce.pt --split val
"""

from __future__ import annotations

import argparse

import torch

from IT3940.data.cifar100 import CIFAR100
from IT3940.evaluation.metrics import evaluate_classification
from IT3940.models.wrn import WideResNet
from IT3940.utils.checkpoint import load_checkpoint
from IT3940.utils.hub import download_checkpoint


ARCHS = {
    "teacher": (40, 2),
    "student": (16, 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one HF Hub checkpoint: Top-1 Accuracy and NLL.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Path inside the HF repo, e.g. teacher/teacher.pt or student/student_ce.pt",
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
        help="Architecture preset (teacher=WRN-40-2, student=WRN-16-2). Inferred from path if omitted.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "val"],
        default="test",
        help="Evaluation split (default: test; val for sanity checks only)",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def resolve_arch(args: argparse.Namespace) -> tuple[int, int]:
    if args.arch is not None:
        return ARCHS[args.arch]

    if args.filename.startswith("teacher/"):
        return ARCHS["teacher"]
    if args.filename.startswith("student/"):
        return ARCHS["student"]

    raise SystemExit(
        "Could not infer architecture from --filename. Pass --arch teacher|student."
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

    checkpoint = load_checkpoint(checkpoint_path, model=model, device=device)
    if "epoch" in checkpoint or "accuracy" in checkpoint:
        epoch = checkpoint.get("epoch", "?")
        accuracy = checkpoint.get("accuracy", None)
        acc_str = f"{accuracy:.4f}" if isinstance(accuracy, float) else str(accuracy)
        print(f"Checkpoint meta: epoch={epoch}, recorded_val_accuracy={acc_str}")

    data = CIFAR100()
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
        f"CIFAR-100 {args.split} — "
        f"Top-1: {100.0 * top1:.2f}% ({top1:.4f}), "
        f"NLL: {nll:.4f}"
    )


if __name__ == "__main__":
    main()
