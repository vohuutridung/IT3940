"""Precompute, cache, and reuse teacher confidence percentile ranks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from IT3940.kd.confidence import (
    ConfidenceSignal,
    compute_confidence_scores,
    scores_to_percentile_ranks,
)


def sanitize_name(name: str) -> str:
    """Turn model/dataset ids into filesystem-safe folder names."""
    name = name.strip().replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    return name.strip("-_.") or "unknown"


@dataclass(frozen=True)
class RankCacheMetadata:
    """Metadata required to validate a cached rank tensor."""

    teacher_name: str
    teacher_checkpoint: str
    teacher_checksum: str
    signal: ConfidenceSignal
    dataset_id: str
    val_ratio: float
    seed: int
    num_samples: int

    @property
    def dataset_name(self) -> str:
        return sanitize_name(self.dataset_id.split("/")[-1])

    def relative_path(self) -> Path:
        """
        cache/{teacher}/{dataset}/seed{seed}_val{val}/checksum{8}/{signal}.pt

        Different model or dataset => different folder tree.
        Checksum folder invalidates cache when teacher weights change.
        """
        split_tag = f"seed{self.seed}_val{self.val_ratio:g}"
        checksum_tag = f"checksum{self.teacher_checksum[:8]}"
        return (
            Path(sanitize_name(self.teacher_name))
            / self.dataset_name
            / split_tag
            / checksum_tag
            / f"{self.signal}.pt"
        )


class ConfidenceRankStore:
    """In-memory store of percentile ranks aligned with train-set indices."""

    def __init__(
        self,
        ranks: torch.Tensor,
        metadata: RankCacheMetadata,
        scores: torch.Tensor | None = None,
    ) -> None:
        if ranks.ndim != 1:
            raise ValueError(f"Expected rank tensor with shape (N,), got {ranks.shape}.")
        if ranks.numel() != metadata.num_samples:
            raise ValueError(
                f"Rank length {ranks.numel()} does not match metadata "
                f"num_samples={metadata.num_samples}."
            )

        self.ranks = ranks
        self.metadata = metadata
        self.scores = scores

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Fetch r_i for batch indices."""
        return self.ranks[indices.long().cpu()].to(device=indices.device)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": asdict(self.metadata),
            "scores": self.scores,
            "ranks": self.ranks,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> ConfidenceRankStore:
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = RankCacheMetadata(**payload["metadata"])
        return cls(
            ranks=payload["ranks"],
            metadata=metadata,
            scores=payload.get("scores"),
        )


def checkpoint_checksum(path: str | Path) -> str:
    """Return SHA-256 checksum for a teacher checkpoint file."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_cache_path(
    cache_dir: str | Path,
    metadata: RankCacheMetadata,
) -> Path:
    return Path(cache_dir) / metadata.relative_path()


def metadata_matches(
    cached: RankCacheMetadata,
    expected: RankCacheMetadata,
) -> bool:
    # The checkpoint checksum identifies the weights. Absolute checkpoint paths
    # differ between local files and Hugging Face cache locations and must not
    # invalidate an otherwise identical rank cache.
    cached_values = asdict(cached)
    expected_values = asdict(expected)
    cached_values.pop("teacher_checkpoint")
    expected_values.pop("teacher_checkpoint")
    return cached_values == expected_values


def _as_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value)


@torch.inference_mode()
def precompute_confidence_ranks(
    teacher: nn.Module,
    dataset,
    signal: ConfidenceSignal,
    device: torch.device,
    batch_size: int = 256,
    eps: float = 1e-8,
    num_workers: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run teacher once on the training set and return (scores, ranks)."""
    teacher.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    score_chunks: list[torch.Tensor] = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = _as_tensor(batch["labels"]).to(device, non_blocking=True)

        teacher_logits = teacher(images)
        batch_scores = compute_confidence_scores(
            teacher_logits=teacher_logits,
            labels=labels,
            signal=signal,
            eps=eps,
        )
        score_chunks.append(batch_scores.cpu())

    scores = torch.cat(score_chunks, dim=0)
    ranks = scores_to_percentile_ranks(scores)
    return scores, ranks


def build_rank_store(
    teacher: nn.Module,
    dataset,
    signal: ConfidenceSignal,
    device: torch.device,
    metadata: RankCacheMetadata,
    batch_size: int = 256,
    eps: float = 1e-8,
    num_workers: int = 0,
) -> ConfidenceRankStore:
    scores, ranks = precompute_confidence_ranks(
        teacher=teacher,
        dataset=dataset,
        signal=signal,
        device=device,
        batch_size=batch_size,
        eps=eps,
        num_workers=num_workers,
    )
    return ConfidenceRankStore(
        ranks=ranks,
        metadata=metadata,
        scores=scores,
    )


def get_or_build_rank_store(
    teacher: nn.Module,
    dataset,
    signal: ConfidenceSignal,
    device: torch.device,
    teacher_checkpoint: str | Path,
    teacher_name: str,
    dataset_id: str,
    val_ratio: float,
    seed: int,
    cache_dir: str | Path = "cache/confidence_ranks",
    batch_size: int = 256,
    eps: float = 1e-8,
    num_workers: int = 0,
    force_recompute: bool = False,
) -> ConfidenceRankStore:
    """Load cached ranks when metadata matches, otherwise precompute and save."""
    teacher_checkpoint = Path(teacher_checkpoint)
    checksum = checkpoint_checksum(teacher_checkpoint)
    metadata = RankCacheMetadata(
        teacher_name=teacher_name,
        teacher_checkpoint=str(teacher_checkpoint.resolve()),
        teacher_checksum=checksum,
        signal=signal,
        dataset_id=dataset_id,
        val_ratio=val_ratio,
        seed=seed,
        num_samples=len(dataset),
    )
    cache_path = default_cache_path(cache_dir, metadata)

    if cache_path.exists() and not force_recompute:
        store = ConfidenceRankStore.load(cache_path)
        if metadata_matches(store.metadata, metadata):
            return store

    store = build_rank_store(
        teacher=teacher,
        dataset=dataset,
        signal=signal,
        device=device,
        metadata=metadata,
        batch_size=batch_size,
        eps=eps,
        num_workers=num_workers,
    )
    store.save(cache_path)
    return store
