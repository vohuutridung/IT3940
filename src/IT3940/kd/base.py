from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class KDObjective(nn.Module, ABC):
    """Base interface for KD objectives."""

    @abstractmethod
    def forward(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the total, CE, and KD losses."""
        raise NotImplementedError
