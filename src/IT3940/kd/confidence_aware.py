from typing import Literal

import torch
import torch.nn.functional as F

from IT3940.kd.base import KDObjective
from IT3940.kd.confidence import (
    ConfidenceSignal,
    PriorityOrientation,
    compute_priority_from_rank,
)
from IT3940.kd.rank_cache import ConfidenceRankStore


KDMechanism = Literal[
    "weighting",
    "mixing",
    "temperature",
]


class ConfidenceAwareKD(KDObjective):
    def __init__(
        self,
        rank_store: ConfidenceRankStore,
        mechanism: KDMechanism,
        orientation: PriorityOrientation,
        alpha: float = 0.5,
        temperature: float = 4.0,
        weight_gamma: float = 1.0,
        mix_delta: float = 0.2,
        temp_beta: float = 0.5,
        alpha_min: float = 0.1,
        alpha_max: float = 0.9,
        temperature_min: float = 1.0,
        temperature_max: float = 8.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.rank_store = rank_store
        self.mechanism = mechanism
        self.orientation = orientation

        self.alpha = alpha
        self.temperature = temperature
        self.weight_gamma = weight_gamma
        self.mix_delta = mix_delta
        self.temp_beta = temp_beta

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.eps = eps

    @property
    def signal(self) -> ConfidenceSignal:
        return self.rank_store.metadata.signal

    def forward(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            labels=labels,
            indices=indices,
        )

        ranks = self.rank_store.lookup(indices)
        priority_s, priority_a = compute_priority_from_rank(
            ranks=ranks,
            orientation=self.orientation,
        )

        ce_per_sample = F.cross_entropy(
            student_logits,
            labels,
            reduction="none",
        )

        if self.mechanism == "weighting":
            total_loss, kd_per_sample = self._weighting_loss(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                ce_per_sample=ce_per_sample,
                priority_s=priority_s,
            )
        elif self.mechanism == "mixing":
            total_loss, kd_per_sample = self._mixing_loss(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                ce_per_sample=ce_per_sample,
                priority_a=priority_a,
            )
        elif self.mechanism == "temperature":
            total_loss, kd_per_sample = self._temperature_loss(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                ce_per_sample=ce_per_sample,
                priority_a=priority_a,
            )
        else:
            raise ValueError(f"Unknown KD mechanism: {self.mechanism}")

        return {
            "total": total_loss,
            "ce": ce_per_sample.mean(),
            "kd": kd_per_sample.mean(),
        }


    @staticmethod
    def _kd_per_sample(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        temperature: float | torch.Tensor,
    ) -> torch.Tensor:
        """Compute one KL-divergence loss per sample."""
        if isinstance(temperature, torch.Tensor):
            temperature = temperature.unsqueeze(1)

        student_log_probs = F.log_softmax(
            student_logits / temperature,
            dim=1,
        )
        teacher_probs = F.softmax(
            teacher_logits / temperature,
            dim=1,
        )
        kd_per_sample = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=1)

        if isinstance(temperature, torch.Tensor):
            temperature = temperature.squeeze(1)

        return kd_per_sample * temperature**2


    def _weighting_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        ce_per_sample: torch.Tensor,
        priority_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kd_per_sample = self._kd_per_sample(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            temperature=self.temperature,
        )

        weights = (self.eps + priority_s).pow(self.weight_gamma)
        weights = weights / weights.mean().detach()

        total_per_sample = (
            (1.0 - self.alpha) * ce_per_sample
            + self.alpha * weights * kd_per_sample
        )
        return total_per_sample.mean(), kd_per_sample

    
    def _mixing_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        ce_per_sample: torch.Tensor,
        priority_a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kd_per_sample = self._kd_per_sample(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            temperature=self.temperature,
        )

        alpha_per_sample = torch.clamp(
            self.alpha + self.mix_delta * priority_a,
            min=self.alpha_min,
            max=self.alpha_max,
        )
        total_per_sample = (
            (1.0 - alpha_per_sample) * ce_per_sample
            + alpha_per_sample * kd_per_sample
        )
        return total_per_sample.mean(), kd_per_sample


    def _temperature_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        ce_per_sample: torch.Tensor,
        priority_a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temperature_per_sample = torch.clamp(
            self.temperature * torch.exp(self.temp_beta * priority_a),
            min=self.temperature_min,
            max=self.temperature_max,
        )
        kd_per_sample = self._kd_per_sample(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            temperature=temperature_per_sample,
        )
        total_per_sample = (
            (1.0 - self.alpha) * ce_per_sample
            + self.alpha * kd_per_sample
        )
        return total_per_sample.mean(), kd_per_sample


    @staticmethod
    def _validate_inputs(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor | None,
    ) -> None:
        if indices is None:
            raise ValueError(
                "ConfidenceAwareKD requires batch indices to lookup cached percentile ranks."
            )

        if student_logits.shape != teacher_logits.shape:
            raise ValueError("Student and teacher logits must have the same shape.")

        if student_logits.ndim != 2:
            raise ValueError(
                f"Expected logits with shape (B, C), got {student_logits.shape}."
            )

        if labels.shape != student_logits.shape[:1]:
            raise ValueError(
                f"Expected labels with shape (B,), got {labels.shape}."
            )

        if indices.shape != labels.shape:
            raise ValueError(
                f"Expected indices with shape (B,), got {indices.shape}."
            )
