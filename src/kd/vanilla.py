import torch.nn.functional as F
import torch
from IT3940.kd.base import KDObjective


class VanillaKD(KDObjective):
    """Standard logit-based KD objective."""
    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 2.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature


    def forward(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the total, CE, and KD losses."""
        ce_loss = F.cross_entropy(student_logits, labels)

        student_log_probs = F.log_softmax(
            student_logits / self.temperature,
            dim=1,
        )
        teacher_probs = F.softmax(
            teacher_logits / self.temperature,
            dim=1,
        )
        kd_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
        ) * (self.temperature ** 2)

        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss

        return {
            'total': total_loss,
            'ce': ce_loss,
            'kd': kd_loss,
        }