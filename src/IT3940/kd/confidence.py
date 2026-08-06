from typing import Literal

import torch
import torch.nn.functional as F


ConfidenceSignal = Literal[
    "mcp",
    "entropy",
    "margin",
    "gt_prob",
]

PriorityOrientation = Literal[
    "confidence",
    "uncertainty",
]


def compute_confidence_scores(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    signal: ConfidenceSignal,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute raw teacher-confidence score s_i for each sample (before ranking)."""
    teacher_probs = F.softmax(teacher_logits, dim=1)

    if signal == "mcp":
        scores = teacher_probs.max(dim=1).values

    elif signal == "entropy":
        num_classes = teacher_probs.size(1)
        entropy = -(teacher_probs * torch.log(teacher_probs.clamp_min(eps))).sum(dim=1)
        max_entropy = torch.log(teacher_probs.new_tensor(float(num_classes)))
        scores = 1.0 - entropy / max_entropy

    elif signal == "margin":
        top_two_probs = teacher_probs.topk(k=2, dim=1).values
        scores = top_two_probs[:, 0] - top_two_probs[:, 1]

    elif signal == "gt_prob":
        scores = teacher_probs.gather(
            dim=1,
            index=labels.unsqueeze(1),
        ).squeeze(1)

    else:
        raise ValueError(f"Unknown confidence signal: {signal}")

    return scores


def scores_to_percentile_ranks(scores: torch.Tensor) -> torch.Tensor:
    """Map raw scores to empirical CDF ranks r_i in [0, 1] using mid-rank for ties."""
    scores = scores.detach().flatten().to(torch.float64)
    num_samples = scores.numel()
    if num_samples == 0:
        return scores.to(torch.float32)

    sorted_scores, order = scores.sort()
    ranks = torch.empty(num_samples, dtype=torch.float64, device=scores.device)

    start = 0
    while start < num_samples:
        end = start + 1
        while end < num_samples and sorted_scores[end] == sorted_scores[start]:
            end += 1

        # 1-indexed mid-rank for tied values, then normalize to [0, 1].
        mid_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = (mid_rank - 0.5) / num_samples
        start = end

    return ranks.to(torch.float32)


def compute_priority_from_rank(
    ranks: torch.Tensor,
    orientation: PriorityOrientation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert percentile ranks r_i into priority scores (s_i, a_i)."""
    if orientation == "confidence":
        priority_s = ranks
        priority_a = 2.0 * ranks - 1.0
    elif orientation == "uncertainty":
        priority_s = 1.0 - ranks
        priority_a = 1.0 - 2.0 * ranks
    else:
        raise ValueError(f"Unknown priority orientation: {orientation}")

    return priority_s, priority_a
