


class KDTrainer:
    """Train a student model using knowledge distillation."""
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        objective: KDObjective,
        optimizer: Optimizer,
        device: torch.device,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.objective = objective
        self.optimizer = optimizer
        self.device = device

        self.teacher.requires_grad_(False)
        self.teacher.eval()

    
    def train_one_epoch(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """Train the student model for one epoch."""
        self.teacher.eval()
        self.student.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_kd_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)

            batch_size = labels.size(0)

            self.optimizer.