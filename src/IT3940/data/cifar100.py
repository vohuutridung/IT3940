from datasets import load_dataset
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
import os
import torch


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Cap workers so 2 DDP processes don't oversubscribe the CPU.
NUM_WORKERS = min(4, max(1, (os.cpu_count() or 1) // 2))

# Student: fixed weak aug. Teacher: strong aug (RandAugment + RandomErasing).
STUDENT_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

TEACHER_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    transforms.RandomErasing(p=0.25),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])


class CIFAR100:
    def __init__(
        self,
        id: str = "uoft-cs/cifar100",
        val_ratio: float = 0.1,
        seed: int = 42,
        train_aug: str = "student",
    ):
        if train_aug not in {"student", "teacher"}:
            raise ValueError(f"train_aug must be 'student' or 'teacher', got {train_aug!r}")

        self.id = id
        self.val_ratio = val_ratio
        self.seed = seed
        self.train_aug = train_aug

        self.full_train_dataset = load_dataset(id, split="train")
        self.test_dataset = load_dataset(id, split="test")

        split_dataset = self.full_train_dataset.train_test_split(test_size=val_ratio, seed=seed)
        indexed_train = split_dataset["train"].add_column(
            "index",
            list(range(len(split_dataset["train"]))),
        )
        self.train_dataset = indexed_train
        self.val_dataset = split_dataset["test"]

        self.train_transform = (
            TEACHER_TRAIN_TRANSFORM if train_aug == "teacher" else STUDENT_TRAIN_TRANSFORM
        )
        self.test_transform = TEST_TRANSFORM

        # Register lazy transforms: applied on-the-fly whenever samples are fetched
        self.train_dataset.set_transform(self._transform_train_batch)
        self.train_eval_dataset = indexed_train.with_transform(self._transform_eval_batch)
        self.val_dataset.set_transform(self._transform_test_batch)
        self.test_dataset.set_transform(self._transform_test_batch)

    def _transform_train_batch(self, batch):
        return {
            "images": torch.stack([self.train_transform(image) for image in batch["img"]]),
            "labels": batch["fine_label"],
            "indices": batch["index"],
        }

    def _transform_eval_batch(self, batch):
        return {
            "images": torch.stack([self.test_transform(image) for image in batch["img"]]),
            "labels": batch["fine_label"],
            "indices": batch["index"],
        }

    def _transform_test_batch(self, batch):
        return {
            "images": torch.stack([self.test_transform(image) for image in batch["img"]]),
            "labels": batch["fine_label"],
        }

    def get_train_loader(
        self,
        batch_size: int = 128,
        sampler: Sampler | None = None,
        shuffle: bool | None = None,
        num_workers: int = NUM_WORKERS,
    ):
        if shuffle is None:
            shuffle = sampler is None
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_train_eval_loader(
        self,
        batch_size: int = 256,
        num_workers: int = NUM_WORKERS,
    ):
        """Deterministic train loader for teacher rank precomputation."""
        return DataLoader(
            self.train_eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_val_loader(
        self,
        batch_size: int = 128,
        sampler: Sampler | None = None,
        num_workers: int = NUM_WORKERS,
    ):
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_test_loader(
        self,
        batch_size: int = 128,
        sampler: Sampler | None = None,
        num_workers: int = NUM_WORKERS,
    ):
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
