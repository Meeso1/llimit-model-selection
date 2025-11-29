from dataclasses import dataclass


@dataclass
class TrainingHistoryEntry:
    epoch: int
    total_loss: float
    val_loss: float | None
    train_accuracy: float
    val_accuracy: float | None

    def to_wandb_dict(self) -> dict[str, float | None]:
        return {
            "train_loss": self.total_loss,
            "val_loss": self.val_loss,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
        }


@dataclass
class TrainingHistory:
    total_loss: list[float]
    val_loss: list[float | None]
    train_accuracy: list[float]
    val_accuracy: list[float | None]

    @staticmethod
    def from_entries(entries: list[TrainingHistoryEntry]) -> "TrainingHistory":
        ordered_entries = sorted(entries, key=lambda x: x.epoch)
        return TrainingHistory(
            total_loss=[entry.total_loss for entry in ordered_entries],
            val_loss=[entry.val_loss for entry in ordered_entries],
            train_accuracy=[entry.train_accuracy for entry in ordered_entries],
            val_accuracy=[entry.val_accuracy for entry in ordered_entries],
        )
