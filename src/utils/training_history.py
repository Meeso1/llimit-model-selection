from dataclasses import dataclass


@dataclass
class TrainingHistoryEntry:
    epoch: int
    total_loss: float
    val_loss: float | None = None
    # TODO: Expand

    def to_wandb_dict(self) -> dict[str, float | None]:
        return {
            "train_loss": self.total_loss,
            "val_loss": self.val_loss,
        }


@dataclass
class TrainingHistory:
    total_loss: list[float]
    val_loss: list[float | None]

    @staticmethod
    def from_entries(entries: list[TrainingHistoryEntry]) -> "TrainingHistory":
        ordered_entries = sorted(entries, key=lambda x: x.epoch)
        return TrainingHistory(
            total_loss=[entry.total_loss for entry in ordered_entries],
            val_loss=[entry.val_loss for entry in ordered_entries],
        )
