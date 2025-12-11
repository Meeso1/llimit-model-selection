from dataclasses import dataclass, field


@dataclass
class TrainingHistoryEntry:
    epoch: int
    total_loss: float
    val_loss: float | None
    train_accuracy: float
    val_accuracy: float | None
    additional_metrics: dict[str, float] = field(default_factory=dict)

    def to_wandb_dict(self) -> dict[str, float | None]:
        result = {
            "train_loss": self.total_loss,
            "val_loss": self.val_loss,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
        }
        result.update(self.additional_metrics)
        return result


@dataclass
class TrainingHistory:
    total_loss: list[float]
    val_loss: list[float | None]
    train_accuracy: list[float]
    val_accuracy: list[float | None]
    additional_metrics: dict[str, list[float | None]] = field(default_factory=dict)

    @staticmethod
    def from_entries(entries: list[TrainingHistoryEntry]) -> "TrainingHistory":
        ordered_entries = sorted(entries, key=lambda x: x.epoch)
        
        # Collect all unique metric keys from all entries
        all_metric_keys = set()
        for entry in ordered_entries:
            all_metric_keys.update(entry.additional_metrics.keys())
        
        # Build additional_metrics dict with lists
        additional_metrics = {}
        for key in all_metric_keys:
            additional_metrics[key] = [
                entry.additional_metrics.get(key, None)
                for entry in ordered_entries
            ]
        
        return TrainingHistory(
            total_loss=[entry.total_loss for entry in ordered_entries],
            val_loss=[entry.val_loss for entry in ordered_entries],
            train_accuracy=[entry.train_accuracy for entry in ordered_entries],
            val_accuracy=[entry.val_accuracy for entry in ordered_entries],
            additional_metrics=additional_metrics,
        )
