from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from src.models.model_base import ModelBase, ModelKind
from src.data_models.data_models import TrainingData, InputData, OutputData
from src.utils.data_split import ValidationSplit, train_val_split


class ScoringModelBase(ModelBase[OutputData], ABC):
    """Base class for scoring models."""

    @property
    def model_kind(self) -> ModelKind:
        """Return the kind of model."""
        return "scoring"

    @abstractmethod
    def get_preprocessor(self) -> Any:
        """Return the preprocessor used by this model."""

    def evaluate(self, data: TrainingData, split: ValidationSplit) -> tuple[float, float]:
        """Compute pairwise accuracy on train and validation subsets.

        The data is split at the raw entry level; each subset is evaluated
        independently using predict(). Ties and both_bad comparisons are skipped.

        Args:
            data: Raw evaluation data.
            split: How to divide data into train and validation.

        Returns:
            (train_accuracy, val_accuracy)
        """
        train_data, val_data = train_val_split(data, split.val_fraction, split.seed)
        train_acc = self._evaluate_subset(train_data)
        val_acc = self._evaluate_subset(val_data) if val_data is not None else 0.0
        return train_acc, val_acc

    def _evaluate_subset(self, data: TrainingData) -> float:
        """Compute pairwise accuracy on a single data subset.

        Entries are grouped by model pair so that predict() is called once per
        unique (model_a, model_b) combination with only the two relevant models,
        avoiding unnecessary computation for the full model pool.
        """
        valid_entries = [e for e in data.entries if e.winner in ("model_a", "model_b")]
        if not valid_entries:
            return 0.0

        groups: dict[frozenset[str], list] = defaultdict(list)
        for entry in valid_entries:
            groups[frozenset({entry.model_a, entry.model_b})].append(entry)

        correct = 0
        counted = 0
        for model_pair, entries in groups.items():
            model_names = list(model_pair)
            unique_prompts = list(dict.fromkeys(e.user_prompt for e in entries))
            prompt_to_idx = {p: i for i, p in enumerate(unique_prompts)}

            output = self.predict(InputData(prompts=unique_prompts, model_names=model_names))
            scores = output.scores  # dict[str, np.ndarray[n_prompts]]

            for entry in entries:
                if entry.model_a not in scores or entry.model_b not in scores:
                    continue
                pidx = prompt_to_idx[entry.user_prompt]
                score_a = scores[entry.model_a][pidx]
                score_b = scores[entry.model_b][pidx]
                if entry.winner == "model_a":
                    correct += int(score_a > score_b)
                else:
                    correct += int(score_b > score_a)
                counted += 1

        return correct / counted if counted > 0 else 0.0

    @staticmethod
    def assert_kind(model: ModelBase) -> "ScoringModelBase":
        if model.model_kind != "scoring":
            raise ValueError(f"Expected scoring model, but got {model.model_kind} model")

        return model
