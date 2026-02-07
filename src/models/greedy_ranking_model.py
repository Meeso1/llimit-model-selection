"""Greedy ranking model based on net wins (minimum feedback arc set approximation)."""

from dataclasses import dataclass
from typing import Any
import numpy as np

from src.models.scoring_model_base import ScoringModelBase
from src.data_models.data_models import TrainingData, InputData
from src.data_models.dense_network_types import PromptRoutingOutput
from src.data_models.simple_scoring_types import PreprocessedTrainingData, PreprocessedComparison
from src.preprocessing.simple_scoring_preprocessor import SimpleScoringPreprocessor
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.wandb_details import WandbDetails
from src.utils.string_encoder import StringEncoder
from src.utils.timer import Timer
from src.utils.data_split import ValidationSplit


class GreedyRankingModel(ScoringModelBase):
    """
    Greedy ranking model that finds the best linear ordering of models.
    
    This model uses a non-iterative greedy algorithm to rank models based on
    their net wins (total wins minus total losses). This approximates the
    solution to the minimum feedback arc set problem, finding a ranking that
    minimizes contradictions in the comparison data.
    """
    
    def __init__(
        self,
        min_model_occurrences: int = 1000,
        score_normalization: str = "negative_rank",
        print_summary: bool = True,
        wandb_details: WandbDetails | None = None,
    ) -> None:
        """
        Initialize Greedy Ranking model.
        
        Args:
            min_model_occurrences: Minimum number of times a model must appear to be included
            score_normalization: How to convert rank to score ("negative_rank", "normalized", "centered")
            print_summary: Whether to print a summary after computing ranking
            wandb_details: Weights & Biases configuration
        """
        super().__init__(wandb_details)
        self.min_model_occurrences = min_model_occurrences
        self.score_normalization = score_normalization
        self.print_summary = print_summary
        self.preprocessor = SimpleScoringPreprocessor(min_model_occurrences=min_model_occurrences)

        self._model_encoder: StringEncoder | None = None
        self._ranking: np.ndarray | None = None  # [num_models] - indices in rank order
        self._scores: np.ndarray | None = None  # [num_models] - scores for each model
        self._net_scores: np.ndarray | None = None  # [num_models] - net win counts
        self._disagreements: int = 0
        self._total_comparisons: int = 0
        self.last_timer: Timer | None = None

    def get_config_for_wandb(self) -> dict[str, Any]:
        return {
            "model_type": "greedy_ranking",
            "min_model_occurrences": self.min_model_occurrences,
            "score_normalization": self.score_normalization,
            "num_models": self._model_encoder.size if self._model_encoder else 0,
        }

    @property
    def ranking(self) -> np.ndarray:
        """Get current ranking array."""
        if self._ranking is None:
            raise RuntimeError("Ranking not computed yet")
        return self._ranking

    @property
    def scores(self) -> np.ndarray:
        """Get current scores array."""
        if self._scores is None:
            raise RuntimeError("Scores not computed yet")
        return self._scores

    def train(
        self,
        data: TrainingData,
        validation_split: ValidationSplit | None = None,
        epochs: int = 0,
        batch_size: int = 32,
    ) -> None:
        """
        Compute the greedy ranking from comparison data.
        
        Note: This model doesn't use iterative training, so epochs and batch_size
        are ignored. The ranking is computed in a single pass.
        
        Args:
            data: Training data containing pairwise comparisons
            validation_split: Not used (kept for API compatibility)
            epochs: Not used (kept for API compatibility, should be 0)
            batch_size: Not used (kept for API compatibility)
        """
        if epochs != 0:
            print(f"Warning: GreedyRankingModel doesn't use iterative training. "
                  f"The 'epochs' parameter (value={epochs}) will be ignored.")
        
        if validation_split is not None and validation_split.val_fraction > 0:
            print(f"Warning: GreedyRankingModel doesn't use validation split. "
                  f"All data will be used to compute the ranking.")

        with Timer("train", verbosity="start+end") as train_timer:
            self.last_timer = train_timer
            
            with Timer("preprocess", verbosity="start+end", parent=train_timer):
                preprocessed_data = self._initialize_and_preprocess(data)
            
            with Timer("compute_ranking", verbosity="start+end", parent=train_timer):
                self._compute_ranking(preprocessed_data)
            
            with Timer("compute_metrics", verbosity="start+end", parent=train_timer):
                metrics = self._compute_metrics(preprocessed_data)
            
            if self.print_summary:
                self._print_summary(metrics)
            
            # Log to wandb if configured
            if self.wandb_details is not None:
                self.init_wandb_if_needed()
                # Create a single "epoch" entry for logging
                entry = TrainingHistoryEntry(
                    epoch=1,
                    total_loss=None,
                    val_loss=None,
                    train_accuracy=metrics["accuracy"],
                    val_accuracy=None,
                    additional_metrics=metrics,
                )
                self.log_to_wandb(entry)
                self.finish_wandb_if_needed()

    def predict(
        self,
        X: InputData,
        batch_size: int = 32,
    ) -> PromptRoutingOutput:
        """
        Predict best model for each prompt based on learned ranking.
        
        Since this model doesn't look at prompts, it will always return
        the same scores regardless of the prompt.
        
        Args:
            X: Input data containing prompts and model names
            batch_size: Not used (kept for API compatibility)
            
        Returns:
            PromptRoutingOutput with scores for each model across all prompts
        """
        if self._model_encoder is None or self._scores is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        n_prompts = len(X.prompts)
        scores_dict = {}

        encoded_model_names, known_model_names = self._model_encoder.encode_known(X.model_names)
        
        for model_name, model_id in zip(known_model_names, encoded_model_names):
            score = self._scores[model_id]
            scores_dict[model_name] = np.full(n_prompts, score)

        # Unknown models get score of 0 (average)
        for model_name in X.model_names:
            if model_name not in known_model_names:
                scores_dict[model_name] = np.zeros(n_prompts)
        
        return PromptRoutingOutput(_scores=scores_dict)

    def get_history(self) -> TrainingHistory:
        """Get training history (single entry for this non-iterative model)."""
        # Return empty history since there's no iterative training
        return TrainingHistory.from_entries([])

    def get_all_model_scores(self) -> dict[str, float]:
        """
        Get scores for all known models.
        
        Returns:
            Dictionary mapping model names to their learned scores
        """
        if self._model_encoder is None or self._scores is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        return {
            self._model_encoder.decode(i): float(self._scores[i])
            for i in range(self._model_encoder.size)
        }

    def get_ranking(self) -> list[tuple[str, float, float]]:
        """
        Get the ranking with model names, scores, and net scores.
        
        Returns:
            List of (model_name, score, net_score) tuples in rank order
        """
        if self._model_encoder is None or self._ranking is None or self._scores is None or self._net_scores is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        return [
            (
                self._model_encoder.decode([model_id])[0],
                float(self._scores[model_id]),
                float(self._net_scores[model_id])
            )
            for model_id in self._ranking
        ]

    def get_state_dict(self) -> dict[str, Any]:
        """
        Get state dictionary for saving the model.
        
        Returns:
            State dictionary containing all model parameters and configuration
        """
        if self._ranking is None or self._scores is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        if self._model_encoder is None:
            raise RuntimeError("Model encoder not initialized")
        
        return {
            "min_model_occurrences": self.min_model_occurrences,
            "score_normalization": self.score_normalization,
            "print_summary": self.print_summary,
            "ranking": self._ranking.tolist(),
            "scores": self._scores.tolist(),
            "net_scores": self._net_scores.tolist() if self._net_scores is not None else None,
            "disagreements": self._disagreements,
            "total_comparisons": self._total_comparisons,
            "model_encoder": self._model_encoder.get_state_dict(),
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "GreedyRankingModel":
        """
        Load model from state dictionary.
        
        Args:
            state_dict: State dictionary from get_state_dict()
            
        Returns:
            Loaded model instance
        """
        model = cls(
            min_model_occurrences=state_dict["min_model_occurrences"],
            score_normalization=state_dict["score_normalization"],
            print_summary=state_dict.get("print_summary", True),
        )
        
        model._model_encoder = StringEncoder.load_state_dict(state_dict["model_encoder"])
        model._ranking = np.array(state_dict["ranking"])
        model._scores = np.array(state_dict["scores"])
        model._net_scores = np.array(state_dict["net_scores"]) if state_dict.get("net_scores") is not None else None
        model._disagreements = state_dict.get("disagreements", 0)
        model._total_comparisons = state_dict.get("total_comparisons", 0)
        
        return model

    def _initialize_and_preprocess(self, data: TrainingData) -> PreprocessedTrainingData:
        """
        Initialize model and preprocess training data.
        
        Args:
            data: Training data to initialize from
            
        Returns:
            Preprocessed training data
        """
        preprocessed_data = self.preprocessor.preprocess(data)
        self._model_encoder = preprocessed_data.model_encoder
        return preprocessed_data

    def _compute_ranking(self, data: PreprocessedTrainingData) -> None:
        """
        Compute the greedy ranking based on net wins.
        
        Args:
            data: Preprocessed training data
        """
        n_models = self._model_encoder.size
        
        # Construct win matrix W[i, j] = number of times model i beat model j
        W = np.zeros((n_models, n_models), dtype=np.int64)  # [n_models, n_models]
        
        for comparison in data.comparisons:
            i = comparison.model_id_a
            j = comparison.model_id_b
            
            # Only count win/loss comparisons for ranking
            if comparison.winner == "model_a":
                W[i, j] += 1
            elif comparison.winner == "model_b":
                W[j, i] += 1
            # Ignore ties and both_bad for ranking computation
        
        # Compute net scores: total wins - total losses
        net_scores = W.sum(axis=1) - W.sum(axis=0)  # [n_models]
        self._net_scores = net_scores.astype(np.float64)
        
        # Sort models by net score (descending) to get ranking
        # argsort gives ascending order, so we negate to get descending
        self._ranking = np.argsort(-net_scores)  # [n_models]
        
        # Convert ranking to scores
        self._scores = self._ranking_to_scores(self._ranking, n_models)

    def _ranking_to_scores(self, ranking: np.ndarray, n_models: int) -> np.ndarray:
        """
        Convert ranking array to score array.
        
        Args:
            ranking: Array of model indices in rank order  # [n_models]
            n_models: Total number of models
            
        Returns:
            Score array  # [n_models]
        """
        # Create a reverse mapping: model_id -> rank
        rank_of_model = np.zeros(n_models, dtype=np.int64)  # [n_models]
        for rank, model_id in enumerate(ranking):
            rank_of_model[model_id] = rank
        
        # Convert ranks to scores
        scores = np.zeros(n_models, dtype=np.float64)  # [n_models]
        
        if self.score_normalization == "negative_rank":
            # Higher rank (smaller number) gets less negative score
            scores = -(rank_of_model.astype(np.float64))
        elif self.score_normalization == "normalized":
            # Normalize to [0, 1] range
            scores = (n_models - rank_of_model - 1) / n_models
        elif self.score_normalization == "centered":
            # Center around 0
            scores = -(rank_of_model - (n_models - 1) / 2.0)
        else:
            raise ValueError(f"Unknown score_normalization: {self.score_normalization}")
        
        return scores

    def _compute_metrics(self, data: PreprocessedTrainingData) -> dict[str, float]:
        """
        Compute accuracy and optimality metrics.
        
        Args:
            data: Preprocessed training data
            
        Returns:
            Dictionary of metrics
        """
        n_models = self._model_encoder.size
        
        # Create rank lookup
        rank_of_model = np.zeros(n_models, dtype=np.int64)
        for rank, model_id in enumerate(self._ranking):
            rank_of_model[model_id] = rank
        
        # Compute accuracy on all comparisons
        accuracy, component_accuracies = self._compute_accuracy(
            data.comparisons, rank_of_model, n_models
        )
        
        # Compute disagreements (backward edges)
        disagreements = self._compute_disagreements(data.comparisons, rank_of_model)
        self._disagreements = disagreements
        self._total_comparisons = len(data.comparisons)
        
        # Theoretical maximum accuracy
        max_accuracy = (self._total_comparisons - disagreements) / self._total_comparisons if self._total_comparisons > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "ranking_accuracy": component_accuracies["ranking_accuracy"],
            "tie_accuracy": component_accuracies["tie_accuracy"],
            "both_bad_accuracy": component_accuracies["both_bad_accuracy"],
            "disagreements": float(disagreements),
            "total_comparisons": float(self._total_comparisons),
            "max_theoretical_accuracy": max_accuracy,
            "net_score_mean": float(np.mean(self._net_scores)),
            "net_score_std": float(np.std(self._net_scores)),
            "net_score_min": float(np.min(self._net_scores)),
            "net_score_max": float(np.max(self._net_scores)),
        }
        
        return metrics

    def _compute_accuracy(
        self,
        comparisons: list[PreprocessedComparison],
        rank_of_model: np.ndarray,
        n_models: int,
    ) -> tuple[float, dict[str, float]]:
        """
        Compute accuracy on comparisons.
        
        Args:
            comparisons: List of comparisons to evaluate
            rank_of_model: Array mapping model_id to rank  # [n_models]
            n_models: Total number of models
            
        Returns:
            Tuple of (overall_accuracy, component_accuracies)
        """
        n_comparisons = len(comparisons)
        if n_comparisons == 0:
            return 0.0, {
                "ranking_accuracy": 0.0,
                "tie_accuracy": 0.0,
                "both_bad_accuracy": 0.0,
            }
        
        # Component tracking
        n_ranking = 0
        n_ties = 0
        n_both_bad = 0
        ranking_accuracy_sum = 0.0
        tie_accuracy_sum = 0.0
        both_bad_accuracy_sum = 0.0
        
        for comparison in comparisons:
            model_id_a = comparison.model_id_a
            model_id_b = comparison.model_id_b
            
            rank_a = rank_of_model[model_id_a]
            rank_b = rank_of_model[model_id_b]
            
            # Check accuracy based on comparison type
            if comparison.winner == "model_a":
                n_ranking += 1
                # Correct if a is ranked higher (lower rank number)
                if rank_a < rank_b:
                    ranking_accuracy_sum += 1.0
            elif comparison.winner == "model_b":
                n_ranking += 1
                # Correct if b is ranked higher (lower rank number)
                if rank_b < rank_a:
                    ranking_accuracy_sum += 1.0
            elif comparison.winner == "tie":
                n_ties += 1
                # Correct if both are in top half
                if rank_a < n_models / 2 and rank_b < n_models / 2:
                    tie_accuracy_sum += 1.0
            elif comparison.winner == "both_bad":
                n_both_bad += 1
                # Correct if both are in bottom half
                if rank_a >= n_models / 2 and rank_b >= n_models / 2:
                    both_bad_accuracy_sum += 1.0
        
        # Compute component accuracies
        ranking_accuracy = ranking_accuracy_sum / n_ranking if n_ranking > 0 else 0.0
        tie_accuracy = tie_accuracy_sum / n_ties if n_ties > 0 else 0.0
        both_bad_accuracy = both_bad_accuracy_sum / n_both_bad if n_both_bad > 0 else 0.0
        
        # Overall accuracy
        total_accuracy = (ranking_accuracy_sum + tie_accuracy_sum + both_bad_accuracy_sum) / n_comparisons
        
        accuracy_components = {
            "ranking_accuracy": ranking_accuracy,
            "tie_accuracy": tie_accuracy,
            "both_bad_accuracy": both_bad_accuracy,
        }
        
        return total_accuracy, accuracy_components

    def _compute_disagreements(
        self,
        comparisons: list[PreprocessedComparison],
        rank_of_model: np.ndarray,
    ) -> int:
        """
        Compute number of disagreements (backward edges).
        
        Args:
            comparisons: List of comparisons
            rank_of_model: Array mapping model_id to rank  # [n_models]
            
        Returns:
            Number of comparisons that contradict the ranking
        """
        disagreements = 0
        
        for comparison in comparisons:
            model_id_a = comparison.model_id_a
            model_id_b = comparison.model_id_b
            
            rank_a = rank_of_model[model_id_a]
            rank_b = rank_of_model[model_id_b]
            
            # Count disagreements for win/loss comparisons only
            if comparison.winner == "model_a":
                # Disagreement if a won but is ranked lower (higher rank number)
                if rank_a > rank_b:
                    disagreements += 1
            elif comparison.winner == "model_b":
                # Disagreement if b won but is ranked lower (higher rank number)
                if rank_b > rank_a:
                    disagreements += 1
        
        return disagreements

    def _print_summary(self, metrics: dict[str, float]) -> None:
        """Print a summary of the ranking results."""
        print("\n" + "="*70)
        print("GREEDY RANKING MODEL - SUMMARY")
        print("="*70)
        
        print(f"\nRanking Statistics:")
        print(f"  Total models: {self._model_encoder.size}")
        print(f"  Net score range: [{metrics['net_score_min']:.0f}, {metrics['net_score_max']:.0f}]")
        print(f"  Net score mean: {metrics['net_score_mean']:.2f} ± {metrics['net_score_std']:.2f}")
        
        print(f"\nAccuracy Metrics:")
        print(f"  Overall accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Ranking accuracy: {metrics['ranking_accuracy']*100:.2f}%")
        print(f"  Tie accuracy: {metrics['tie_accuracy']*100:.2f}%")
        print(f"  Both bad accuracy: {metrics['both_bad_accuracy']*100:.2f}%")
        
        print(f"\nOptimality:")
        print(f"  Total comparisons: {int(metrics['total_comparisons'])}")
        print(f"  Disagreements (backward edges): {int(metrics['disagreements'])}")
        print(f"  Theoretical max accuracy: {metrics['max_theoretical_accuracy']*100:.2f}%")
        
        print(f"\nTop 10 Models:")
        ranking_list = self.get_ranking()
        for i in range(min(10, len(ranking_list))):
            model_name, score, net_score = ranking_list[i]
            print(f"  {i+1:>2}. {model_name:40s} (net wins: {net_score:>6.0f}, score: {score:>8.3f})")
        
        print("="*70 + "\n")

