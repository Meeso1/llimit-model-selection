"""ELO-based scoring model that learns per-model scores using ELO rating system."""

from dataclasses import dataclass
from typing import Any
import numpy as np

from src.models.model_base import ModelBase
from src.data_models.data_models import TrainingData, InputData
from src.data_models.dense_network_types import PromptRoutingOutput
from src.data_models.simple_scoring_types import PreprocessedTrainingData
from src.preprocessing.simple_scoring_preprocessor import SimpleScoringPreprocessor
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.wandb_details import WandbDetails
from src.utils.string_encoder import StringEncoder
from src.utils.timer import Timer
from src.utils.data_split import ValidationSplit, split_simple_scoring_preprocessed_data


class EloScoringModel(ModelBase):
    """
    ELO-based model that learns ratings for each LLM using ELO rating system.
    
    This model uses the ELO rating system to update model scores based on
    pairwise comparisons. Like SimpleScoringModel, it ignores prompts and
    responses, learning only from comparison outcomes.
    """
    
    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
        balance_model_samples: bool = True,
        print_every: int | None = 1,
        tie_both_bad_epsilon: float = 100.0,
        non_ranking_loss_coeff: float = 0.1,
        min_model_occurrences: int = 1000,
        wandb_details: WandbDetails | None = None,
    ) -> None:
        """
        Initialize ELO scoring model.
        
        Args:
            initial_rating: Initial ELO rating for all models
            k_factor: Maximum rating change per comparison (standard ELO K-factor)
            balance_model_samples: Whether to balance samples by model frequency
            print_every: Print progress every N epochs (None to disable)
            tie_both_bad_epsilon: Threshold for tie/both_bad penalties
            non_ranking_loss_coeff: Weight for non-ranking penalties
            min_model_occurrences: Minimum number of times a model must appear to be included
            wandb_details: Weights & Biases configuration
        """
        super().__init__(wandb_details)
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.balance_model_samples = balance_model_samples
        self.print_every = print_every
        self.tie_both_bad_epsilon = tie_both_bad_epsilon
        self.non_ranking_loss_coeff = non_ranking_loss_coeff
        self.min_model_occurrences = min_model_occurrences
        self.preprocessor = SimpleScoringPreprocessor(min_model_occurrences=min_model_occurrences)

        self._history_entries: list[TrainingHistoryEntry] = []
        self._model_encoder: StringEncoder | None = None
        self._ratings: np.ndarray | None = None  # [num_models]
        self.last_timer: Timer | None = None

    def get_config_for_wandb(self) -> dict[str, Any]:
        return {
            "model_type": "elo_scoring",
            "initial_rating": self.initial_rating,
            "k_factor": self.k_factor,
            "balance_model_samples": self.balance_model_samples,
            "tie_both_bad_epsilon": self.tie_both_bad_epsilon,
            "non_ranking_loss_coeff": self.non_ranking_loss_coeff,
            "min_model_occurrences": self.min_model_occurrences,
            "num_models": self._model_encoder.size if self._model_encoder else 0,
        }

    @property
    def ratings(self) -> np.ndarray:
        """Get current ratings array."""
        if self._ratings is None:
            raise RuntimeError("Ratings not initialized")
        return self._ratings

    def train(
        self,
        data: TrainingData,
        validation_split: ValidationSplit | None = None,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        """
        Train the model using ELO rating updates.
        
        Args:
            data: Training data containing pairwise comparisons
            validation_split: Configuration for train/val split (if None, no validation)
            epochs: Number of passes through the training data
            batch_size: Not used in ELO (kept for API consistency)
        """
        if validation_split is None:
            validation_split = ValidationSplit(val_fraction=0)

        with Timer("train", verbosity="start+end") as train_timer:
            self.last_timer = train_timer
            
            with Timer("preprocess", verbosity="start+end", parent=train_timer):
                preprocessed_data = self._initialize_and_preprocess(data)
            
            with Timer("split_data", verbosity="start+end", parent=train_timer):
                preprocessed_train, preprocessed_val = split_simple_scoring_preprocessed_data(
                    preprocessed_data,
                    val_fraction=validation_split.val_fraction,
                    seed=validation_split.seed,
                )
            
            with Timer("epochs", verbosity="start+end", parent=train_timer) as epochs_timer:
                for epoch in range(1, epochs + 1):
                    result = self._train_epoch(epoch, preprocessed_train, preprocessed_val, epochs_timer)
                    
                    self._log_epoch_result(result)
            
            self.finish_wandb_if_needed()

    def predict(
        self,
        X: InputData,
        batch_size: int = 32,
    ) -> PromptRoutingOutput:
        """
        Predict best model for each prompt based on learned ELO ratings.
        
        Since this model doesn't look at prompts, it will always return
        the same ranking of models regardless of the prompt.
        
        Args:
            X: Input data containing prompts and model names
            batch_size: Not used (kept for API consistency)
            
        Returns:
            PromptRoutingOutput with scores for each model across all prompts
        """
        if self._model_encoder is None or self._ratings is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        # Get ratings for all requested models
        n_prompts = len(X.prompts)
        scores_dict = {}

        encoded_model_names, known_model_names = self._model_encoder.encode_known(X.model_names)
        
        for model_name, model_id in zip(known_model_names, encoded_model_names):
            # Convert ELO ratings to scores (normalize around 0)
            rating = self._ratings[model_id]
            score = rating - self.initial_rating
            scores_dict[model_name] = np.full(n_prompts, score)

        # Unknown models get rating of 0 (equivalent to initial_rating)
        for model_name in X.model_names:
            if model_name not in known_model_names:
                scores_dict[model_name] = np.zeros(n_prompts)
        
        return PromptRoutingOutput(_scores=scores_dict)

    def get_history(self) -> TrainingHistory:
        """Get training history."""
        return TrainingHistory.from_entries(self._history_entries)

    def get_all_model_scores(self) -> dict[str, float]:
        """
        Get ELO ratings for all known models.
        
        Returns:
            Dictionary mapping model names to their learned ELO ratings
        """
        if self._model_encoder is None or self._ratings is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        return {
            self._model_encoder.decode(i): float(self._ratings[i] - self.initial_rating)
            for i in range(self._model_encoder.size)
        }

    def get_state_dict(self) -> dict[str, Any]:
        """
        Get state dictionary for saving the model.
        
        Returns:
            State dictionary containing all model parameters and configuration
        """
        if self._ratings is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        if self._model_encoder is None:
            raise RuntimeError("Model encoder not initialized")
        
        return {
            "initial_rating": self.initial_rating,
            "k_factor": self.k_factor,
            "balance_model_samples": self.balance_model_samples,
            "print_every": self.print_every,
            "tie_both_bad_epsilon": self.tie_both_bad_epsilon,
            "non_ranking_loss_coeff": self.non_ranking_loss_coeff,
            "min_model_occurrences": self.min_model_occurrences,
            "ratings": self._ratings.tolist(),
            "model_encoder": self._model_encoder.get_state_dict(),
            "history_entries": self._history_entries,
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "EloScoringModel":
        """
        Load model from state dictionary.
        
        Args:
            state_dict: State dictionary from get_state_dict()
            
        Returns:
            Loaded model instance
        """
        model = cls(
            initial_rating=state_dict["initial_rating"],
            k_factor=state_dict["k_factor"],
            balance_model_samples=state_dict["balance_model_samples"],
            print_every=state_dict["print_every"],
            tie_both_bad_epsilon=state_dict["tie_both_bad_epsilon"],
            non_ranking_loss_coeff=state_dict["non_ranking_loss_coeff"],
            min_model_occurrences=state_dict["min_model_occurrences"],
        )
        
        model._model_encoder = StringEncoder.load_state_dict(state_dict["model_encoder"])
        model._ratings = np.array(state_dict["ratings"])
        model._history_entries = state_dict["history_entries"]
        
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

        self.init_wandb_if_needed()
        
        if self._ratings is None:
            self._ratings = np.full(self._model_encoder.size, self.initial_rating, dtype=np.float64)
        
        return preprocessed_data        

    def _train_epoch(
        self, 
        epoch: int,
        train_data: PreprocessedTrainingData,
        val_data: PreprocessedTrainingData | None,
        epochs_timer: Timer,
    ) -> "EloScoringModel.EpochResult":
        """
        Train for one epoch by updating ELO ratings based on comparisons.
        
        Args:
            epoch: Current epoch number
            train_data: Training data
            val_data: Validation data (optional)
            epochs_timer: Timer for tracking performance
            
        Returns:
            EpochResult with metrics for this epoch
        """
        with Timer(f"epoch_{epoch}", verbosity="start+end", parent=epochs_timer) as timer:
            # Shuffle comparisons if balancing is enabled
            comparisons = train_data.comparisons.copy()
            if self.balance_model_samples:
                # Create weighted sampling
                weights = self._compute_comparison_weights(comparisons)
                indices = np.random.choice(
                    len(comparisons),
                    size=len(comparisons),
                    replace=True,
                    p=weights / weights.sum(),
                )
                comparisons = [comparisons[i] for i in indices]
            else:
                np.random.shuffle(comparisons)
            
            # Track metrics
            total_rating_changes = 0.0
            total_accuracy = 0.0
            n_comparisons = len(comparisons)
            
            # Component tracking
            n_ranking = 0
            n_ties = 0
            n_both_bad = 0
            ranking_accuracy_sum = 0.0
            tie_accuracy_sum = 0.0
            both_bad_accuracy_sum = 0.0
            
            # Update ratings based on each comparison
            for comparison in comparisons:
                model_id_a = comparison.model_id_a
                model_id_b = comparison.model_id_b
                comparison_type = comparison.comparison_type
                
                rating_a: float = self._ratings[model_id_a]
                rating_b: float = self._ratings[model_id_b]
                
                # Compute expected scores using ELO formula
                expected_a = self._expected_score(rating_a, rating_b)
                expected_b = 1.0 - expected_a
                
                # Determine actual scores and apply penalties
                if comparison_type == "model_a_wins":
                    actual_a, actual_b = 1.0, 0.0
                    n_ranking += 1
                    if rating_a > rating_b:
                        ranking_accuracy_sum += 1.0
                elif comparison_type == "model_b_wins":
                    actual_a, actual_b = 0.0, 1.0
                    n_ranking += 1
                    if rating_b > rating_a:
                        ranking_accuracy_sum += 1.0
                elif comparison_type == "tie":
                    actual_a, actual_b = 0.5, 0.5
                    n_ties += 1
                    # Both should have positive scores (above initial)
                    if rating_a > self.initial_rating and rating_b > self.initial_rating:
                        tie_accuracy_sum += 1.0
                    # Apply penalty if ratings are below threshold
                    if rating_a < self.initial_rating + self.tie_both_bad_epsilon:
                        actual_a += self.non_ranking_loss_coeff * 0.5
                    if rating_b < self.initial_rating + self.tie_both_bad_epsilon:
                        actual_b += self.non_ranking_loss_coeff * 0.5
                elif comparison_type == "both_bad":
                    actual_a, actual_b = 0.5, 0.5
                    n_both_bad += 1
                    # Both should have negative scores (below initial)
                    if rating_a < self.initial_rating and rating_b < self.initial_rating:
                        both_bad_accuracy_sum += 1.0
                    # Apply penalty if ratings are above threshold
                    if rating_a > self.initial_rating - self.tie_both_bad_epsilon:
                        actual_a -= self.non_ranking_loss_coeff * 0.5
                    if rating_b > self.initial_rating - self.tie_both_bad_epsilon:
                        actual_b -= self.non_ranking_loss_coeff * 0.5
                else:
                    raise ValueError(f"Unknown comparison type: {comparison_type}")
                
                # Update ratings
                delta_a = self.k_factor * (actual_a - expected_a)
                delta_b = self.k_factor * (actual_b - expected_b)
                
                self._ratings[model_id_a] += delta_a
                self._ratings[model_id_b] += delta_b
                
                total_rating_changes += abs(delta_a) + abs(delta_b)
            
            # Compute average metrics
            avg_rating_change = total_rating_changes / n_comparisons if n_comparisons > 0 else 0.0
            
            # Compute component accuracies
            ranking_accuracy = ranking_accuracy_sum / n_ranking if n_ranking > 0 else 0.0
            tie_accuracy = tie_accuracy_sum / n_ties if n_ties > 0 else 0.0
            both_bad_accuracy = both_bad_accuracy_sum / n_both_bad if n_both_bad > 0 else 0.0
            
            # Overall accuracy
            total_accuracy = (ranking_accuracy_sum + tie_accuracy_sum + both_bad_accuracy_sum) / n_comparisons
            
            # Compute rating statistics
            rating_stats = self._compute_rating_statistics()
            
            # Validation
            with Timer("perform_validation", verbosity="start+end", parent=timer):
                if val_data is not None:
                    val_accuracy, val_accuracy_components = self._perform_validation(val_data)
                else:
                    val_accuracy = None
                    val_accuracy_components = {}
            
            # Build additional metrics
            additional_metrics = {
                "avg_rating_change": avg_rating_change,
                "avg_rating": rating_stats["avg_rating"],
                "top_10_pct_rating": rating_stats["top_10_pct_rating"],
                "bottom_10_pct_rating": rating_stats["bottom_10_pct_rating"],
                "ranking_accuracy": ranking_accuracy,
                "tie_accuracy": tie_accuracy,
                "both_bad_accuracy": both_bad_accuracy,
            }
            
            # Add validation components with "val_" prefix
            for key, value in val_accuracy_components.items():
                additional_metrics[f"val_{key}"] = value
            
            entry = TrainingHistoryEntry(
                epoch=epoch,
                total_loss=None,  # ELO doesn't have a loss
                val_loss=None,
                train_accuracy=total_accuracy,
                val_accuracy=val_accuracy,
                additional_metrics=additional_metrics,
            )
            self._history_entries.append(entry)
            
            if self.wandb_details is not None:
                self.log_to_wandb(entry)
            
        return self.EpochResult(
            epoch=epoch,
            train_accuracy=total_accuracy,
            val_accuracy=val_accuracy,
            avg_rating_change=avg_rating_change,
            duration=timer.elapsed_time,
            additional_metrics=additional_metrics,
        )

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Compute expected score for model A using ELO formula.
        
        Args:
            rating_a: Rating of model A
            rating_b: Rating of model B
            
        Returns:
            Expected score for model A (between 0 and 1)
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _compute_comparison_weights(self, comparisons: list) -> np.ndarray:
        """
        Compute weights for balanced sampling of comparisons.
        
        Args:
            comparisons: List of comparison objects
            
        Returns:
            Array of weights for each comparison  # [n_comparisons]
        """
        from collections import Counter
        
        # Count how many times each model appears
        model_counts = Counter()
        for comparison in comparisons:
            model_counts[comparison.model_id_a] += 1
            model_counts[comparison.model_id_b] += 1
        
        # Compute weight for each model (inverse frequency)
        model_weights = {
            model_id: 1.0 / count
            for model_id, count in model_counts.items()
        }
        
        # For each comparison, assign weight based on the rarest model
        weights = np.array([
            max(model_weights[comp.model_id_a], model_weights[comp.model_id_b])
            for comp in comparisons
        ])
        
        return weights

    def _compute_rating_statistics(self) -> dict[str, float]:
        """
        Compute statistics about the current model ratings.
        
        Returns:
            Dictionary with avg_rating, top_10_pct_rating, bottom_10_pct_rating
        """
        if self._ratings is None:
            return {"avg_rating": 0.0, "top_10_pct_rating": 0.0, "bottom_10_pct_rating": 0.0}
        
        # Convert to scores (relative to initial rating)
        scores = self._ratings - self.initial_rating
        
        avg_rating = float(np.mean(scores))
        
        # Compute top 10% and bottom 10%
        n_models = len(scores)
        top_k = max(1, n_models // 10)
        bottom_k = max(1, n_models // 10)
        
        sorted_scores = np.sort(scores)
        top_10_pct_rating = float(np.mean(sorted_scores[-top_k:]))
        bottom_10_pct_rating = float(np.mean(sorted_scores[:bottom_k]))
        
        return {
            "avg_rating": avg_rating,
            "top_10_pct_rating": top_10_pct_rating,
            "bottom_10_pct_rating": bottom_10_pct_rating,
        }

    def _perform_validation(
        self,
        val_data: PreprocessedTrainingData,
    ) -> tuple[float, dict[str, float]]:
        """
        Perform validation on validation data.
        
        Args:
            val_data: Validation data
            
        Returns:
            Tuple of (accuracy, accuracy_components)
        """
        total_accuracy = 0.0
        n_comparisons = len(val_data.comparisons)
        
        # Component tracking
        n_ranking = 0
        n_ties = 0
        n_both_bad = 0
        ranking_accuracy_sum = 0.0
        tie_accuracy_sum = 0.0
        both_bad_accuracy_sum = 0.0
        
        for comparison in val_data.comparisons:
            model_id_a = comparison.model_id_a
            model_id_b = comparison.model_id_b
            comparison_type = comparison.comparison_type
            
            rating_a = self._ratings[model_id_a]
            rating_b = self._ratings[model_id_b]
            
            # Check accuracy based on comparison type
            if comparison_type == "model_a_wins":
                n_ranking += 1
                if rating_a > rating_b:
                    ranking_accuracy_sum += 1.0
            elif comparison_type == "model_b_wins":
                n_ranking += 1
                if rating_b > rating_a:
                    ranking_accuracy_sum += 1.0
            elif comparison_type == "tie":
                n_ties += 1
                if rating_a > self.initial_rating and rating_b > self.initial_rating:
                    tie_accuracy_sum += 1.0
            elif comparison_type == "both_bad":
                n_both_bad += 1
                if rating_a < self.initial_rating and rating_b < self.initial_rating:
                    both_bad_accuracy_sum += 1.0
        
        # Compute component accuracies
        ranking_accuracy = ranking_accuracy_sum / n_ranking if n_ranking > 0 else 0.0
        tie_accuracy = tie_accuracy_sum / n_ties if n_ties > 0 else 0.0
        both_bad_accuracy = both_bad_accuracy_sum / n_both_bad if n_both_bad > 0 else 0.0
        
        # Overall accuracy
        total_accuracy = (ranking_accuracy_sum + tie_accuracy_sum + both_bad_accuracy_sum) / n_comparisons if n_comparisons > 0 else 0.0
        
        accuracy_components = {
            "ranking_accuracy": ranking_accuracy,
            "tie_accuracy": tie_accuracy,
            "both_bad_accuracy": both_bad_accuracy,
        }
        
        return total_accuracy, accuracy_components

    def _log_epoch_result(self, result: "EloScoringModel.EpochResult") -> None:
        """Log epoch results to console."""
        if self.print_every is None:
            return
        
        if not result.epoch % self.print_every == 0:
            return
        
        metrics = result.additional_metrics
        rating_str = f"ratings: avg={metrics['avg_rating']:.1f}, top10%={metrics['top_10_pct_rating']:.1f}, bot10%={metrics['bottom_10_pct_rating']:.1f}"
        
        if result.val_accuracy is None:
            print(f"Epoch {result.epoch:>4}: accuracy = {(result.train_accuracy*100):.2f}%, {rating_str}, Δrating={result.avg_rating_change:.2f} - {result.duration:.2f}s")
        else:
            print(f"Epoch {result.epoch:>4}: accuracy = {(result.train_accuracy*100):.2f}%/{(result.val_accuracy*100):.2f}%, {rating_str}, Δrating={result.avg_rating_change:.2f} - {result.duration:.2f}s")

    @dataclass
    class EpochResult:
        """Results from training one epoch."""
        epoch: int
        train_accuracy: float
        val_accuracy: float | None
        avg_rating_change: float
        duration: float
        additional_metrics: dict[str, float]

