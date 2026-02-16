"""Least squares scoring model that learns per-model scores using least squares algorithm."""

from dataclasses import dataclass
from typing import Any
import numpy as np

from src.models.scoring_model_base import ScoringModelBase
from src.data_models.data_models import TrainingData, InputData
from src.data_models.dense_network_types import PromptRoutingOutput
from src.preprocessing.utils import filter_out_rare_models, filter_out_empty_entries, filter_out_both_bad, filter_out_ties, create_encoder
from src.utils import accuracy
from src.utils.training_history import TrainingHistory
from src.utils.model_scores_stats import compute_model_scores_stats
from src.utils.string_encoder import StringEncoder
from src.utils.timer import Timer
from src.utils.data_split import ValidationSplit, train_val_split
from src.analysis import exploration


class LeastSquaresScoringModel(ScoringModelBase):
    """
    Least squares based model that learns scores for each LLM.
    
    This model uses a least squares algorithm to compute model scores based on
    pairwise comparisons.
    """
    
    def __init__(
        self,
        min_model_occurrences: int = 1000,
        print_summary: bool = True,
        run_name: str | None = None,
    ) -> None:
        """
        Initialize Least squares scoring model.
        
        Args:
            min_model_occurrences: Minimum number of times a model must appear to be included
            print_summary: Whether to print a summary after computing scores
            run_name: Name for logging this training run
        """
        super().__init__(run_name)
        self.min_model_occurrences = min_model_occurrences
        self.print_summary = print_summary

        self._model_encoder: StringEncoder | None = None
        self._scores: np.ndarray | None = None  # [num_models] - scores for each model
        self.last_timer: Timer | None = None

    def get_config_for_logging(self) -> dict[str, Any]:
        return {}

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
        batch_size: int = 0,
    ) -> None:
        with Timer("train", verbosity="start+end") as train_timer:
            self.last_timer = train_timer
            
            with Timer("preprocess", verbosity="start+end", parent=train_timer):
                filtered_data, self._model_encoder, _ = self._filter_and_fit_encoder(data)
            
            validation_split = validation_split or ValidationSplit(val_fraction=0, seed=42)
            train_data, val_data = train_val_split(
                filtered_data,
                val_fraction=validation_split.val_fraction,
                seed=validation_split.seed
            )
            
            with Timer("compute_scores", verbosity="start+end", parent=train_timer):
                metrics = self._compute_least_squares_scores(train_data, val_data)
            
            if self.print_summary:
                self._print_summary(metrics)
            
            self._log_metrics(metrics)

    def predict(
        self,
        X: InputData,
        batch_size: int = 32,
    ) -> PromptRoutingOutput:
        if self._model_encoder is None or self._scores is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        n_prompts = len(X.prompts)
        scores_dict = {}

        encoded_model_names, known_model_names = self._model_encoder.encode_known(X.model_names)
        
        for model_name, model_id in zip(known_model_names, encoded_model_names):
            score = self._scores[model_id]
            scores_dict[model_name] = np.full(n_prompts, score)

        # Unknown models get score of 0
        for model_name in X.model_names:
            if model_name not in known_model_names:
                scores_dict[model_name] = np.zeros(n_prompts)
        
        return PromptRoutingOutput(_scores=scores_dict)

    def get_history(self) -> TrainingHistory:
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

    def get_state_dict(self) -> dict[str, Any]:
        if self._scores is None or self._model_encoder is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        return {
            "min_model_occurrences": self.min_model_occurrences,
            "print_summary": self.print_summary,
            "scores": self._scores.tolist(),
            "model_encoder": self._model_encoder.get_state_dict(),
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "LeastSquaresScoringModel":
        model = cls(
            min_model_occurrences=state_dict["min_model_occurrences"],
            print_summary=state_dict["print_summary"],
        )
        
        model._model_encoder = StringEncoder.load_state_dict(state_dict["model_encoder"])
        model._scores = np.array(state_dict["scores"])
        
        return model

    def _filter_and_fit_encoder(self, data: TrainingData) -> tuple[TrainingData, StringEncoder, list[int]]:
        filtered_data, indexes = filter_out_rare_models(data, self.min_model_occurrences)
        filtered_data, indexes = filter_out_empty_entries(filtered_data, indexes)
        filtered_data, indexes = filter_out_both_bad(filtered_data, indexes)
        filtered_data, indexes = filter_out_ties(filtered_data, indexes)
        if len(filtered_data.entries) == 0:
            raise ValueError(
                "No valid training data after filtering. "
                f"All models were filtered out (min_model_occurrences={self.min_model_occurrences}). "
                "Try lowering min_model_occurrences or providing more training data."
            )

        model_encoder = create_encoder(filtered_data)
        
        return filtered_data, model_encoder, indexes

    def _compute_least_squares_scores(
        self,
        train_data: TrainingData,
        val_data: TrainingData | None
    ) -> "LeastSquaresScoringModel.TrainMetrics":
        """
        Compute scores using least squares algorithm on train data.
        Evaluate metrics on both train and validation data.
        
        Fits scores that minimize squared error between score differences and win rate differences.
        The optimization problem: minimize Σ_{pairs (i,j)} (s_i - s_j - target_ij)²
        where target_ij = win_rate_ij - win_rate_ji
        
        Args:
            train_data: Training data to fit scores on
            val_data: Optional validation data to evaluate metrics on
        """
        if self._model_encoder is None:
            raise RuntimeError("Model encoder not initialized")
        
        train_wins_matrix, _ = exploration.get_wins_matrix(train_data, self._model_encoder)  # [num_models, num_models]
        
        L, b = self._build_laplacian_system(train_wins_matrix)
        self._scores, lstsq_residual = self._solve_least_squares(L, b)
        
        train_error_metrics = self._compute_error_metrics(train_wins_matrix, self._scores)
        
        train_metrics = {
            'accuracy': accuracy.compute_comparisons_accuracy(train_data, self._scores, self._model_encoder),
            'total_squared_error': train_error_metrics['total_squared_error'],
            'mean_squared_error': train_error_metrics['mean_squared_error'],
            'rmse': train_error_metrics['rmse'],
            'num_compared_pairs': train_error_metrics['num_compared_pairs'],
        }
        
        val_metrics = None
        if val_data is not None and len(val_data.entries) > 0:
            val_wins_matrix, _ = exploration.get_wins_matrix(val_data, self._model_encoder)
            val_error_metrics = self._compute_error_metrics(val_wins_matrix, self._scores)
            
            val_metrics = {
                'accuracy': accuracy.compute_comparisons_accuracy(val_data, self._scores, self._model_encoder),
                'total_squared_error': val_error_metrics['total_squared_error'],
                'mean_squared_error': val_error_metrics['mean_squared_error'],
                'rmse': val_error_metrics['rmse'],
                'num_compared_pairs': val_error_metrics['num_compared_pairs'],
            }
        
        return self.TrainMetrics(
            min_score=np.min(self._scores),
            max_score=np.max(self._scores),
            avg_score=np.mean(self._scores),
            top_10_pct_score=np.percentile(self._scores, 90),
            bottom_10_pct_score=np.percentile(self._scores, 10),
            lstsq_residual=lstsq_residual,
            train_accuracy=train_metrics['accuracy'],
            train_total_squared_error=train_metrics['total_squared_error'],
            train_mean_squared_error=train_metrics['mean_squared_error'],
            train_rmse=train_metrics['rmse'],
            train_num_compared_pairs=train_metrics['num_compared_pairs'],
            val_accuracy=val_metrics['accuracy'] if val_metrics else None,
            val_total_squared_error=val_metrics['total_squared_error'] if val_metrics else None,
            val_mean_squared_error=val_metrics['mean_squared_error'] if val_metrics else None,
            val_rmse=val_metrics['rmse'] if val_metrics else None,
            val_num_compared_pairs=val_metrics['num_compared_pairs'] if val_metrics else None,
        )
    
    def _build_laplacian_system(self, wins_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the graph Laplacian system L @ s = b for least squares optimization.
        
        The system minimizes: Σ_{pairs (i,j)} (s_i - s_j - target_ij)²
        where target_ij = win_rate_ij - win_rate_ji
        
        Args:
            wins_matrix: [num_models, num_models] - wins_matrix[i, j] = times i beat j
            
        Returns:
            L: [num_models, num_models] - graph Laplacian matrix
            b: [num_models] - right-hand side vector
        """
        num_models = wins_matrix.shape[0]
        
        # Compute total comparisons and win rates
        total_comparisons = wins_matrix + wins_matrix.T  # [num_models, num_models]
        compared = total_comparisons > 0  # [num_models, num_models] - mask for pairs that compared
        
        # Win rates: win_rate[i,j] = wins[i,j] / total[i,j]
        win_rates = np.where(compared, wins_matrix / np.maximum(total_comparisons, 1), 0.5)
        
        # Target differences: target[i,j] = win_rate[i,j] - win_rate[j,i]
        # This is antisymmetric: target[j,i] = -target[i,j]
        targets = win_rates - win_rates.T  # [num_models, num_models]
        
        # Build graph Laplacian
        # L[i,i] = degree of node i (number of nodes i was compared with)
        # L[i,j] = -1 if i and j compared, 0 otherwise
        adjacency = compared.astype(float)
        np.fill_diagonal(adjacency, 0)  # Remove self-loops
        degree = np.diag(adjacency.sum(axis=1))  # Degree matrix
        L = degree - adjacency  # Laplacian
        
        # Right-hand side: b[i] = Σ_j target[i,j] for all j compared with i
        b = targets.sum(axis=1)  # [num_models]
        
        return L, b
    
    def _solve_least_squares(self, L: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Solve the least squares system L @ s = b.
        
        The Laplacian L is singular (constant vectors are in its null space),
        so we use pseudoinverse to get the minimum-norm solution.
        
        Args:
            L: [num_models, num_models] - Laplacian matrix
            b: [num_models] - right-hand side vector
            
        Returns:
            scores: [num_models] - fitted scores (normalized to mean 0, std 1)
            residual: sum of squared residuals from lstsq (before normalization)
        """
        # Solve using least squares (handles singular matrices)
        scores, residuals, rank, s_values = np.linalg.lstsq(L, b, rcond=None)
        
        # Extract residual (if available, otherwise compute it)
        if residuals.size > 0:
            residual = float(residuals[0])
        else:
            # Compute manually if not provided
            residual = float(np.sum((L @ scores - b) ** 2))
        
        scores = self._normalize_scores(scores)
        
        return scores, residual
    
    def _compute_error_metrics(self, wins_matrix: np.ndarray, scores: np.ndarray) -> dict[str, float]:
        """
        Compute error metrics for the fitted scores.
        
        Computes the objective function value: Σ_{pairs (i,j)} (s_i - s_j - target_ij)²
        where target_ij = win_rate_ij - win_rate_ji
        
        Args:
            wins_matrix: [num_models, num_models] - wins_matrix[i, j] = times i beat j
            scores: [num_models] - fitted scores
            
        Returns:
            Dictionary with error metrics:
                - total_squared_error: sum of squared errors
                - mean_squared_error: average squared error per compared pair
                - rmse: root mean squared error
                - num_compared_pairs: number of model pairs that were compared
        """
        num_models = wins_matrix.shape[0]
        
        # Compute total comparisons and win rates
        total_comparisons = wins_matrix + wins_matrix.T  # [num_models, num_models]
        compared = total_comparisons > 0  # [num_models, num_models]
        
        # Win rates: win_rate[i,j] = wins[i,j] / total[i,j]
        win_rates = np.where(compared, wins_matrix / np.maximum(total_comparisons, 1), 0.5)
        
        # Target differences: target[i,j] = win_rate[i,j] - win_rate[j,i]
        targets = win_rates - win_rates.T  # [num_models, num_models]
        
        # Score differences: diff[i,j] = scores[i] - scores[j]
        score_diffs = scores[:, None] - scores[None, :]  # [num_models, num_models]
        
        # Squared errors for compared pairs only
        errors = (score_diffs - targets) ** 2  # [num_models, num_models]
        
        # Only count upper triangle to avoid double-counting
        # (since error[i,j] = error[j,i] due to antisymmetry)
        upper_triangle = np.triu_indices(num_models, k=1)
        compared_upper = compared[upper_triangle]
        errors_upper = errors[upper_triangle]
        
        # Compute metrics
        num_compared_pairs = np.sum(compared_upper)
        total_squared_error = float(np.sum(errors_upper[compared_upper]))
        mean_squared_error = total_squared_error / max(num_compared_pairs, 1)
        rmse = np.sqrt(mean_squared_error)
        
        return {
            'total_squared_error': total_squared_error,
            'mean_squared_error': mean_squared_error,
            'rmse': rmse,
            'num_compared_pairs': int(num_compared_pairs),
        }
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std < 1e-8:
            # All scores are the same, return zeros
            return np.zeros_like(scores)
        
        return (scores - mean) / std

    def _print_summary(self, metrics: "LeastSquaresScoringModel.TrainMetrics") -> None:
        has_val = metrics.val_accuracy is not None \
            and metrics.val_num_compared_pairs is not None \
            and metrics.val_total_squared_error is not None \
            and metrics.val_mean_squared_error is not None \
            and metrics.val_rmse is not None
        
        if has_val:
            print(f"Accuracy: {metrics.train_accuracy*100:.2f}% / {metrics.val_accuracy*100:.2f}%")
            print(f"Compared pairs: {metrics.train_num_compared_pairs} / {metrics.val_num_compared_pairs}")
            print(f"Total squared error: {metrics.train_total_squared_error:.2f} / {metrics.val_total_squared_error:.2f}")
            print(f"Mean squared error: {metrics.train_mean_squared_error:.4f} / {metrics.val_mean_squared_error:.4f}")
            print(f"RMSE: {metrics.train_rmse:.4f} / {metrics.val_rmse:.4f}")
        else:
            print(f"Accuracy: {metrics.train_accuracy*100:.2f}%")
            print(f"Compared pairs: {metrics.train_num_compared_pairs}")
            print(f"Total squared error: {metrics.train_total_squared_error:.2f}")
            print(f"Mean squared error: {metrics.train_mean_squared_error:.4f}")
            print(f"RMSE: {metrics.train_rmse:.4f}")
        
        print(f"Lstsq residual: {metrics.lstsq_residual:.4f}")
        print(f"Scores: {metrics.min_score:.2f}/{metrics.bottom_10_pct_score:.2f}/{metrics.avg_score:.2f}/{metrics.top_10_pct_score:.2f}/{metrics.max_score:.2f}")

    def _log_metrics(self, metrics: "LeastSquaresScoringModel.TrainMetrics") -> None:
        self.init_logger_if_needed()
        final_metrics = compute_model_scores_stats(self.get_all_model_scores())
        final_metrics.update(metrics)
        self.finish_logger_if_needed(final_metrics=final_metrics)

    @dataclass
    class TrainMetrics:
        min_score: float
        max_score: float
        avg_score: float
        top_10_pct_score: float
        bottom_10_pct_score: float
        
        train_accuracy: float
        train_total_squared_error: float
        train_mean_squared_error: float
        train_rmse: float
        train_num_compared_pairs: int
        lstsq_residual: float
        
        val_accuracy: float | None = None
        val_total_squared_error: float | None = None
        val_mean_squared_error: float | None = None
        val_rmse: float | None = None
        val_num_compared_pairs: int | None = None
