"""Min-Cost-Max-Flow scoring model that learns per-model scores using MCMF algorithm."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import networkx as nx
from collections import deque
import warnings

from src.models.model_base import ModelBase
from src.data_models.data_models import TrainingData, InputData
from src.data_models.dense_network_types import PromptRoutingOutput
from src.data_models.simple_scoring_types import PreprocessedTrainingData
from src.preprocessing.simple_scoring_preprocessor import SimpleScoringPreprocessor
from src.preprocessing.utils import filter_out_rare_models, filter_out_empty_entries, filter_out_both_bad, filter_out_ties, create_encoder
from src.utils.training_history import TrainingHistory
from src.utils.wandb_details import WandbDetails
from src.utils.string_encoder import StringEncoder
from src.utils.timer import Timer
from src.utils.data_split import ValidationSplit
from src.analysis import exploration


class McmfScoringModel(ModelBase):
    """
    Min-Cost-Max-Flow based model that learns scores for each LLM.
    
    This model uses a min-cost-max-flow algorithm to compute model scores based on
    pairwise comparisons. The network is constructed with:
    - Source and sink nodes
    - One node per model
    - Edges from source to models (capacity = wins, cost = 0)
    - Edges from models to sink (capacity = losses, cost = 0)
    - Edges between models (capacity = times i beat j, cost = 1)
    
    Scores are derived from node potentials after solving the MCMF problem.
    This model ignores prompts, ties, and both_bad comparisons.
    """
    
    def __init__(
        self,
        min_model_occurrences: int = 1000,
        print_summary: bool = True,
        wandb_details: WandbDetails | None = None,
    ) -> None:
        """
        Initialize MCMF scoring model.
        
        Args:
            min_model_occurrences: Minimum number of times a model must appear to be included
            print_summary: Whether to print a summary after computing scores
            wandb_details: Weights & Biases configuration
        """
        if wandb_details is not None:
            warnings.warn("McmfScoringModel does not use wandb, so wandb_details will be ignored")
            wandb_details = None

        super().__init__(wandb_details)
        self.min_model_occurrences = min_model_occurrences
        self.print_summary = print_summary

        self._model_encoder: StringEncoder | None = None
        self._scores: np.ndarray | None = None  # [num_models] - scores for each model
        self.last_timer: Timer | None = None

    def get_config_for_wandb(self) -> dict[str, Any]:
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
                preprocessed_data, self._model_encoder = self._filter_and_fit_encoder(data)
            
            with Timer("compute_mcmf_scores", verbosity="start+end", parent=train_timer):
                metrics = self._compute_mcmf_scores(preprocessed_data)
            
            if self.print_summary:
                self._print_summary(metrics)

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
        Get MCMF-derived scores for all known models.
        
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
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "McmfScoringModel":
        model = cls(
            min_model_occurrences=state_dict["min_model_occurrences"],
            print_summary=state_dict["print_summary"],
        )
        
        model._model_encoder = StringEncoder.load_state_dict(state_dict["model_encoder"])
        model._scores = np.array(state_dict["scores"])
        
        return model

    def _filter_and_fit_encoder(self, data: TrainingData) -> tuple[TrainingData, StringEncoder]:
        filtered_data = filter_out_rare_models(data, self.min_model_occurrences)
        filtered_data = filter_out_empty_entries(filtered_data)
        filtered_data = filter_out_both_bad(filtered_data)
        filtered_data = filter_out_ties(filtered_data)
        if len(filtered_data.entries) == 0:
            raise ValueError(
                "No valid training data after filtering. "
                f"All models were filtered out (min_model_occurrences={self.min_model_occurrences}). "
                "Try lowering min_model_occurrences or providing more training data."
            )

        model_encoder = create_encoder(filtered_data)
        
        return filtered_data, model_encoder

    def _compute_mcmf_scores(self, filtered_data: TrainingData) -> "McmfScoringModel.TrainMetrics":
        """
        Compute scores using min-cost-max-flow algorithm.
        
        Args:
            preprocessed_data: Preprocessed training data
        """
        if self._model_encoder is None:
            raise RuntimeError("Model encoder not initialized")
        
        G, flow_dict = self._run_min_cost_max_flow(filtered_data)
        self._scores = self._compute_scores(G, flow_dict)

        # All edges between models have cost = 1, so we can just sum the flow values
        flow_cost = sum([flow_dict.get(u, {}).get(v, 0) for u in range(self._model_encoder.size) for v in range(self._model_encoder.size)])
        
        return self.TrainMetrics(
            accuracy=self._compute_accuracy(filtered_data, self._scores),
            min_score=np.min(self._scores),
            max_score=np.max(self._scores),
            avg_score=np.mean(self._scores),
            top_10_pct_score=np.percentile(self._scores, 90),
            bottom_10_pct_score=np.percentile(self._scores, 10),
            flow_cost=flow_cost,
            # Sum of edge capacities is equal to the number of comparisons
            # Sum of flow values represents the number of comparisons excluding cycles (a beats b, b beats c, c beats a)
            pct_comparisons_used_by_flow=flow_cost / len(filtered_data.entries))

    def _compute_scores(self, G: nx.DiGraph, flow_dict: dict[int, dict[int, int]]) -> np.ndarray:
        # TODO: Implement this
        return np.zeros(self._model_encoder.size)

    def _compute_accuracy(self, data: TrainingData, scores: np.ndarray) -> float:
        correct = 0
        for entry in data.entries:
            model_a_id = self._model_encoder.encode(entry.model_a)
            model_b_id = self._model_encoder.encode(entry.model_b)
            if entry.winner == "model_a":
                if scores[model_a_id] > scores[model_b_id]:
                    correct += 1
            elif entry.winner == "model_b":
                if scores[model_b_id] > scores[model_a_id]:
                    correct += 1

        return correct / len(data.entries)

    def _run_min_cost_max_flow(self, filtered_data: TrainingData) -> tuple[nx.DiGraph, dict[int, dict[int, int]]]:
        num_models = self._model_encoder.size

        wins_matrix, _ = exploration.get_wins_matrix(filtered_data, self._model_encoder) # [num_models, num_models]
        wins_by_model: np.ndarray = np.sum(wins_matrix, axis=1)  # [num_models]
        losses_by_model: np.ndarray = np.sum(wins_matrix, axis=0)  # [num_models]
        
        source = num_models
        sink = num_models + 1
        
        G = nx.DiGraph()
        G.add_node(source)
        G.add_node(sink)
        for model_id in range(num_models):
            G.add_node(model_id)
        
        for model_id in range(num_models):
            # Add edges from source to models (capacity = wins, cost = 0)
            if wins_by_model[model_id] > 0:
                G.add_edge(source, model_id, capacity=int(wins_by_model[model_id]), weight=0)
            # Add edges from models to sink (capacity = losses, cost = 0)
            if losses_by_model[model_id] > 0:
                G.add_edge(model_id, sink, capacity=int(losses_by_model[model_id]), weight=0)            
        
        # Add edges between models (capacity = times i beat j, cost = 1)
        for winner_id in range(num_models):
            for loser_id in range(num_models):
                if wins_matrix[winner_id, loser_id] > 0:
                    G.add_edge(winner_id, loser_id, capacity=int(wins_matrix[winner_id, loser_id]), weight=1)
        
        flow_dict = nx.max_flow_min_cost(G, source, sink)
        return G, flow_dict

    def _print_summary(self, metrics: "McmfScoringModel.TrainMetrics") -> None:
        print(f"Accuracy: {metrics.accuracy*100:.2f}%")
        print(f"Flow cost: {metrics.flow_cost:.0f}")
        print(f"Comparisons used: {metrics.pct_comparisons_used_by_flow*100:.2f}%")
        print(f"Scores: {metrics.min_score:.2f}/{metrics.bottom_10_pct_score:.2f}/{metrics.avg_score:.2f}/{metrics.top_10_pct_score:.2f}/{metrics.max_score:.2f}")

    @dataclass
    class TrainMetrics:
        accuracy: float
        min_score: float
        max_score: float
        avg_score: float
        top_10_pct_score: float
        bottom_10_pct_score: float
        flow_cost: float
        pct_comparisons_used_by_flow: float
