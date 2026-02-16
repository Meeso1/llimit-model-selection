"""Min-Cost-Max-Flow scoring model that learns per-model scores using MCMF algorithm."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import networkx as nx

from src.models.scoring_model_base import ScoringModelBase
from src.data_models.data_models import TrainingData, InputData
from src.data_models.dense_network_types import PromptRoutingOutput
from src.preprocessing.utils import filter_out_rare_models, filter_out_empty_entries, filter_out_both_bad, filter_out_ties, create_encoder
from src.utils import accuracy
from src.utils.training_history import TrainingHistory
from src.utils.model_scores_stats import compute_model_scores_stats
from src.utils.string_encoder import StringEncoder
from src.utils.timer import Timer
from src.utils.data_split import ValidationSplit
from src.analysis import exploration


class McmfScoringModel(ScoringModelBase):
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
        run_name: str | None = None,
    ) -> None:
        """
        Initialize MCMF scoring model.
        
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
                preprocessed_data, self._model_encoder, _ = self._filter_and_fit_encoder(data)
            
            with Timer("compute_mcmf_scores", verbosity="start+end", parent=train_timer):
                metrics = self._compute_mcmf_scores(preprocessed_data)
            
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
            accuracy=accuracy.compute_comparisons_accuracy(filtered_data, self._scores, self._model_encoder),
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
        """
        Compute scores from node potentials using Bellman-Ford on residual graph.
        
        Args:
            G: Original graph with capacities and weights
            flow_dict: Flow solution from min cost max flow
            
        Returns:
            scores: [num_models] - normalized scores for each model
        """
        num_models = self._model_encoder.size
        residual = self._build_residual_graph(G, flow_dict)
        
        scores = np.full(num_models, np.inf)
        
        # Find a starting model node
        start_node = self._find_starting_node_from(range(num_models), residual)        
        if start_node is None:
            raise ValueError("No starting node found in residual graph")
        
        # Run Bellman-Ford from the starting node to get potentials
        potentials = self._run_bellman_ford(residual, start_node)
        for model_id in range(num_models):
            scores[model_id] = -potentials.get(model_id, np.inf)
        
        # Handle disconnected components
        unreachable = [i for i in range(num_models) if scores[i] == np.inf]
        while unreachable:
            component_start = self._find_starting_node_from(unreachable, residual)
            if component_start is None:
                # Remaining unreachable nodes don't exist in residual, set to 0
                for node in unreachable:
                    scores[node] = 0.0
                break
            
            component_potentials = self._run_bellman_ford(residual, component_start)
            for model_id in unreachable[:]:
                if model_id in component_potentials:
                    scores[model_id] = -component_potentials[model_id]
                    unreachable.remove(model_id)
        
        scores = self._normalize_scores(scores)
        return scores
    
    def _find_starting_node_from(self, nodes: list[int], residual: nx.DiGraph) -> int:
        for node in nodes:
            if node in residual:
                return node
            
        return None
    
    def _build_residual_graph(
        self,
        G: nx.DiGraph,
        flow_dict: dict[int, dict[int, int]]
    ) -> nx.DiGraph:
        """
        Build residual graph from original graph and flow solution.
        
        In the residual graph:
        - If there's remaining capacity on edge (u, v), add edge (u, v) with cost = weight
        - If there's flow on edge (u, v), add reverse edge (v, u) with cost = -weight
        
        Args:
            G: Original graph with capacities and weights
            flow_dict: Flow solution from min cost max flow
            
        Returns:
            Residual graph
        """
        residual = nx.DiGraph()
        
        for u, v, edge_data in G.edges(data=True):
            # Ignore edges between source/sink and model nodes
            if u >= self._model_encoder.size or v >= self._model_encoder.size:
                continue
            
            capacity = edge_data['capacity']
            weight = edge_data['weight']
            flow = flow_dict.get(u, {}).get(v, 0)
            
            # Forward edge for remaining capacity
            remaining_capacity = capacity - flow
            if remaining_capacity > 0:
                residual.add_edge(u, v, weight=weight)
            
            # Backward edge for used flow
            if flow > 0:
                residual.add_edge(v, u, weight=-weight)
                
        return residual
    
    def _run_bellman_ford(self, G: nx.DiGraph, source: int) -> dict[int, float]:
        """
        Run single-source Bellman-Ford algorithm to find shortest paths.
        
        Args:
            G: Graph to run algorithm on
            source: Source node
            
        Returns:
            Dictionary mapping nodes to their distances from source
        """
        try:
            distances = nx.single_source_bellman_ford_path_length(G, source)
            return distances
        except nx.NetworkXError:
            # No paths from source
            return {source: 0.0}
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to have mean 0 and standard deviation 1.
        
        Args:
            scores: [num_models] - unnormalized scores
            
        Returns:
            normalized_scores: [num_models] - normalized scores
        """
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std < 1e-8:
            # All scores are the same, return zeros
            return np.zeros_like(scores)
        
        return (scores - mean) / std

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


    def _log_metrics(self, metrics: "McmfScoringModel.TrainMetrics") -> None:
        self.init_logger_if_needed()
        final_metrics = compute_model_scores_stats(self.get_all_model_scores())
        final_metrics.update(metrics)
        self.finish_logger_if_needed(final_metrics=final_metrics)

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
