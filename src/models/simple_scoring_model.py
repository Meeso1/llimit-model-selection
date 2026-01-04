"""Simple scoring model that learns per-model scores without looking at prompts."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from collections import Counter

from src.models.model_base import ModelBase
from src.data_models.data_models import TrainingData, InputData
from src.data_models.dense_network_types import PromptRoutingOutput
from src.data_models.simple_scoring_types import PreprocessedTrainingData
from src.preprocessing.simple_scoring_preprocessor import SimpleScoringPreprocessor
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.wandb_details import WandbDetails
from src.utils.string_encoder import StringEncoder
from src.utils.timer import Timer
from src.utils.accuracy import compute_pairwise_accuracy
from src.models.optimizers.optimizer_spec import OptimizerSpecification
from src.models.optimizers.adamw_spec import AdamWSpec
from src.utils.data_split import ValidationSplit, split_simple_scoring_preprocessed_data


class SimpleScoringModel(ModelBase):
    """
    Simple baseline model that learns a single score for each LLM.
    
    This model completely ignores prompts and responses, learning only
    how often each LLM beats other LLMs in pairwise comparisons.
    """
    
    def __init__(
        self,
        optimizer_spec: OptimizerSpecification | None = None,
        balance_model_samples: bool = True,
        print_every: int | None = 1,
        tie_both_bad_epsilon: float = 1e-2,
        non_ranking_loss_coeff: float = 0.01,
        min_model_occurrences: int = 1000,
        wandb_details: WandbDetails | None = None,
    ) -> None:
        super().__init__(wandb_details)
        self.optimizer_spec = optimizer_spec or AdamWSpec()
        self.balance_model_samples = balance_model_samples
        self.print_every = print_every
        self.tie_both_bad_epsilon = tie_both_bad_epsilon
        self.non_ranking_loss_coeff = non_ranking_loss_coeff
        self.min_model_occurrences = min_model_occurrences
        self.preprocessor = SimpleScoringPreprocessor(min_model_occurrences=min_model_occurrences)

        self._history_entries: list[TrainingHistoryEntry] = []
        self._model_encoder: StringEncoder | None = None
        self._network: "SimpleScoringModel._SimpleScoreNetwork" | None = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_timer: Timer | None = None

    def get_config_for_wandb(self) -> dict[str, Any]:
        return {
            "model_type": "simple_scoring",
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "balance_model_samples": self.balance_model_samples,
            "tie_both_bad_epsilon": self.tie_both_bad_epsilon,
            "non_ranking_loss_coeff": self.non_ranking_loss_coeff,
            "min_model_occurrences": self.min_model_occurrences,
            "num_models": self._model_encoder.size
        }

    @property
    def network(self) -> "SimpleScoringModel._SimpleScoreNetwork":
        if self._network is None:
            raise RuntimeError("Network not initialized")
        return self._network

    def train(
        self,
        data: TrainingData,
        validation_split: ValidationSplit | None = None,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        """
        Train the model on pairwise comparison data.
        
        Args:
            data: Training data containing pairwise comparisons
            validation_split: Configuration for train/val split (if None, no validation)
            epochs: Number of training epochs
            batch_size: Batch size for training
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
            
            with Timer("prepare_dataloaders", verbosity="start+end", parent=train_timer):
                dataloader = self._prepare_dataloader(preprocessed_train, batch_size, use_balancing=True)
                val_dataloader = self._prepare_dataloader(preprocessed_val, batch_size, use_balancing=False) if preprocessed_val is not None else None
                
            optimizer = self.optimizer_spec.create_optimizer(self.network)
            scheduler = self.optimizer_spec.create_scheduler(optimizer)
            
            with Timer("epochs", verbosity="start+end", parent=train_timer) as epochs_timer:
                for epoch in range(1, epochs + 1):
                    result = self._train_epoch(epoch, dataloader, val_dataloader, optimizer, epochs_timer)
                    
                    self._log_epoch_result(result)
                    
                    if scheduler is not None:
                        scheduler.step()
            
            self.finish_wandb_if_needed()

    def predict(
        self,
        X: InputData,
        batch_size: int = 32,
    ) -> PromptRoutingOutput:
        """
        Predict best model for each prompt based on learned scores.
        
        Since this model doesn't look at prompts, it will always return
        the same ranking of models regardless of the prompt.
        
        Args:
            X: Input data containing prompts and model names
            batch_size: Batch size for prediction (not used, kept for API consistency)
            
        Returns:
            PromptRoutingOutput with scores for each model across all prompts
        """
        if self._model_encoder is None or self._network is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        self.network.eval()
        
        # Get scores for all requested models
        n_prompts = len(X.prompts)
        scores_dict = {}

        encoded_model_names, known_model_names = self._model_encoder.encode_known(X.model_names)
        model_ids = torch.tensor(encoded_model_names, dtype=torch.long).to(self.device) # [n_models]
        
        with torch.no_grad():
            scores = self.network(model_ids).cpu().numpy() # [n_models]
            for model_name, score in zip(known_model_names, scores):
                scores_dict[model_name] = np.full(n_prompts, score)

            for model_name in X.model_names:
                if model_name not in known_model_names:
                    scores_dict[model_name] = np.zeros(n_prompts)
        
        return PromptRoutingOutput(_scores=scores_dict)

    def get_history(self) -> TrainingHistory:
        """Get training history."""
        return TrainingHistory.from_entries(self._history_entries)

    def get_all_model_scores(self) -> dict[str, float]:
        """
        Get scores for all known models.
        
        Returns:
            Dictionary mapping model names to their learned scores
        """
        if self._model_encoder is None or self._network is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        self.network.eval()
        
        with torch.no_grad():
            all_model_ids = torch.arange(self._model_encoder.size, device=self.device)  # [num_models]
            all_scores = self.network(all_model_ids).cpu().numpy()  # [num_models]
        
        return {
            self._model_encoder.decode(i): float(all_scores[i])
            for i in range(self._model_encoder.size)
        }

    def get_state_dict(self) -> dict[str, Any]:
        """
        Get state dictionary for saving the model.
        
        Returns:
            State dictionary containing all model parameters and configuration
        """
        if self._network is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        if self._model_encoder is None:
            raise RuntimeError("Model encoder not initialized")
        
        return {
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "balance_model_samples": self.balance_model_samples,
            "print_every": self.print_every,
            "tie_both_bad_epsilon": self.tie_both_bad_epsilon,
            "non_ranking_loss_coeff": self.non_ranking_loss_coeff,
            "min_model_occurrences": self.min_model_occurrences,
            "network_state_dict": self.network.cpu().state_dict(),
            "model_encoder": self._model_encoder.get_state_dict(),
            "history_entries": self._history_entries,
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "SimpleScoringModel":
        """
        Load model from state dictionary.
        
        Args:
            state_dict: State dictionary from get_state_dict()
            
        Returns:
            Loaded model instance
        """
        optimizer_spec = OptimizerSpecification.from_serialized(
            state_dict["optimizer_type"],
            state_dict["optimizer_params"],
        )
        
        model = cls(
            optimizer_spec=optimizer_spec,
            balance_model_samples=state_dict["balance_model_samples"],
            print_every=state_dict["print_every"],
            tie_both_bad_epsilon=state_dict["tie_both_bad_epsilon"],
            non_ranking_loss_coeff=state_dict["non_ranking_loss_coeff"],
            min_model_occurrences=state_dict["min_model_occurrences"],
        )
        
        model._model_encoder = StringEncoder.load_state_dict(state_dict["model_encoder"])
        model._initialize_network(num_models=model._model_encoder.size)
        model.network.load_state_dict(
            state_dict["network_state_dict"], 
        )
        model.network.to(model.device)
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

        self.init_wandb_if_needed()  # Needs to be called after model encoder is initialized
        
        if self._network is None:
            self._initialize_network(num_models=self._model_encoder.size)
        
        return preprocessed_data

    def _initialize_network(self, num_models: int) -> None:
        """
        Initialize the neural network.
        
        Args:
            num_models: Number of unique models
        """
        self._network = self._SimpleScoreNetwork(num_models=num_models).to(self.device)

    def _prepare_dataloader(
        self, 
        data: PreprocessedTrainingData, 
        batch_size: int,
        use_balancing: bool,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Prepare dataloader from preprocessed training data.
        
        Handles different comparison types:
        - model_a_wins: label = 1 (a should rank higher)
        - model_b_wins: label = -1 (b should rank higher)
        - tie: label = 0 (special handling - both should be positive)
        - both_bad: label = 2 (special handling - both should be negative)
        
        Args:
            data: Preprocessed training data
            batch_size: Batch size
            use_balancing: Whether to apply sample balancing
            
        Returns:
            DataLoader for training/validation
        """
        if self._model_encoder is None:
            raise RuntimeError("Model encoder not initialized")
        
        # Prepare data for training
        model_ids_a_list = []  # [n_comparisons]
        model_ids_b_list = []  # [n_comparisons]
        labels_list = []  # [n_comparisons]
        
        for comparison in data.comparisons:
            model_ids_a_list.append(comparison.model_id_a)
            model_ids_b_list.append(comparison.model_id_b)
            
            # Map comparison type to label
            if comparison.winner == "model_a":
                labels_list.append(1.0)
            elif comparison.winner == "model_b":
                labels_list.append(-1.0)
            elif comparison.winner == "tie":
                labels_list.append(0.0)
            elif comparison.winner == "both_bad":
                labels_list.append(2.0)
        
        model_ids_a = torch.tensor(model_ids_a_list, dtype=torch.long)  # [n_comparisons]
        model_ids_b = torch.tensor(model_ids_b_list, dtype=torch.long)  # [n_comparisons]
        labels = torch.tensor(labels_list, dtype=torch.float32)  # [n_comparisons]
        
        dataset = TensorDataset(
            model_ids_a,
            model_ids_b,
            labels,
        )
        
        # Apply weighted sampling if balancing is enabled
        sampler = None
        shuffle = True
        if self.balance_model_samples and use_balancing:
            sampler = self._create_balanced_sampler(model_ids_a_list, model_ids_b_list)
            shuffle = False  # Sampler is mutually exclusive with shuffle
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
        )

    def _create_balanced_sampler(
        self,
        model_ids_a: list[int],
        model_ids_b: list[int],
    ) -> WeightedRandomSampler:
        """
        Create a weighted sampler to balance model representation in training.
        
        For each pair, we consider both models involved. The weight for a pair
        is based on the rarest model in that pair.
        
        Args:
            model_ids_a: List of model IDs for position A  # [n_pairs]
            model_ids_b: List of model IDs for position B  # [n_pairs]
            
        Returns:
            WeightedRandomSampler that balances model representation
        """
        # Count how many times each model appears
        model_counts = Counter()
        for model_id_a, model_id_b in zip(model_ids_a, model_ids_b):
            model_counts[model_id_a] += 1
            model_counts[model_id_b] += 1
        
        # Compute weight for each model (inverse frequency)
        model_weights = {
            model_id: 1.0 / count
            for model_id, count in model_counts.items()
        }
        
        # For each pair, assign weight based on the rarest model in the pair
        # This ensures rare models get more representation
        sample_weights = []
        for model_id_a, model_id_b in zip(model_ids_a, model_ids_b):
            weight = max(model_weights[model_id_a], model_weights[model_id_b])
            sample_weights.append(weight)
        
        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)
        
        return WeightedRandomSampler(
            weights=sample_weights_tensor,
            num_samples=len(sample_weights_tensor),
            replacement=True,
        )

    def _train_epoch(
        self, 
        epoch: int,
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None,
        optimizer: optim.Optimizer,
        epochs_timer: Timer,
    ) -> "SimpleScoringModel.EpochResult":
        with Timer(f"epoch_{epoch}", verbosity="start+end", parent=epochs_timer) as timer:
            self.network.train()
            total_loss = 0.0
            total_accuracy = 0.0
            n_batches = 0
            total_samples = 0
            
            # Accumulate loss and accuracy components
            accumulated_loss_components = {"ranking_loss": 0.0, "tie_loss": 0.0, "both_bad_loss": 0.0}
            accumulated_accuracy_components = {"ranking_accuracy": 0.0, "tie_accuracy": 0.0, "both_bad_accuracy": 0.0}
            
            for batch_id_a, batch_id_b, batch_labels in dataloader:
                batch_id_a: torch.Tensor = batch_id_a.to(self.device)  # [batch_size]
                batch_id_b: torch.Tensor = batch_id_b.to(self.device)  # [batch_size]
                batch_labels: torch.Tensor = batch_labels.to(self.device)  # [batch_size]
                
                total_samples += len(batch_id_a)
                
                optimizer.zero_grad()
                scores_a = self.network(batch_id_a)  # [batch_size]
                scores_b = self.network(batch_id_b)  # [batch_size]
                
                loss, loss_components = self._compute_loss(scores_a, scores_b, batch_labels)
                loss.backward()
                
                optimizer.step()
                
                total_loss += loss.item()
                for key, value in loss_components.items():
                    accumulated_loss_components[key] += value
                
                with torch.no_grad():
                    batch_accuracy, accuracy_components = self._compute_accuracy(scores_a, scores_b, batch_labels)
                    total_accuracy += batch_accuracy
                    for key, value in accuracy_components.items():
                        accumulated_accuracy_components[key] += value
                
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            avg_accuracy = total_accuracy / n_batches
            
            # Average the components
            avg_loss_components = {k: v / n_batches for k, v in accumulated_loss_components.items()}
            avg_accuracy_components = {k: v / n_batches for k, v in accumulated_accuracy_components.items()}
            
            # Compute score statistics
            score_stats = self._compute_score_statistics()
            
            with Timer("perform_validation", verbosity="start+end", parent=timer):
                if val_dataloader is not None:
                    val_loss, val_accuracy, val_loss_components, val_accuracy_components = self._perform_validation(val_dataloader, timer)
                else:
                    val_loss = None
                    val_accuracy = None
                    val_loss_components = {}
                    val_accuracy_components = {}
            
            # Build additional metrics
            additional_metrics = {
                "avg_score": score_stats["avg_score"],
                "top_10_pct_score": score_stats["top_10_pct_score"],
                "bottom_10_pct_score": score_stats["bottom_10_pct_score"],
            }
            additional_metrics.update(avg_loss_components)
            additional_metrics.update(avg_accuracy_components)
            
            # Add validation components with "val_" prefix
            for key, value in val_loss_components.items():
                additional_metrics[f"val_{key}"] = value
            for key, value in val_accuracy_components.items():
                additional_metrics[f"val_{key}"] = value
            
            entry = TrainingHistoryEntry(
                epoch=epoch,
                total_loss=avg_loss,
                val_loss=val_loss,
                train_accuracy=avg_accuracy,
                val_accuracy=val_accuracy,
                additional_metrics=additional_metrics,
            )
            self._history_entries.append(entry)
            
            if self.wandb_details is not None:
                self.log_to_wandb(entry)
            
        return self.EpochResult(
            epoch=epoch,
            total_loss=avg_loss,
            train_accuracy=avg_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            duration=timer.elapsed_time,
            additional_metrics=additional_metrics,
        )

    def _compute_loss(
        self,
        scores_a: torch.Tensor,
        scores_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute loss for a batch.
        
        Handles different label types:
        - label = 1: model_a_wins (use margin ranking loss)
        - label = -1: model_b_wins (use margin ranking loss)
        - label = 0: tie (both models should have positive scores)
        - label = 2: both_bad (both models should have negative scores)
        
        Args:
            scores_a: Scores for model A  # [batch_size]
            scores_b: Scores for model B  # [batch_size]
            labels: Comparison labels  # [batch_size]
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Separate different comparison types
        ranking_mask = (labels == 1) | (labels == -1)  # [batch_size]
        tie_mask = labels == 0  # [batch_size]
        both_bad_mask = labels == 2  # [batch_size]
        
        total_loss = torch.tensor(0.0, device=self.device)
        loss_components = {}
        
        # Pairwise ranking loss for model_a_wins and model_b_wins
        if ranking_mask.any():
            ranking_labels = labels[ranking_mask]  # [n_ranking]
            ranking_scores_a = scores_a[ranking_mask]  # [n_ranking]
            ranking_scores_b = scores_b[ranking_mask]  # [n_ranking]
            ranking_loss = nn.functional.margin_ranking_loss(
                ranking_scores_a,
                ranking_scores_b,
                ranking_labels,
                margin=0.1,
            )
            total_loss = total_loss + ranking_loss
            loss_components["ranking_loss"] = ranking_loss.item()
        else:
            loss_components["ranking_loss"] = 0.0
        
        # For ties: both models should have positive scores
        if tie_mask.any():
            tie_scores_a = scores_a[tie_mask]  # [n_ties]
            tie_scores_b = scores_b[tie_mask]  # [n_ties]
            # Target: both scores should be > epsilon
            tie_loss = (
                torch.relu(-tie_scores_a + self.tie_both_bad_epsilon).mean() +
                torch.relu(-tie_scores_b + self.tie_both_bad_epsilon).mean()
            )
            total_loss = total_loss + self.non_ranking_loss_coeff * tie_loss
            loss_components["tie_loss"] = tie_loss.item()
        else:
            loss_components["tie_loss"] = 0.0
        
        # TODO: This might be wrong - loss gets smaller with both_bad acc going to 0
        # For both_bad: both models should have negative scores
        if both_bad_mask.any():
            bad_scores_a = scores_a[both_bad_mask]  # [n_bad]
            bad_scores_b = scores_b[both_bad_mask]  # [n_bad]
            # Target: both scores should be < -epsilon
            bad_loss = (
                torch.relu(bad_scores_a + self.tie_both_bad_epsilon).mean() +
                torch.relu(bad_scores_b + self.tie_both_bad_epsilon).mean()
            )
            total_loss = total_loss + self.non_ranking_loss_coeff * bad_loss
            loss_components["both_bad_loss"] = bad_loss.item()
        else:
            loss_components["both_bad_loss"] = 0.0
        
        return total_loss, loss_components

    def _compute_accuracy(
        self,
        scores_a: torch.Tensor,
        scores_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[float, dict[str, float]]:
        """
        Compute accuracy for a batch.
        
        Computes accuracy for:
        - Ranking comparisons (labels 1 or -1): correct if winner has higher score
        - Tie comparisons (label 0): correct if both scores are positive
        - Both_bad comparisons (label 2): correct if both scores are negative
        
        Args:
            scores_a: Scores for model A  # [batch_size]
            scores_b: Scores for model B  # [batch_size]
            labels: Comparison labels  # [batch_size]
            
        Returns:
            Tuple of (overall_accuracy, accuracy_components)
        """
        ranking_mask = (labels == 1) | (labels == -1)  # [batch_size]
        tie_mask = labels == 0  # [batch_size]
        both_bad_mask = labels == 2  # [batch_size]
        
        total_correct = 0
        total_samples = len(labels)
        accuracy_components = {}
        
        # Ranking accuracy
        if ranking_mask.any():
            ranking_labels = labels[ranking_mask]  # [n_ranking]
            ranking_scores_a = scores_a[ranking_mask]  # [n_ranking]
            ranking_scores_b = scores_b[ranking_mask]  # [n_ranking]
            
            ranking_accuracy = compute_pairwise_accuracy(ranking_scores_a, ranking_scores_b, ranking_labels)
            n_ranking = ranking_mask.sum().item()
            total_correct += ranking_accuracy * n_ranking
            accuracy_components["ranking_accuracy"] = ranking_accuracy
        else:
            accuracy_components["ranking_accuracy"] = 0.0
        
        # Tie accuracy (correct if both scores are positive)
        if tie_mask.any():
            tie_scores_a = scores_a[tie_mask]  # [n_ties]
            tie_scores_b = scores_b[tie_mask]  # [n_ties]
            
            tie_correct = ((tie_scores_a > 0) & (tie_scores_b > 0)).float().mean().item()
            n_ties = tie_mask.sum().item()
            total_correct += tie_correct * n_ties
            accuracy_components["tie_accuracy"] = tie_correct
        else:
            accuracy_components["tie_accuracy"] = 0.0
        
        # Both_bad accuracy (correct if both scores are negative)
        if both_bad_mask.any():
            bad_scores_a = scores_a[both_bad_mask]  # [n_bad]
            bad_scores_b = scores_b[both_bad_mask]  # [n_bad]
            
            bad_correct = ((bad_scores_a < 0) & (bad_scores_b < 0)).float().mean().item()
            n_bad = both_bad_mask.sum().item()
            total_correct += bad_correct * n_bad
            accuracy_components["both_bad_accuracy"] = bad_correct
        else:
            accuracy_components["both_bad_accuracy"] = 0.0
        
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return overall_accuracy, accuracy_components

    def _compute_score_statistics(self) -> dict[str, float]:
        """
        Compute statistics about the current model scores.
        
        Returns:
            Dictionary with avg_score, top_10_pct_score, bottom_10_pct_score
        """
        if self._model_encoder is None or self._network is None:
            return {"avg_score": 0.0, "top_10_pct_score": 0.0, "bottom_10_pct_score": 0.0}
        
        self.network.eval()
        with torch.no_grad():
            all_model_ids = torch.arange(self._model_encoder.size, device=self.device)  # [num_models]
            all_scores = self.network(all_model_ids).cpu().numpy()  # [num_models]
        self.network.train()
        
        avg_score = float(np.mean(all_scores))
        
        # Compute top 10% and bottom 10%
        n_models = len(all_scores)
        top_k = max(1, n_models // 10)
        bottom_k = max(1, n_models // 10)
        
        sorted_scores = np.sort(all_scores)
        top_10_pct_score = float(np.mean(sorted_scores[-top_k:]))
        bottom_10_pct_score = float(np.mean(sorted_scores[:bottom_k]))
        
        return {
            "avg_score": avg_score,
            "top_10_pct_score": top_10_pct_score,
            "bottom_10_pct_score": bottom_10_pct_score,
        }

    def _perform_validation(
        self,
        val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        timer: Timer,
    ) -> tuple[float, float, dict[str, float], dict[str, float]]:
        """
        Perform validation on validation dataloader.
        
        Returns:
            Tuple of (avg_loss, avg_accuracy, loss_components, accuracy_components)
        """
        self.network.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        n_batches = 0
        total_samples = 0
        
        # Accumulate loss and accuracy components
        accumulated_loss_components = {"ranking_loss": 0.0, "tie_loss": 0.0, "both_bad_loss": 0.0}
        accumulated_accuracy_components = {"ranking_accuracy": 0.0, "tie_accuracy": 0.0, "both_bad_accuracy": 0.0}
        
        for batch_id_a, batch_id_b, batch_labels in val_dataloader:
            with Timer(f"batch_{n_batches}", verbosity="start+end", parent=timer):
                batch_id_a: torch.Tensor = batch_id_a.to(self.device)  # [batch_size]
                batch_id_b: torch.Tensor = batch_id_b.to(self.device)  # [batch_size]
                batch_labels: torch.Tensor = batch_labels.to(self.device)  # [batch_size]
                
                total_samples += len(batch_id_a)
                
                with torch.no_grad():
                    scores_a = self.network(batch_id_a)  # [batch_size]
                    scores_b = self.network(batch_id_b)  # [batch_size]
                    
                    loss, loss_components = self._compute_loss(scores_a, scores_b, batch_labels)
                    batch_accuracy, accuracy_components = self._compute_accuracy(scores_a, scores_b, batch_labels)
                    
                    total_loss += loss.item()
                    total_accuracy += batch_accuracy
                    
                    for key, value in loss_components.items():
                        accumulated_loss_components[key] += value
                    for key, value in accuracy_components.items():
                        accumulated_accuracy_components[key] += value
                    
                    n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_accuracy = total_accuracy / n_batches
        avg_loss_components = {k: v / n_batches for k, v in accumulated_loss_components.items()}
        avg_accuracy_components = {k: v / n_batches for k, v in accumulated_accuracy_components.items()}
        
        return avg_loss, avg_accuracy, avg_loss_components, avg_accuracy_components

    def _log_epoch_result(self, result: "SimpleScoringModel.EpochResult") -> None:
        if self.print_every is None:
            return
        
        if not result.epoch % self.print_every == 0:
            return
        
        metrics = result.additional_metrics
        score_str = f"scores: avg={metrics['avg_score']:.3f}, top10%={metrics['top_10_pct_score']:.3f}, bot10%={metrics['bottom_10_pct_score']:.3f}"
        
        if result.val_loss is None or result.val_accuracy is None:
            print(f"Epoch {result.epoch:>4}: loss = {result.total_loss:.4f}, accuracy = {(result.train_accuracy*100):.2f}%, {score_str} - {result.duration:.2f}s")
        else:
            print(f"Epoch {result.epoch:>4}: loss = {result.total_loss:.4f}/{result.val_loss:.4f}, accuracy = {(result.train_accuracy*100):.2f}%/{(result.val_accuracy*100):.2f}%, {score_str} - {result.duration:.2f}s")

    @dataclass
    class EpochResult:
        epoch: int
        total_loss: float
        train_accuracy: float
        val_loss: float | None
        val_accuracy: float | None
        duration: float
        additional_metrics: dict[str, float]

    class _SimpleScoreNetwork(nn.Module):
        """
        Inner PyTorch module that learns a single score per model.
        
        This is just a lookup table - each model gets a learnable score parameter.
        """

        def __init__(self, num_models: int) -> None:
            """
            Initialize the network.
            
            Args:
                num_models: Number of unique models
            """
            super().__init__()
            
            # Just a single score parameter per model
            self.scores = nn.Parameter(torch.zeros(num_models))

        def forward(self, model_id: torch.Tensor) -> torch.Tensor:
            """
            Forward pass - just look up the score for each model.
            
            Args:
                model_id: Model IDs  # [batch_size] or [num_models]
                
            Returns:
                Scores for the requested models  # [batch_size] or [num_models]
            """
            return self.scores[model_id]

