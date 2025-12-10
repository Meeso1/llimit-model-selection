"""Dense network model for prompt routing."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from collections import Counter

from src.models.model_base import ModelBase
from src.data_models.data_models import TrainingData, InputData, OutputData
from src.data_models.dn_embedding_network_types import PreprocessedPromptPair, PreprocessedTrainingData, PromptRoutingOutput
from src.models.model_behavior_encoder import ModelBehaviorEncoder
from src.preprocessing.prompt_embedding_preprocessor import PromptEmbeddingPreprocessor
from src.utils.string_encoder import StringEncoder
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.wandb_details import WandbDetails
from src.utils.timer import Timer
from src.utils.accuracy import compute_pairwise_accuracy
from src.utils.data_split import ValidationSplit, split_preprocessed_data
from src.models.optimizers.optimizer_spec import OptimizerSpecification
from src.models.optimizers.adamw_spec import AdamWSpec


class DnEmbeddingModel(ModelBase):
    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        optimizer_spec: OptimizerSpecification | None = None,
        balance_model_samples: bool = True,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_model_hidden_dims: list[int] | None = None,
        embedding_model_optimizer_spec: OptimizerSpecification | None = None,
        triplet_margin: float = 0.2,
        regularization_weight: float = 0.01,
        min_model_comparisons: int = 20,
        identity_positive_ratio: float = 0.8,
        embedding_model_epochs: int = 10,
        wandb_details: WandbDetails | None = None,
        print_every: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(wandb_details)
        
        self.hidden_dims = hidden_dims if hidden_dims is not None else [256, 128, 64]
        self.optimizer_spec = optimizer_spec if optimizer_spec is not None else AdamWSpec(learning_rate=0.001)
        self.balance_model_samples = balance_model_samples
        self.print_every = print_every
        
        self.embedding_model_epochs = embedding_model_epochs
        self.embedding_model = ModelBehaviorEncoder(
            encoder_model_name=embedding_model_name,
            hidden_dims=embedding_model_hidden_dims,
            optimizer_spec=embedding_model_optimizer_spec,
            triplet_margin=triplet_margin,
            regularization_weight=regularization_weight,
            min_model_comparisons=min_model_comparisons,
            identity_positive_ratio=identity_positive_ratio,
            preprocessor_seed=seed,
            print_every=print_every,
        )
        
        self.preprocessor = PromptEmbeddingPreprocessor(
            embedding_model_name=embedding_model_name,
        )
        
        self._prompt_embedding_dim: int | None = None
        self._network: DnEmbeddingModel._DenseNetwork | None = None
        self._model_embeddings: dict[str, np.ndarray] | None = None
        
        self._history_entries: list[TrainingHistoryEntry] = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.last_timer: Timer | None = None

    @property
    def network(self) -> "_DenseNetwork":
        """Get the neural network (must be initialized first)."""
        if self._network is None:
            raise RuntimeError("Network not initialized. Train or load a model first.")
        return self._network

    @property
    def model_embeddings(self) -> dict[str, np.ndarray]:
        """Get the model encoder (must be initialized first)."""
        if self._model_embeddings is None:
            raise RuntimeError("Model embeddings not initialized. Train or load a model first.")
        return self._model_embeddings

    def _initialize_network(
        self,
        prompt_embedding_dim: int,
    ) -> None:
        self._prompt_embedding_dim = prompt_embedding_dim
        self._network = self._DenseNetwork(
            prompt_embedding_dim=prompt_embedding_dim,
            model_id_embedding_dim=self.embedding_model_hidden_dims[-1],
            hidden_dims=self.hidden_dims,
        ).to(self.device)

    def get_config_for_wandb(self) -> dict[str, Any]:
        """Get configuration dictionary for Weights & Biases logging."""
        return {
            "model_type": "dense_network",
            "hidden_dims": self.hidden_dims,
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "balance_model_samples": self.balance_model_samples,
            "preprocessor_version": self.preprocessor.version,
            "embedding_model_name": self.embedding_model.encoder_model_name,
            "embedding_model_hidden_dims": self.embedding_model.hidden_dims,
            "embedding_model_optimizer_type": self.embedding_model.optimizer_spec.optimizer_type,
            "embedding_model_optimizer_params": self.embedding_model.optimizer_spec.to_dict(),
            "embedding_model_epochs": self.embedding_model_epochs,
            "triplet_margin": self.embedding_model.triplet_margin,
            "regularization_weight": self.embedding_model.regularization_weight,
            "min_model_comparisons": self.embedding_model.min_model_comparisons,
            "identity_positive_ratio": self.embedding_model.identity_positive_ratio,
        }

    def train(
        self,
        data: TrainingData,
        validation_split: ValidationSplit | None = None,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        with Timer("train", verbosity="start+end") as train_timer:
            self.last_timer = train_timer
            
            with Timer("train_embedding_model", verbosity="start+end", parent=train_timer):
                self.embedding_model.train(
                    data, 
                    validation_split=validation_split, 
                    epochs=self.embedding_model_epochs, 
                    batch_size=batch_size,
                )
            
            if self._network is None:
                self._initialize_network(
                    prompt_embedding_dim=self.embedding_model.embedding_dim,
                )
            
            with Timer("encode_prompts", verbosity="start+end", parent=train_timer):
                encoded_prompts = self.preprocessor.preprocess(data)
                
            with Timer("prepare_preprocessed_data", verbosity="start+end", parent=train_timer):
                preprocessed_pairs = [
                    PreprocessedPromptPair(
                        prompt_embedding=pair.prompt_embedding,
                        model_embedding_a=self.model_embeddings[encoded_prompts.model_encoder.decode(pair.model_id_a)],
                        model_embedding_b=self.model_embeddings[encoded_prompts.model_encoder.decode(pair.model_id_b)],
                        winner_label=pair.winner_label,
                    )
                    for pair in encoded_prompts.pairs
                ]
                preprocessed_data = PreprocessedTrainingData(pairs=preprocessed_pairs)
            
            with Timer("split_preprocessed_data", verbosity="start+end", parent=train_timer):
                preprocessed_train, preprocessed_val = split_preprocessed_data(
                    preprocessed_data,
                    val_fraction=validation_split.val_fraction if validation_split is not None else 0,
                    seed=validation_split.seed if validation_split is not None else 42,
                )
            
            with Timer("prepare_dataloaders", verbosity="start+end", parent=train_timer):
                dataloader = self._prepare_dataloader(preprocessed_train, batch_size, use_balancing=True)
                val_dataloader = self._prepare_dataloader(preprocessed_val, batch_size, use_balancing=False) if preprocessed_val is not None else None
                
            optimizer = self.optimizer_spec.create_optimizer(self.network)
            scheduler = self.optimizer_spec.create_scheduler(optimizer)
            
            # MarginRankingLoss: loss = max(0, -label * (score_a - score_b) + margin)
            # When label=1, we want score_a > score_b
            # When label=-1, we want score_b > score_a
            criterion = nn.MarginRankingLoss(margin=0.1)
            
            with Timer("epochs", verbosity="start+end", parent=train_timer) as epochs_timer:
                for epoch in range(1, epochs + 1):
                    result = self._train_epoch(epoch, dataloader, val_dataloader, optimizer, criterion, epochs_timer)
                    
                    self._log_epoch_result(result)
                    
                    if scheduler is not None:
                        scheduler.step()
            
            self.finish_wandb_if_needed()

    def predict(
        self,
        X: InputData,
        batch_size: int = 32,
    ) -> OutputData:
        """
        Predict scores for the given prompts and models.
        
        Args:
            X: Input data with prompts and model_names
            batch_size: Batch size for prediction
            
        Returns:
            PromptRoutingOutput with scores for each model
        """
        if self._network is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        with Timer("predict", verbosity="start+end") as predict_timer:
            self.last_timer = predict_timer
            with Timer("preprocess_input", verbosity="start+end", parent=predict_timer):
                encoded_prompts = self.preprocessor.preprocess_for_inference(
                    prompts=X.prompts,
                    model_names=X.model_names,
                    model_encoder=StringEncoder() # We don't need model IDs here
                )
            
            prompt_embeddings = torch.from_numpy(encoded_prompts.prompt_embeddings).to(self.device)  # [n_prompts, embedding_dim]
            model_ids = torch.from_numpy(np.array([
                self.model_embeddings[model_name] if model_name in self.model_embeddings else self.model_embeddings["default"] 
                for model_name in X.model_names
            ])).to(self.device) # [n_models, model_embedding_dim]
            
            self.network.eval()
            scores_dict: dict[str, np.ndarray] = {}
            
            with torch.no_grad():
                for model_id, model_name in zip(model_ids, X.model_names):
                    model_scores = []
                    
                    for i in range(0, len(prompt_embeddings), batch_size):
                        batch_embeddings = prompt_embeddings[i:i + batch_size]  # [batch_size, embedding_dim]
                        batch_size_actual = len(batch_embeddings)
                        
                        batch_model_ids = torch.full(
                            (batch_size_actual,),
                            model_id,
                            dtype=torch.long,
                            device=self.device,
                        )  # [batch_size]
                        
                        batch_scores = self.network(
                            batch_embeddings,
                            batch_model_ids,
                        )  # [batch_size]
                        
                        # Apply tanh to constrain to [-1, 1]
                        batch_scores = torch.tanh(batch_scores)  # [batch_size]
                        model_scores.append(batch_scores)
                    
                    all_scores = torch.cat(model_scores)  # [n_prompts]
                    scores_dict[model_name] = all_scores.cpu().numpy()
            
            return PromptRoutingOutput(_scores=scores_dict)

    def get_history(self) -> TrainingHistory:
        """Get training history."""
        return TrainingHistory.from_entries(self._history_entries)

    def get_state_dict(self) -> dict[str, Any]:
        """
        Get state dictionary for saving the model.
        
        Returns:
            State dictionary containing all model parameters and configuration
        """
        if self._network is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        if self.embedding_model._module is None:
            raise RuntimeError("Embedding model not initialized")
        
        return {
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "hidden_dims": self.hidden_dims,
            "balance_model_samples": self.balance_model_samples,
            "print_every": self.print_every,
            "preprocessor_version": self.preprocessor.version,
            "prompt_embedding_dim": self._prompt_embedding_dim,
            "network_state_dict": self.network.state_dict(),
            "history_entries": self._history_entries,
        
            "embedding_model_name": self.embedding_model.encoder_model_name,
            "embedding_model_hidden_dims": self.embedding_model.hidden_dims,
            "embedding_model_optimizer_type": self.embedding_model.optimizer_spec.optimizer_type,
            "embedding_model_optimizer_params": self.embedding_model.optimizer_spec.to_dict(),
            "embedding_model_epochs": self.embedding_model_epochs,
            "triplet_margin": self.embedding_model.triplet_margin,
            "regularization_weight": self.embedding_model.regularization_weight,
            "min_model_comparisons": self.embedding_model.min_model_comparisons,
            "identity_positive_ratio": self.embedding_model.identity_positive_ratio,
            "embedding_model_state_dict": self.embedding_model.get_state_dict(),
            "seed": self.embedding_model.preprocessor_seed,
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "DnEmbeddingModel":
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
        
        embedding_model_optimizer_spec = OptimizerSpecification.from_serialized(
            state_dict["embedding_model_optimizer_type"],
            state_dict["embedding_model_optimizer_params"],
        )
        
        model = cls(
            hidden_dims=state_dict["hidden_dims"],
            optimizer_spec=optimizer_spec,
            balance_model_samples=state_dict["balance_model_samples"],
            embedding_model_name=state_dict["embedding_model_name"],
            embedding_model_hidden_dims=state_dict["embedding_model_hidden_dims"],
            embedding_model_optimizer_spec=embedding_model_optimizer_spec,
            triplet_margin=state_dict["triplet_margin"],
            regularization_weight=state_dict["regularization_weight"],
            min_model_comparisons=state_dict["min_model_comparisons"],
            identity_positive_ratio=state_dict["identity_positive_ratio"],
            embedding_model_epochs=state_dict["embedding_model_epochs"],
            print_every=state_dict["print_every"],
            seed=state_dict["seed"],
        )
        
        model.embedding_model = ModelBehaviorEncoder.load_state_dict(state_dict["embedding_model_state_dict"])

        model._initialize_network(
            prompt_embedding_dim=state_dict["prompt_embedding_dim"],
        )
        model.network.load_state_dict(state_dict["network_state_dict"])
        
        model._history_entries = state_dict["history_entries"]
        
        return model

    def _prepare_dataloader(
        self, 
        preprocessed_data: PreprocessedTrainingData, 
        batch_size: int,
        use_balancing: bool,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Prepare dataloader from preprocessed training data.
        
        Args:
            preprocessed_data: Preprocessed training data
            batch_size: Batch size
            use_balancing: Whether to apply sample balancing
            
        Returns:
            DataLoader for training/validation
        """
        # Prepare data for training
        # Each comparison pair becomes a training sample with margin ranking loss
        prompt_embeddings_a_list = []  # [n_pairs, embedding_dim]
        prompt_embeddings_b_list = []  # [n_pairs, embedding_dim]
        model_ids_a_list = []  # [n_pairs]
        model_ids_b_list = []  # [n_pairs]
        labels_list = []  # [n_pairs] - 1 if a wins, -1 if b wins
        
        for pair in preprocessed_data.pairs:
            prompt_embeddings_a_list.append(torch.from_numpy(pair.prompt_embedding))
            prompt_embeddings_b_list.append(torch.from_numpy(pair.prompt_embedding))
            model_ids_a_list.append(pair.model_id_a)
            model_ids_b_list.append(pair.model_id_b)
            # label: 1 if model_a should be ranked higher, -1 if model_b should be ranked higher
            labels_list.append(1.0 if pair.winner_label == 0 else -1.0)
        
        prompt_embeddings_a = torch.stack(prompt_embeddings_a_list)  # [n_pairs, embedding_dim]
        prompt_embeddings_b = torch.stack(prompt_embeddings_b_list)  # [n_pairs, embedding_dim]
        model_ids_a = torch.tensor(model_ids_a_list, dtype=torch.long)  # [n_pairs]
        model_ids_b = torch.tensor(model_ids_b_list, dtype=torch.long)  # [n_pairs]
        labels = torch.tensor(labels_list, dtype=torch.float32)  # [n_pairs]
        
        dataset = TensorDataset(
            prompt_embeddings_a,
            model_ids_a,
            prompt_embeddings_b,
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
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epochs_timer: Timer,
    ) -> "DnEmbeddingModel.EpochResult":
        with Timer(f"epoch_{epoch}", verbosity="start+end", parent=epochs_timer) as timer:
            self.network.train()
            total_loss = 0.0
            total_accuracy = 0.0
            n_batches = 0
            total_samples = 0
            
            for batch_emb_a, batch_id_a, batch_emb_b, batch_id_b, batch_labels in dataloader:
                batch_emb_a: torch.Tensor = batch_emb_a.to(self.device)  # [batch_size, embedding_dim]
                batch_id_a: torch.Tensor = batch_id_a.to(self.device)  # [batch_size]
                batch_emb_b: torch.Tensor = batch_emb_b.to(self.device)  # [batch_size, embedding_dim]
                batch_id_b: torch.Tensor = batch_id_b.to(self.device)  # [batch_size]
                batch_labels: torch.Tensor = batch_labels.to(self.device)  # [batch_size]
                
                total_samples += len(batch_emb_a)
                
                optimizer.zero_grad()
                scores_a = self.network(
                    batch_emb_a,
                    batch_id_a,
                )  # [batch_size]
                scores_b = self.network(
                    batch_emb_b,
                    batch_id_b,
                )  # [batch_size]
                
                loss: torch.Tensor = criterion(scores_a, scores_b, batch_labels) # [batch_size]
                loss.backward()
                
                optimizer.step()
                
                total_loss += loss.mean().item()
                
                with torch.no_grad():
                    batch_accuracy = compute_pairwise_accuracy(scores_a, scores_b, batch_labels)
                    total_accuracy += batch_accuracy
                
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            avg_accuracy = total_accuracy / n_batches
            
            with Timer("perform_validation", verbosity="start+end", parent=timer):
                val_loss, val_accuracy = self._perform_validation(val_dataloader, criterion, timer) if val_dataloader is not None else (None, None)
            
            entry = TrainingHistoryEntry(
                epoch=epoch,
                total_loss=avg_loss,
                val_loss=val_loss,
                train_accuracy=avg_accuracy,
                val_accuracy=val_accuracy,
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
        )

    def _perform_validation(
        self,
        val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
        timer: Timer,
    ) -> tuple[float, float]:
        self.network.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        n_batches = 0
        total_samples = 0
        
        for batch_emb_a, batch_id_a, batch_emb_b, batch_id_b, batch_labels in val_dataloader:
            with Timer(f"batch_{n_batches}", verbosity="start+end", parent=timer):
                batch_emb_a: torch.Tensor = batch_emb_a.to(self.device)  # [batch_size, embedding_dim]
                batch_id_a: torch.Tensor = batch_id_a.to(self.device)  # [batch_size]
                batch_emb_b: torch.Tensor = batch_emb_b.to(self.device)  # [batch_size, embedding_dim]
                batch_id_b: torch.Tensor = batch_id_b.to(self.device)  # [batch_size]
                batch_labels: torch.Tensor = batch_labels.to(self.device)  # [batch_size]
                
                total_samples += len(batch_emb_a)
                
                with torch.no_grad():
                    scores_a = self.network(
                        batch_emb_a,
                        batch_id_a,
                    )  # [batch_size]
                    scores_b = self.network(
                        batch_emb_b,
                        batch_id_b,
                    )  # [batch_size]
                    
                    loss: torch.Tensor = criterion(scores_a, scores_b, batch_labels) # [batch_size]
                    batch_accuracy = compute_pairwise_accuracy(scores_a, scores_b, batch_labels)
                    
                    total_loss += loss.mean().item()
                    total_accuracy += batch_accuracy
                    n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_accuracy = total_accuracy / n_batches
        return avg_loss, avg_accuracy

    def _log_epoch_result(self, result: "DnEmbeddingModel.EpochResult") -> None:
        if self.print_every is None:
            return
        
        if not result.epoch % self.print_every == 0:
            return
        
        if result.val_loss is None or result.val_accuracy is None:
            print(f"Epoch {result.epoch:>4}: loss = {result.total_loss:.4f}, accuracy = {(result.train_accuracy*100):.4f}% - {result.duration:.2f}s")
        else:
            print(f"Epoch {result.epoch:>4}: loss = {result.total_loss:.4f}/{result.val_loss:.4f}, accuracy = {(result.train_accuracy*100):.4f}%/{(result.val_accuracy*100):.4f}% - {result.duration:.2f}s")

    @dataclass
    class EpochResult:
        epoch: int
        total_loss: float
        train_accuracy: float
        val_loss: float | None
        val_accuracy: float | None
        duration: float

    class _DenseNetwork(nn.Module):
        def __init__(
            self,
            prompt_embedding_dim: int,
            model_embedding_dim: int,
            hidden_dims: list[int],
        ) -> None:
            super().__init__()
            
            layers = []
            prev_dim = prompt_embedding_dim + model_embedding_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            
            self.dense_network = nn.Sequential(*layers)

        def forward(
            self,
            prompt_embedding: torch.Tensor,
            model_embedding: torch.Tensor,
        ) -> torch.Tensor:
            combined = torch.cat([prompt_embedding, model_embedding], dim=-1)  # [batch_size, prompt_embedding_dim + model_embedding_dim]
            output: torch.Tensor = self.dense_network(combined)  # [batch_size, 1]
            
            return output.squeeze(-1)  # [batch_size]

