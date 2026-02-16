"""Dense network model for prompt routing."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from collections import Counter

from src.models.scoring_model_base import ScoringModelBase
from src.data_models.data_models import TrainingData, InputData, OutputData
from src.data_models.dense_network_types import PreprocessedTrainingData, PromptRoutingOutput
from src.preprocessing.prompt_embedding_preprocessor import PromptEmbeddingPreprocessor
from src.preprocessing.simple_scaler import SimpleScaler
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.string_encoder import StringEncoder
from src.utils.timer import Timer
from src.utils.torch_utils import state_dict_to_cpu
from src.utils.accuracy import compute_pairwise_accuracy
from src.utils.data_split import ValidationSplit, split_dense_network_preprocessed_data
from src.models.optimizers.optimizer_spec import OptimizerSpecification
from src.models.optimizers.adamw_spec import AdamWSpec
from src.utils.best_model_tracker import BestModelTracker


_DataLoaderType = DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


class DenseNetworkModel(ScoringModelBase):
    """
    Dense neural network for routing prompts to LLMs.
    
    The model takes a prompt embedding and model ID as input and outputs a score in [-1, 1].
    During training, it learns to assign higher scores to winning LLMs.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        hidden_dims: list[int] | None = None,
        model_id_embedding_dim: int = 32,
        optimizer_spec: OptimizerSpecification | None = None,
        balance_model_samples: bool = True,
        run_name: str | None = None,
        print_every: int | None = None,
    ) -> None:
        """
        Initialize the dense network model.
        
        Args:
            embedding_model_name: Name of the sentence transformer model for embeddings
            hidden_dims: List of hidden layer dimensions (default: [256, 128, 64])
            model_id_embedding_dim: Dimension for learned model ID embeddings (default: 32)
            optimizer_spec: Optimizer specification (default: AdamW with LR 0.001)
            balance_model_samples: Whether to balance samples by model frequency
            wandb_details: Weights & Biases configuration
        """
        super().__init__(run_name)
        
        self.embedding_model_name = embedding_model_name
        self.hidden_dims = hidden_dims if hidden_dims is not None else [256, 128, 64]
        self.model_id_embedding_dim = model_id_embedding_dim
        self.optimizer_spec = optimizer_spec if optimizer_spec is not None else AdamWSpec(learning_rate=0.001)
        self.balance_model_samples = balance_model_samples
        self.print_every = print_every
        
        self.preprocessor = PromptEmbeddingPreprocessor(
            embedding_model_name=embedding_model_name,
        )
        
        self._network: DenseNetworkModel._DenseNetwork | None = None
        self._embedding_dim: int | None = None
        self._prompt_features_dim: int | None = None
        self._model_encoder: StringEncoder | None = None
        self._prompt_features_scaler: SimpleScaler | None = None
        self._best_model_tracker = BestModelTracker()
        
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
    def model_encoder(self) -> StringEncoder:
        """Get the model encoder (must be initialized first)."""
        if self._model_encoder is None:
            raise RuntimeError("Model encoder not initialized. Train or load a model first.")
        return self._model_encoder

    def _initialize_network(
        self,
        embedding_dim: int,
        prompt_features_dim: int,
        num_models: int,
    ) -> None:
        """
        Initialize the neural network with the given dimensions.
        
        Args:
            embedding_dim: Dimension of prompt embeddings
            prompt_features_dim: Dimension of prompt features
            num_models: Number of unique models (for embedding layer)
        """
        self._embedding_dim = embedding_dim
        self._prompt_features_dim = prompt_features_dim
        self._network = self._DenseNetwork(
            prompt_embedding_dim=embedding_dim,
            prompt_features_dim=prompt_features_dim,
            num_models=num_models,
            model_id_embedding_dim=self.model_id_embedding_dim,
            hidden_dims=self.hidden_dims,
        ).to(self.device)

    def get_config_for_logging(self) -> dict[str, Any]:
        """Get configuration dictionary for Weights & Biases logging."""
        return {
            "model_type": "dense_network",
            "embedding_model_name": self.embedding_model_name,
            "hidden_dims": self.hidden_dims,
            "model_id_embedding_dim": self.model_id_embedding_dim,
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "balance_model_samples": self.balance_model_samples,
            "preprocessor_version": self.preprocessor.version,
            "embedding_dim": self._embedding_dim,
            "prompt_features_dim": self._prompt_features_dim,
            "num_models": self._model_encoder.size if self._model_encoder else None,
        }

    def train(
        self,
        data: TrainingData,
        validation_split: ValidationSplit | None = None,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        """
        Train the model on the given data.
        
        Args:
            data: Training data (will be split if validation_split is provided)
            validation_split: Configuration for train/val split (if None, no validation)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        with Timer("train", verbosity="start+end") as train_timer:
            self.last_timer = train_timer
            with Timer("initialize_and_preprocess", verbosity="start+end", parent=train_timer):
                preprocessed_data = self._initialize_and_preprocess(data)
            
            with Timer("split_preprocessed_data", verbosity="start+end", parent=train_timer):
                preprocessed_train, preprocessed_val = split_dense_network_preprocessed_data(
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
                    
                    self._best_model_tracker.record_state(
                        accuracy=result.val_accuracy if result.val_accuracy is not None else result.train_accuracy,
                        state_dict=self.get_state_dict(),
                        epoch=epoch
                    )
                    
                    if scheduler is not None:
                        scheduler.step()
            
            # Revert to best model parameters if available
            if self._best_model_tracker.has_best_state:
                print(f"\nReverting to best model parameters from epoch {self._best_model_tracker.best_epoch} (accuracy={self._best_model_tracker.best_accuracy:.4f})")
                self.load_state_dict(self._best_model_tracker.best_state_dict, instance=self)
            
            final_metrics = {
                "best_epoch": self._best_model_tracker.best_epoch,
                "best_accuracy": self._best_model_tracker.best_accuracy,
                "total_epochs": epochs,
            }
            self.finish_logger_if_needed(final_metrics=final_metrics)

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
                preprocessed_input = self.preprocessor.preprocess_for_inference(
                    prompts=X.prompts,
                    model_names=X.model_names,
                    model_encoder=self.model_encoder,
                    scaler=self._prompt_features_scaler,
                )
            
            prompt_embeddings = torch.from_numpy(preprocessed_input.prompt_embeddings).to(self.device)  # [n_prompts, embedding_dim]
            prompt_features = torch.from_numpy(preprocessed_input.prompt_features).to(self.device)  # [n_prompts, prompt_features_dim]
            model_ids = preprocessed_input.model_ids
            
            self.network.eval()
            scores_dict: dict[str, np.ndarray] = {}
            
            with torch.no_grad():
                for model_id, model_name in zip(model_ids, X.model_names):
                    model_scores = []
                    
                    for i in range(0, len(prompt_embeddings), batch_size):
                        batch_embeddings = prompt_embeddings[i:i + batch_size]  # [batch_size, embedding_dim]
                        batch_features = prompt_features[i:i + batch_size]  # [batch_size, prompt_features_dim]
                        batch_size_actual = len(batch_embeddings)
                        
                        batch_model_ids = torch.full(
                            (batch_size_actual,),
                            model_id,
                            dtype=torch.long,
                            device=self.device,
                        )  # [batch_size]
                        
                        batch_scores = self.network(
                            batch_embeddings,
                            batch_features,
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
        if self._network is None or self._prompt_features_scaler is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        if self._model_encoder is None:
            raise RuntimeError("Model encoder not initialized")
        
        return {
            "embedding_model_name": self.embedding_model_name,
            "hidden_dims": self.hidden_dims,
            "model_id_embedding_dim": self.model_id_embedding_dim,
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "balance_model_samples": self.balance_model_samples,
            "print_every": self.print_every,
            "preprocessor_version": self.preprocessor.version,
            "embedding_dim": self._embedding_dim,
            "prompt_features_dim": self._prompt_features_dim,
            "network_state_dict": state_dict_to_cpu(self.network.state_dict()),
            "model_encoder": self._model_encoder.get_state_dict(),
            "prompt_features_scaler_state": self._prompt_features_scaler.get_state_dict(),
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any], instance: "DenseNetworkModel | None" = None) -> "DenseNetworkModel":
        """
        Load model from state dictionary.
        
        Args:
            state_dict: State dictionary from get_state_dict()
            instance: Optional existing model instance to load into
            
        Returns:
            Loaded model instance
        """
        if instance is not None:
            if not isinstance(instance, cls):
                raise TypeError(f"instance must be of type {cls.__name__}, got {type(instance).__name__}")
            model = instance
        else:
            optimizer_spec = OptimizerSpecification.from_serialized(
                state_dict["optimizer_type"],
                state_dict["optimizer_params"],
            )
            
            model = cls(
                embedding_model_name=state_dict["embedding_model_name"],
                hidden_dims=state_dict["hidden_dims"],
                model_id_embedding_dim=state_dict["model_id_embedding_dim"],
                optimizer_spec=optimizer_spec,
                balance_model_samples=state_dict["balance_model_samples"],
                print_every=state_dict["print_every"],
            )
        
        model._model_encoder = StringEncoder.load_state_dict(state_dict["model_encoder"])

        if model._network is None:
            model._initialize_network(
                embedding_dim=state_dict["embedding_dim"],
                prompt_features_dim=state_dict["prompt_features_dim"],
                num_models=model._model_encoder.size,
            )
        model.network.load_state_dict(state_dict["network_state_dict"])
        model.network.to(model.device)
        
        return model

    def _initialize_and_preprocess(
        self,
        data: TrainingData,
    ) -> PreprocessedTrainingData:
        """
        Initialize model from training data.
        
        This should be called once before training, and handles:
        - Wandb initialization
        - Data preprocessing
        - Model encoder creation
        - Network initialization
        
        Args:
            data: Training data to initialize from
            
        Returns:
            Preprocessed training data
        """
        self.init_logger_if_needed()
        
        preprocessed_data = self.preprocessor.preprocess(data)
        self._model_encoder = preprocessed_data.model_encoder
        self._prompt_features_scaler = SimpleScaler.from_state_dict(preprocessed_data.scaler_state)
        
        if self._network is None:
            self._initialize_network(
                embedding_dim=preprocessed_data.embedding_dim,
                prompt_features_dim=preprocessed_data.prompt_features_dim,
                num_models=self._model_encoder.size,
            )
        
        return preprocessed_data

    def _prepare_dataloader(
        self, 
        preprocessed_data: PreprocessedTrainingData, 
        batch_size: int,
        use_balancing: bool,
    ) -> _DataLoaderType:
        """
        Prepare dataloader from preprocessed training data.
        
        Note: _initialize_and_preprocess() must be called before this method.
        
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
        prompt_features_a_list = []  # [n_pairs, prompt_features_dim]
        prompt_features_b_list = []  # [n_pairs, prompt_features_dim]
        model_ids_a_list = []  # [n_pairs]
        model_ids_b_list = []  # [n_pairs]
        labels_list = []  # [n_pairs] - 1 if a wins, -1 if b wins
        
        for pair in preprocessed_data.pairs:
            prompt_embeddings_a_list.append(torch.from_numpy(pair.prompt_embedding))
            prompt_embeddings_b_list.append(torch.from_numpy(pair.prompt_embedding))
            prompt_features_a_list.append(torch.from_numpy(pair.prompt_features))
            prompt_features_b_list.append(torch.from_numpy(pair.prompt_features))
            model_ids_a_list.append(pair.model_id_a)
            model_ids_b_list.append(pair.model_id_b)
            # label: 1 if model_a should be ranked higher, -1 if model_b should be ranked higher
            labels_list.append(1.0 if pair.winner_label == 0 else -1.0)
        
        prompt_embeddings_a = torch.stack(prompt_embeddings_a_list)  # [n_pairs, embedding_dim]
        prompt_embeddings_b = torch.stack(prompt_embeddings_b_list)  # [n_pairs, embedding_dim]
        prompt_features_a = torch.stack(prompt_features_a_list)  # [n_pairs, prompt_features_dim]
        prompt_features_b = torch.stack(prompt_features_b_list)  # [n_pairs, prompt_features_dim]
        model_ids_a = torch.tensor(model_ids_a_list, dtype=torch.long)  # [n_pairs]
        model_ids_b = torch.tensor(model_ids_b_list, dtype=torch.long)  # [n_pairs]
        labels = torch.tensor(labels_list, dtype=torch.float32)  # [n_pairs]
        
        dataset = TensorDataset(
            prompt_embeddings_a,
            prompt_features_a,
            model_ids_a,
            prompt_embeddings_b,
            prompt_features_b,
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
        dataloader: _DataLoaderType,
        val_dataloader: _DataLoaderType | None,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epochs_timer: Timer,
    ) -> "DenseNetworkModel.EpochResult":
        with Timer(f"epoch_{epoch}", verbosity="start+end", parent=epochs_timer) as timer:
            self.network.train()
            total_loss = 0.0
            total_accuracy = 0.0
            n_batches = 0
            total_samples = 0
            
            for batch_emb_a, batch_features_a, batch_id_a, batch_emb_b, batch_features_b, batch_id_b, batch_labels in dataloader:
                batch_emb_a: torch.Tensor = batch_emb_a.to(self.device)  # [batch_size, embedding_dim]
                batch_features_a: torch.Tensor = batch_features_a.to(self.device)  # [batch_size, prompt_features_dim]
                batch_id_a: torch.Tensor = batch_id_a.to(self.device)  # [batch_size]
                batch_emb_b: torch.Tensor = batch_emb_b.to(self.device)  # [batch_size, embedding_dim]
                batch_features_b: torch.Tensor = batch_features_b.to(self.device)  # [batch_size, prompt_features_dim]
                batch_id_b: torch.Tensor = batch_id_b.to(self.device)  # [batch_size]
                batch_labels: torch.Tensor = batch_labels.to(self.device)  # [batch_size]
                
                total_samples += len(batch_emb_a)
                
                optimizer.zero_grad()
                scores_a = self.network(
                    batch_emb_a,
                    batch_features_a,
                    batch_id_a,
                )  # [batch_size]
                scores_b = self.network(
                    batch_emb_b,
                    batch_features_b,
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
            
            self.append_entry_to_log(entry)
            
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
        val_dataloader: _DataLoaderType,
        criterion: nn.Module,
        timer: Timer,
    ) -> tuple[float, float]:
        self.network.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        n_batches = 0
        total_samples = 0
        
        for batch_emb_a, batch_features_a, batch_id_a, batch_emb_b, batch_features_b, batch_id_b, batch_labels in val_dataloader:
            with Timer(f"batch_{n_batches}", verbosity="start+end", parent=timer):
                batch_emb_a: torch.Tensor = batch_emb_a.to(self.device)  # [batch_size, embedding_dim]
                batch_features_a: torch.Tensor = batch_features_a.to(self.device)  # [batch_size, prompt_features_dim]
                batch_id_a: torch.Tensor = batch_id_a.to(self.device)  # [batch_size]
                batch_emb_b: torch.Tensor = batch_emb_b.to(self.device)  # [batch_size, embedding_dim]
                batch_features_b: torch.Tensor = batch_features_b.to(self.device)  # [batch_size, prompt_features_dim]
                batch_id_b: torch.Tensor = batch_id_b.to(self.device)  # [batch_size]
                batch_labels: torch.Tensor = batch_labels.to(self.device)  # [batch_size]
                
                total_samples += len(batch_emb_a)
                
                with torch.no_grad():
                    scores_a = self.network(
                        batch_emb_a,
                        batch_features_a,
                        batch_id_a,
                    )  # [batch_size]
                    scores_b = self.network(
                        batch_emb_b,
                        batch_features_b,
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

    def _log_epoch_result(self, result: "DenseNetworkModel.EpochResult") -> None:
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
        """
        Inner PyTorch module implementing the dense neural network.
        
        Architecture:
        - Model ID embedding layer
        - Concatenation of prompt embedding and model ID embedding
        - Hidden layers with ReLU activation and dropout
        - Output layer with single neuron (score)
        - Tanh applied during prediction to constrain to [-1, 1]
        """

        def __init__(
            self,
            prompt_embedding_dim: int,
            prompt_features_dim: int,
            num_models: int,
            model_id_embedding_dim: int,
            hidden_dims: list[int],
        ) -> None:
            """
            Initialize the network.
            
            Args:
                prompt_embedding_dim: Dimension of prompt embeddings
                prompt_features_dim: Dimension of prompt features
                num_models: Number of unique models (for embedding layer)
                model_id_embedding_dim: Dimension of model ID embeddings
                hidden_dims: List of hidden layer dimensions
            """
            super().__init__()
            
            self.model_embedding = nn.Embedding(
                num_embeddings=num_models,
                embedding_dim=model_id_embedding_dim,
            )
            
            layers = []
            prev_dim = prompt_embedding_dim + prompt_features_dim + model_id_embedding_dim
            
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
            prompt_features: torch.Tensor,
            model_id: torch.Tensor,
        ) -> torch.Tensor:
            """
            Forward pass through the network.
            
            Args:
                prompt_embedding: Prompt embeddings  # [batch_size, prompt_embedding_dim]
                prompt_features: Prompt features  # [batch_size, prompt_features_dim]
                model_id: Model IDs  # [batch_size]
                
            Returns:
                Scores (raw scores, tanh applied separately)  # [batch_size]
            """
            model_embedding = self.model_embedding(model_id)  # [batch_size, model_id_embedding_dim]
            combined = torch.cat([prompt_embedding, prompt_features, model_embedding], dim=-1)  # [batch_size, prompt_embedding_dim + prompt_features_dim + model_id_embedding_dim]
            output: torch.Tensor = self.dense_network(combined)  # [batch_size, 1]
            
            return output.squeeze(-1)  # [batch_size]

