"""Transformer embedding model for prompt routing with LoRA fine-tuning."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from transformers import AutoModel, AutoTokenizer
from pydantic import TypeAdapter
from transformers.modeling_outputs import BaseModelOutputWithPast

from src.models.model_base import ModelBase
from src.data_models.data_models import TrainingData, InputData, OutputData
from src.data_models.transformer_embedding_types import (
    PreprocessedPromptPair,
    PreprocessedTrainingData,
    PromptRoutingOutput,
)
from src.models.embedding_specs.embedding_spec_union import EmbeddingSpec
from src.models.embedding_models.embedding_model_base import EmbeddingModelBase
from src.models.finetuning_specs.finetuning_spec_union import FineTuningSpec
from src.models.finetuning_specs.lora_spec import LoraSpec
from src.models.optimizers.adamw_spec import AdamWSpec
from src.preprocessing.transformer_embedding_preprocessor import TransformerEmbeddingPreprocessor
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.wandb_details import WandbDetails
from src.utils.timer import Timer
from src.utils.accuracy import compute_pairwise_accuracy
from src.utils.data_split import ValidationSplit, split_transformer_embedding_preprocessed_data
from src.models.optimizers.optimizer_spec import OptimizerSpecification
from src.models.finetuning_specs.finetuning_spec import FineTuningSpecification
from src.utils.transformer_pooling_utils import detect_pooling_method, pool_embeddings, PoolingMethod


class TransformerEmbeddingModel(ModelBase):
    def __init__(
        self,
        transformer_model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        finetuning_spec: FineTuningSpec | None = None,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        max_length: int = 256,
        optimizer_spec: OptimizerSpecification | None = None,
        balance_model_samples: bool = True,
        embedding_spec: EmbeddingSpec | None = None,
        load_embedding_model_from: str | None = None,
        min_model_comparisons: int = 20,
        embedding_model_epochs: int = 10,
        scoring_head_lr_multiplier: float = 1.0,
        wandb_details: WandbDetails | None = None,
        print_every: int | None = None,
        save_every: int | None = None,
        checkpoint_name: str | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(wandb_details)

        if load_embedding_model_from is None and embedding_spec is None:
            raise ValueError("Either embedding_spec or load_embedding_model_from must be specified")
        
        self.transformer_model_name = transformer_model_name
        self.finetuning_spec = finetuning_spec if finetuning_spec is not None else LoraSpec(rank=16, alpha=32, dropout=0.05)
        self.hidden_dims = hidden_dims if hidden_dims is not None else [256, 128]
        self.dropout = dropout
        self.max_length = max_length
        self.optimizer_spec = optimizer_spec if optimizer_spec is not None else AdamWSpec(learning_rate=0.0001)
        self.balance_model_samples = balance_model_samples
        self.print_every = print_every
        self.save_every = save_every
        self.checkpoint_name = checkpoint_name or "transformer-embedding-model"
        self.min_model_comparisons = min_model_comparisons
        self.embedding_model_epochs = embedding_model_epochs
        self.scoring_head_lr_multiplier = scoring_head_lr_multiplier
        self.seed = seed
        
        self.embedding_spec = embedding_spec
        self.embedding_model: EmbeddingModelBase | None = None
        
        self.preprocessor = TransformerEmbeddingPreprocessor(
            min_model_comparisons=min_model_comparisons,
        )
        
        self._embedding_model_source: str | None = load_embedding_model_from
        self._prompt_features_dim: int | None = None
        self._network: TransformerEmbeddingModel._TransformerNetwork | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._history_entries: list[TrainingHistoryEntry] = []
        self._pooling_method: PoolingMethod | None = None
        self._optimizer_state: dict[str, Any] | None = None
        self._scheduler_state: dict[str, Any] | None = None
        self._epochs_completed: int = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.last_timer: Timer | None = None

    @property
    def network(self) -> "_TransformerNetwork":
        """Get the neural network (must be initialized first)."""
        if self._network is None:
            raise RuntimeError("Network not initialized. Train or load a model first.")
        return self._network

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer (must be initialized first)."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Train or load a model first.")
        return self._tokenizer

    @property
    def model_embeddings(self) -> dict[str, np.ndarray]:
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not created. Train or load a model first.")
        
        return self.embedding_model.model_embeddings

    def _initialize_network(
        self,
        prompt_features_dim: int,
    ) -> None:
        self._prompt_features_dim = prompt_features_dim
        self._tokenizer = AutoTokenizer.from_pretrained(self.transformer_model_name)
        self._pooling_method = detect_pooling_method(self.transformer_model_name)
        
        if self.print_every is not None:
            print(f"Detected pooling method for {self.transformer_model_name}: {self._pooling_method}")
        
        self._network = self._TransformerNetwork(
            transformer_model_name=self.transformer_model_name,
            prompt_features_dim=prompt_features_dim,
            model_embedding_dim=self.embedding_model.embedding_dim,
            finetuning_spec=self.finetuning_spec,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            pooling_method=self._pooling_method,
            quiet=self.print_every is None,
        ).to(self.device)

    def get_config_for_wandb(self) -> dict[str, Any]:
        """Get configuration dictionary for Weights & Biases logging."""
        return {
            "model_type": "transformer_embedding",
            "transformer_model_name": self.transformer_model_name,
            "finetuning_method": self.finetuning_spec.method,
            "finetuning_spec": self.finetuning_spec.model_dump(),
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "max_length": self.max_length,
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "balance_model_samples": self.balance_model_samples,
            "preprocessor_version": self.preprocessor.version,
            "embedding_type": self.embedding_spec.embedding_type,
            "embedding_spec": self.embedding_spec.model_dump(),
            "min_model_comparisons": self.min_model_comparisons,
            "embedding_model_epochs": self.embedding_model_epochs,
            "scoring_head_lr_multiplier": self.scoring_head_lr_multiplier,
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
            
            with Timer("init_or_load_embedding_model", verbosity="start+end", parent=train_timer):
                if self.embedding_model is None:
                    if self._embedding_model_source is not None:
                        self._load_embedding_model_from_source()
                    elif self.embedding_spec is not None:
                        self.embedding_model = self.embedding_spec.create_model(
                            min_model_comparisons=self.min_model_comparisons,
                            preprocessor_seed=self.seed,
                            print_every=self.print_every,
                        )
                    else:
                        raise RuntimeError("No embedding model available and no way to create one")
            
            with Timer("train_embedding_model", verbosity="start+end", parent=train_timer):
                if not self.embedding_model.is_initialized:
                    self.embedding_model.train(
                        data, 
                        validation_split=validation_split, 
                        epochs=self.embedding_model_epochs, 
                        batch_size=batch_size,
                    )
                elif self.print_every is not None:
                    print("Embedding model is already trained")
            
            with Timer("preprocess", verbosity="start+end", parent=train_timer):
                preprocessed_data = self.preprocessor.preprocess(data)
                
            if self._network is None:
                self._initialize_network(
                    prompt_features_dim=preprocessed_data.prompt_features_dim,
                )
            
            with Timer("add_model_embeddings_to_training_data", verbosity="start+end", parent=train_timer):
                preprocessed_pairs = [
                    PreprocessedPromptPair(
                        prompt=pair.prompt,
                        prompt_features=pair.prompt_features,
                        model_embedding_a=self.model_embeddings[preprocessed_data.model_encoder.decode(pair.model_id_a)],
                        model_embedding_b=self.model_embeddings[preprocessed_data.model_encoder.decode(pair.model_id_b)],
                        model_id_a=pair.model_id_a,
                        model_id_b=pair.model_id_b,
                        winner_label=pair.winner_label,
                    )
                    for pair in preprocessed_data.pairs
                ]
                preprocessed_data_with_embeddings = PreprocessedTrainingData(
                    pairs=preprocessed_pairs,
                    prompt_features_dim=preprocessed_data.prompt_features_dim,
                    model_encoder=preprocessed_data.model_encoder,
                )
            
            with Timer("split_preprocessed_data", verbosity="start+end", parent=train_timer):
                preprocessed_train, preprocessed_val = split_transformer_embedding_preprocessed_data(
                    preprocessed_data_with_embeddings,
                    val_fraction=validation_split.val_fraction if validation_split is not None else 0,
                    seed=validation_split.seed if validation_split is not None else 42,
                )
            
            with Timer("prepare_dataloaders", verbosity="start+end", parent=train_timer):
                dataloader = self._prepare_dataloader(preprocessed_train, batch_size, use_balancing=True)
                val_dataloader = self._prepare_dataloader(preprocessed_val, batch_size, use_balancing=False) if preprocessed_val is not None else None
            
            optimizer, scheduler = self._create_optimizer_and_scheduler()
            
            # MarginRankingLoss: loss = max(0, -label * (score_a - score_b) + margin)
            # When label=1, we want score_a > score_b
            # When label=-1, we want score_b > score_a
            criterion = nn.MarginRankingLoss(margin=0.1)
            
            with Timer("epochs", verbosity="start+end", parent=train_timer) as epochs_timer:
                for epoch in range(self._epochs_completed + 1, self._epochs_completed + epochs + 1):
                    result = self._train_epoch(epoch, dataloader, val_dataloader, optimizer, criterion, epochs_timer)
                    
                    self._log_epoch_result(result)
                    
                    if scheduler is not None:
                        scheduler.step()
                    
                    self._epochs_completed = epoch
                    
                    if self.save_every is not None and epoch % self.save_every == 0:
                        checkpoint_name = f"{self.checkpoint_name}@ep{epoch}"
                        print(f"Saving checkpoint: {checkpoint_name}")
                        self.save(checkpoint_name)
            
            self._optimizer_state = optimizer.state_dict()
            self._scheduler_state = scheduler.state_dict() if scheduler is not None else None
            
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
                preprocessed_input = self.preprocessor.preprocess_for_inference(
                    prompts=X.prompts,
                    model_names=X.model_names,
                    model_embeddings=self.model_embeddings,
                )
            
            prompt_features = torch.from_numpy(preprocessed_input.prompt_features).to(self.device)  # [n_prompts, prompt_features_dim]
            model_embeddings = torch.from_numpy(preprocessed_input.model_embeddings).to(self.device)  # [n_models, model_embedding_dim]
            
            self.network.eval()
            scores_dict: dict[str, np.ndarray] = {}
            
            with torch.no_grad():
                for model_idx, model_name in enumerate(X.model_names):
                    model_embedding = model_embeddings[model_idx]  # [model_embedding_dim]
                    model_scores = []
                    
                    for i in range(0, len(X.prompts), batch_size):
                        batch_prompts = X.prompts[i:i + batch_size]  # [batch_size]
                        batch_features = prompt_features[i:i + batch_size]  # [batch_size, prompt_features_dim]
                        batch_size_actual = len(batch_prompts)
                        
                        # Tokenize prompts
                        tokenized: dict[str, torch.Tensor] = self.tokenizer(
                            batch_prompts,
                            padding=True,
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt",
                        )
                        input_ids = tokenized["input_ids"].to(self.device)  # [batch_size, seq_len]
                        attention_mask = tokenized["attention_mask"].to(self.device)  # [batch_size, seq_len]
                        
                        # Expand model embedding to batch
                        batch_model_embeddings = model_embedding.unsqueeze(0).expand(batch_size_actual, -1)  # [batch_size, model_embedding_dim]
                        
                        batch_scores = self.network(
                            input_ids,
                            attention_mask,
                            batch_features,
                            batch_model_embeddings,
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
        
        if not self.embedding_model.is_initialized:
            raise RuntimeError("Embedding model not initialized")
        
        return {
            "transformer_model_name": self.transformer_model_name,
            "finetuning_method": self.finetuning_spec.method,
            "finetuning_spec": self.finetuning_spec.model_dump(),
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "max_length": self.max_length,
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "optimizer_state": self._optimizer_state,
            "scheduler_state": self._scheduler_state,
            "balance_model_samples": self.balance_model_samples,
            "print_every": self.print_every,
            "save_every": self.save_every,
            "checkpoint_name": self.checkpoint_name,
            "preprocessor_version": self.preprocessor.version,
            "prompt_features_dim": self._prompt_features_dim,
            "network_state_dict": self.network.cpu().state_dict(),
            "history_entries": self._history_entries,
            "epochs_completed": self._epochs_completed,
            
            "embedding_type": self.embedding_spec.embedding_type,
            "embedding_spec": self.embedding_spec.model_dump(),
            "min_model_comparisons": self.min_model_comparisons,
            "embedding_model_epochs": self.embedding_model_epochs,
            "embedding_model_state_dict": self.embedding_model.get_state_dict(),
            "seed": self.seed,
            "scoring_head_lr_multiplier": self.scoring_head_lr_multiplier,
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "TransformerEmbeddingModel":
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
        
        # Parse embedding spec using Pydantic TypeAdapter
        embedding_spec_adapter = TypeAdapter(EmbeddingSpec)
        embedding_spec = embedding_spec_adapter.validate_python(state_dict["embedding_spec"])
        
        finetuning_spec = FineTuningSpecification.from_serialized(
            state_dict["finetuning_method"],
            state_dict["finetuning_spec"],
        )
        
        model = cls(
            transformer_model_name=state_dict["transformer_model_name"],
            finetuning_spec=finetuning_spec,
            hidden_dims=state_dict["hidden_dims"],
            dropout=state_dict["dropout"],
            max_length=state_dict["max_length"],
            optimizer_spec=optimizer_spec,
            balance_model_samples=state_dict["balance_model_samples"],
            embedding_spec=embedding_spec,
            min_model_comparisons=state_dict["min_model_comparisons"],
            embedding_model_epochs=state_dict["embedding_model_epochs"],
            scoring_head_lr_multiplier=state_dict.get("scoring_head_lr_multiplier", 1.0),
            print_every=state_dict["print_every"],
            save_every=state_dict.get("save_every", None),
            checkpoint_name=state_dict.get("checkpoint_name", None),
            seed=state_dict["seed"],
        )
        
        model.embedding_model = EmbeddingModelBase.load_from_state_dict(state_dict["embedding_model_state_dict"])

        model._initialize_network(
            prompt_features_dim=state_dict["prompt_features_dim"],
        )
        model.network.load_state_dict(
            state_dict["network_state_dict"],
            strict=False,
        )
        model.network.to(model.device)
        
        model._history_entries = state_dict["history_entries"]
        model._optimizer_state = state_dict.get("optimizer_state")
        model._scheduler_state = state_dict.get("scheduler_state")
        model._epochs_completed = state_dict.get("epochs_completed", 0)
        
        return model

    def _prepare_dataloader(
        self, 
        preprocessed_data: PreprocessedTrainingData, 
        batch_size: int,
        use_balancing: bool,
    ) -> DataLoader[dict[str, torch.Tensor]]:
        """
        Prepare dataloader from preprocessed training data.
        
        Args:
            preprocessed_data: Preprocessed training data
            batch_size: Batch size
            use_balancing: Whether to apply sample balancing
            
        Returns:
            DataLoader for training/validation
        """
        dataset = self._PairwiseDataset(
            pairs=preprocessed_data.pairs,
            max_length=self.max_length,
        )
        
        # Apply weighted sampling if balancing is enabled
        sampler = None
        shuffle = True
        if self.balance_model_samples and use_balancing:
            model_ids_a = [pair.model_id_a for pair in preprocessed_data.pairs]
            model_ids_b = [pair.model_id_b for pair in preprocessed_data.pairs]
            sampler = self._create_balanced_sampler(model_ids_a, model_ids_b)
            shuffle = False  # Sampler is mutually exclusive with shuffle
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Collate function for dataloader."""
        prompts: list[str] = [item["prompt"] for item in batch]
        
        tokenized: dict[str, torch.Tensor] = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": tokenized["input_ids"],  # [batch_size, seq_len]
            "attention_mask": tokenized["attention_mask"],  # [batch_size, seq_len]
            "prompt_features": torch.stack([item["prompt_features"] for item in batch]),  # [batch_size, prompt_features_dim]
            "model_embedding_a": torch.stack([item["model_embedding_a"] for item in batch]),  # [batch_size, model_embedding_dim]
            "model_embedding_b": torch.stack([item["model_embedding_b"] for item in batch]),  # [batch_size, model_embedding_dim]
            "labels": torch.stack([item["label"] for item in batch]),  # [batch_size]
        }
        
    def _create_optimizer_and_scheduler(self) -> tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler | None]:
        optimizer = self.optimizer_spec.create_optimizer(
            self.network,
            lr_multipliers={
                self.network.scoring_head: self.scoring_head_lr_multiplier,
            },
        )
        
        if self._optimizer_state is not None:
            optimizer.load_state_dict(self._optimizer_state)
            if self.print_every is not None:
                print("Restored optimizer state from checkpoint")
        
        scheduler = self.optimizer_spec.create_scheduler(optimizer)
        if scheduler is not None and self._scheduler_state is not None:
            scheduler.load_state_dict(self._scheduler_state)
            if self.print_every is not None:
                print("Restored scheduler state from checkpoint")
                
        return optimizer, scheduler

    def _create_balanced_sampler(
        self,
        model_ids_a: list[int],
        model_ids_b: list[int],
    ) -> torch.utils.data.WeightedRandomSampler:
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
        
        return torch.utils.data.WeightedRandomSampler(
            weights=sample_weights_tensor,
            num_samples=len(sample_weights_tensor),
            replacement=True,
        )

    def _train_epoch(
        self, 
        epoch: int,
        dataloader: DataLoader[dict[str, torch.Tensor]],
        val_dataloader: DataLoader[dict[str, torch.Tensor]] | None,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epochs_timer: Timer,
    ) -> "TransformerEmbeddingModel.EpochResult":
        with Timer(f"epoch_{epoch}", verbosity="start+end", parent=epochs_timer) as timer:
            self.network.train()
            total_loss = 0.0
            total_accuracy = 0.0
            n_batches = 0
            
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)  # [batch_size, seq_len]
                attention_mask = batch["attention_mask"].to(self.device)  # [batch_size, seq_len]
                prompt_features = batch["prompt_features"].to(self.device)  # [batch_size, prompt_features_dim]
                model_embedding_a = batch["model_embedding_a"].to(self.device)  # [batch_size, model_embedding_dim]
                model_embedding_b = batch["model_embedding_b"].to(self.device)  # [batch_size, model_embedding_dim]
                labels = batch["labels"].to(self.device)  # [batch_size]
                
                optimizer.zero_grad()
                scores_a = self.network(
                    input_ids,
                    attention_mask,
                    prompt_features,
                    model_embedding_a,
                )  # [batch_size]
                scores_b = self.network(
                    input_ids,
                    attention_mask,
                    prompt_features,
                    model_embedding_b,
                )  # [batch_size]
                
                loss: torch.Tensor = criterion(scores_a, scores_b, labels)
                loss.backward()
                
                optimizer.step()
                
                total_loss += loss.item()
                
                with torch.no_grad():
                    batch_accuracy = compute_pairwise_accuracy(scores_a, scores_b, labels)
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
        val_dataloader: DataLoader[dict[str, torch.Tensor]],
        criterion: nn.Module,
        timer: Timer,
    ) -> tuple[float, float]:
        self.network.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(self.device)  # [batch_size, seq_len]
                attention_mask = batch["attention_mask"].to(self.device)  # [batch_size, seq_len]
                prompt_features = batch["prompt_features"].to(self.device)  # [batch_size, prompt_features_dim]
                model_embedding_a = batch["model_embedding_a"].to(self.device)  # [batch_size, model_embedding_dim]
                model_embedding_b = batch["model_embedding_b"].to(self.device)  # [batch_size, model_embedding_dim]
                labels = batch["labels"].to(self.device)  # [batch_size]
                
                scores_a = self.network(
                    input_ids,
                    attention_mask,
                    prompt_features,
                    model_embedding_a,
                )  # [batch_size]
                scores_b = self.network(
                    input_ids,
                    attention_mask,
                    prompt_features,
                    model_embedding_b,
                )  # [batch_size]
                
                loss: torch.Tensor = criterion(scores_a, scores_b, labels)
                batch_accuracy = compute_pairwise_accuracy(scores_a, scores_b, labels)
                
                total_loss += loss.item()
                total_accuracy += batch_accuracy
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_accuracy = total_accuracy / n_batches
        return avg_loss, avg_accuracy

    def _load_embedding_model_from_source(self) -> None:
        if self._embedding_model_source is None:
            raise RuntimeError("No embedding model source specified")

        loaded: TransformerEmbeddingModel = TransformerEmbeddingModel.load(self._embedding_model_source)
        self.embedding_model = loaded.embedding_model
        self.embedding_spec = loaded.embedding_spec
        self.embedding_model_epochs = loaded.embedding_model_epochs
        
        if self.print_every is not None:
            print(f"Loaded embedding model from {self._embedding_model_source}")

    def _log_epoch_result(self, result: "TransformerEmbeddingModel.EpochResult") -> None:
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

    class _PairwiseDataset(Dataset):
        """Dataset for pairwise comparisons."""
        
        def __init__(
            self,
            pairs: list[PreprocessedPromptPair],
            max_length: int,
        ) -> None:
            self.pairs = pairs
            self.max_length = max_length
        
        def __len__(self) -> int:
            return len(self.pairs)
        
        def __getitem__(self, idx: int) -> dict:
            pair = self.pairs[idx]
            
            # Return raw data, tokenization happens in collate_fn
            return {
                "prompt": pair.prompt,
                "prompt_features": torch.from_numpy(pair.prompt_features),
                "model_embedding_a": torch.from_numpy(pair.model_embedding_a),
                "model_embedding_b": torch.from_numpy(pair.model_embedding_b),
                "label": torch.tensor(1.0 if pair.winner_label == 0 else -1.0, dtype=torch.float32),
            }

    class _TransformerNetwork(nn.Module):
        """Transformer-based network with configurable fine-tuning."""
        
        def __init__(
            self,
            transformer_model_name: str,
            prompt_features_dim: int,
            model_embedding_dim: int,
            finetuning_spec: FineTuningSpec,
            hidden_dims: list[int],
            dropout: float,
            pooling_method: PoolingMethod,
            quiet: bool = False,
        ) -> None:
            super().__init__()
            
            self.pooling_method = pooling_method
            
            quantization_config = finetuning_spec.get_quantization_config()
            if quantization_config is not None:
                self.transformer = AutoModel.from_pretrained(
                    transformer_model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            else:
                self.transformer = AutoModel.from_pretrained(transformer_model_name)
            
            transformer_hidden_size: int = self.transformer.config.hidden_size
            
            self.transformer = finetuning_spec.apply_to_model(self.transformer, quiet=quiet)
            
            # Build configurable scoring head
            layers = []
            prev_dim = transformer_hidden_size + prompt_features_dim + model_embedding_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.LeakyReLU(0.1))
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            
            self.scoring_head = nn.Sequential(*layers)
        
        def forward(
            self,
            input_ids: torch.Tensor,  # [batch_size, seq_len]
            attention_mask: torch.Tensor,  # [batch_size, seq_len]
            prompt_features: torch.Tensor,  # [batch_size, prompt_features_dim]
            model_embedding: torch.Tensor,  # [batch_size, model_embedding_dim]
        ) -> torch.Tensor:
            outputs: BaseModelOutputWithPast = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            prompt_embedding = pool_embeddings(
                outputs.last_hidden_state,  # [batch_size, seq_len, transformer_hidden_size]
                attention_mask,  # [batch_size, seq_len]
                self.pooling_method,
            )  # [batch_size, transformer_hidden_size]
            
            combined = torch.cat([prompt_embedding, prompt_features, model_embedding], dim=1)  # [batch_size, combined_dim]
            output: torch.Tensor = self.scoring_head(combined)  # [batch_size, 1]
            
            return output.squeeze(-1)  # [batch_size]
