"""Transformer embedding model for prompt routing with LoRA fine-tuning."""

from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from collections import Counter
from transformers import AutoModel, AutoTokenizer
from pydantic import TypeAdapter
from transformers.modeling_outputs import BaseModelOutputWithPast

from src.models.scoring.scoring_model_base import ScoringModelBase
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
from src.preprocessing.simple_scaler import SimpleScaler
from src.preprocessing.transformer_embedding_preprocessor import TransformerEmbeddingPreprocessor
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.timer import Timer
from src.utils.torch_utils import state_dict_to_cpu, state_dict_to_device
from src.utils.accuracy import compute_pairwise_accuracy
from src.utils.data_split import ValidationSplit, split_transformer_embedding_preprocessed_data
from src.utils.best_model_tracker import BestModelTracker
from src.models.optimizers.optimizer_spec import OptimizerSpecification
from src.models.finetuning_specs.finetuning_spec import FineTuningSpecification
from src.utils.transformer_pooling_utils import detect_pooling_method, pool_embeddings, PoolingMethod
from src.models.model_outputs_cache import ModelOutputsCache
from src.models import model_loading
from src.utils.ranking_loss import PairwiseRankingLossType, compute_pairwise_ranking_loss
from src.analysis.training_diagnostics import EpochDiagnosticsAccumulator, split_tensor_with_grad
from src.preprocessing.scoring_feature_extraction import get_feature_descriptions


class TransformerEmbeddingModel(ScoringModelBase):
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
        base_model_name: str | None = None,
        run_name: str | None = None,
        print_every: int | None = None,
        save_every: int | None = None,
        checkpoint_name: str | None = None,
        seed: int = 42,
        ranking_loss_type: PairwiseRankingLossType = "margin_ranking",
        proj_dim: int = 64,
    ) -> None:
        super().__init__(run_name)

        self.ranking_loss_type = ranking_loss_type
        self.proj_dim = proj_dim
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

        self._base_model_name: str | None = base_model_name
        self._model_outputs_cache: ModelOutputsCache | None = None
        
        self._embedding_model_source: str | None = load_embedding_model_from
        self._prompt_features_dim: int | None = None
        self._network: TransformerEmbeddingModel._TransformerNetwork | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._history_entries: list[TrainingHistoryEntry] = []
        self._pooling_method: PoolingMethod | None = None
        self._optimizer_state: dict[str, Any] | None = None
        self._scheduler_state: dict[str, Any] | None = None
        self._epochs_completed: int = 0
        self._prompt_features_scaler: SimpleScaler | None = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._best_model_tracker = BestModelTracker()
        
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
        self._tokenizer = AutoTokenizer.from_pretrained(self.transformer_model_name, use_fast=("deberta" not in self.transformer_model_name)) # TODO: Fix this - fast tokenizer fails for deberta models
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
            proj_dim=self.proj_dim,
            quiet=self.print_every is None,
        ).to(self.device)

    def get_config_for_logging(self) -> dict[str, Any]:
        """Get configuration dictionary for training logging."""
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
            "embedding_type": self.embedding_model.embedding_type,
            "embedding_spec": self.embedding_spec.model_dump() if self.embedding_spec is not None else None,
            "min_model_comparisons": self.min_model_comparisons,
            "embedding_model_epochs": self.embedding_model_epochs,
            "scoring_head_lr_multiplier": self.scoring_head_lr_multiplier,
            "base_model": self._base_model_name,
            "ranking_loss_type": self.ranking_loss_type,
            "proj_dim": self.proj_dim,
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
            
            with Timer("load_base_model", verbosity="start+end", parent=train_timer):
                self._load_base_model()
            
            with Timer("init_or_load_embedding_model", verbosity="start+end", parent=train_timer):
                if self.embedding_model is None:
                    if self._embedding_model_source is not None:
                        self.embedding_model = model_loading.load_embedding_model_from_model(self._embedding_model_source)
                    elif self.embedding_spec is not None:
                        self.embedding_model = self.embedding_spec.create_model(
                            min_model_comparisons=self.min_model_comparisons,
                            preprocessor_seed=self.seed,
                            print_every=self.print_every,
                        )
                    else:
                        raise RuntimeError("No embedding model available and no way to create one")
            
            with Timer("preprocess", verbosity="start+end", parent=train_timer):
                preprocessed_data = self.preprocessor.preprocess(data)

            self._prompt_features_scaler = SimpleScaler.from_state_dict(preprocessed_data.scaler_state)
            if self._network is None:
                self._initialize_network(
                    prompt_features_dim=preprocessed_data.prompt_features_dim,
                )

            self.init_logger_if_needed()
            self.embedding_model.set_training_logger(self._logger)

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
                    filtered_indexes=preprocessed_data.filtered_indexes,
                    scaler_state=preprocessed_data.scaler_state,
                )

            with Timer("cache_base_model_predictions", verbosity="start+end", parent=train_timer):
                if self._model_outputs_cache is not None:
                    self._model_outputs_cache.compute_and_cache(
                        entries=[data.entries[i] for i in preprocessed_data.filtered_indexes],
                        indexes=preprocessed_data.filtered_indexes,
                        timer=train_timer,
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

            with Timer("epochs", verbosity="start+end", parent=train_timer) as epochs_timer:
                for epoch in range(self._epochs_completed + 1, self._epochs_completed + epochs + 1):
                    result = self._train_epoch(epoch, dataloader, val_dataloader, optimizer, epochs_timer)
                    
                    self._log_epoch_result(result)
                    
                    self._best_model_tracker.record_state(
                        accuracy=result.val_accuracy if result.val_accuracy is not None else result.train_accuracy,
                        state_dict=self.get_state_dict(),
                        epoch=epoch
                    )
                    
                    if scheduler is not None:
                        scheduler.step()
                    
                    self._epochs_completed = epoch
                    
                    if self.save_every is not None and epoch % self.save_every == 0:
                        checkpoint_name = f"{self.checkpoint_name}@ep{epoch}"
                        print(f"Saving checkpoint: {checkpoint_name}")
                        self.save(checkpoint_name)
            
            # Revert to best model parameters if available
            if self._best_model_tracker.has_best_state:
                print(f"\nReverting to best model parameters from epoch {self._best_model_tracker.best_epoch} (accuracy={self._best_model_tracker.best_accuracy:.4f})")
                self.load_state_dict(self._best_model_tracker.best_state_dict, instance=self)
            
            self._optimizer_state = optimizer.state_dict()
            self._scheduler_state = scheduler.state_dict() if scheduler is not None else None

            with Timer("sensitivity_analysis", verbosity="start+end", parent=train_timer):
                sensitivity_metrics = TransformerEmbeddingModel._SensitivityAnalysis.compute(
                    self.network,
                    train_dataloader=dataloader,
                    val_dataloader=val_dataloader,
                    seed=self.seed,
                )

            final_metrics = {
                "best_epoch": self._best_model_tracker.best_epoch,
                "best_accuracy": self._best_model_tracker.best_accuracy,
                "total_epochs": self._epochs_completed,
                **sensitivity_metrics,
            }

        self.finish_logger_if_needed(
            final_metrics=final_metrics,
            log_timings_from=self.last_timer,
        )

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

            with Timer("predict_base_model", verbosity="start+end", parent=predict_timer):
                base_result = self._model_outputs_cache.predict(X, batch_size=batch_size) \
                    if self._model_outputs_cache is not None else None

            with Timer("preprocess_input", verbosity="start+end", parent=predict_timer):
                preprocessed_input = self.preprocessor.preprocess_for_inference(
                    prompts=X.prompts,
                    model_names=X.model_names,
                    model_embeddings=self.model_embeddings,
                    scaler=self._prompt_features_scaler,
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

                # Add base model scores if available
                if base_result is not None:
                    accumulated = {}
                    for model_name, computed_scores in scores_dict.items():
                        base_scores = base_result.scores.get(model_name, np.zeros_like(computed_scores))
                        accumulated[model_name] = computed_scores + base_scores
                    scores_dict = accumulated
            
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
            "network_state_dict": state_dict_to_cpu(self.network.state_dict()),
            "epochs_completed": self._epochs_completed,
            "prompt_features_scaler_state": self._prompt_features_scaler.get_state_dict(),
            
            "embedding_type": self.embedding_model.embedding_type,
            "embedding_spec": self.embedding_spec.model_dump() if self.embedding_spec is not None else None,
            "min_model_comparisons": self.min_model_comparisons,
            "embedding_model_epochs": self.embedding_model_epochs,
            "embedding_model_state_dict": self.embedding_model.get_state_dict(),
            "seed": self.seed,
            "scoring_head_lr_multiplier": self.scoring_head_lr_multiplier,

            "base_model_name": self._base_model_name,
            "base_model_state_dict": self._model_outputs_cache.model.get_state_dict() if self._model_outputs_cache is not None else None,
            "ranking_loss_type": self.ranking_loss_type,
            "proj_dim": self.proj_dim,
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any], instance: "TransformerEmbeddingModel | None" = None) -> "TransformerEmbeddingModel":
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
            
            # Parse embedding spec using Pydantic TypeAdapter
            embedding_spec_adapter = TypeAdapter(EmbeddingSpec)
            embedding_spec = embedding_spec_adapter.validate_python(state_dict["embedding_spec"]) \
                if state_dict["embedding_spec"] is not None else None
            
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
                base_model_name=state_dict.get("base_model_name", None),
                print_every=state_dict["print_every"],
                save_every=state_dict.get("save_every", None),
                checkpoint_name=state_dict.get("checkpoint_name", None),
                seed=state_dict["seed"],
                ranking_loss_type=state_dict.get("ranking_loss_type", "margin_ranking"),
                proj_dim=state_dict["proj_dim"],
            )
        
        model.embedding_model = EmbeddingModelBase.load_from_state_dict(state_dict["embedding_model_state_dict"])
        model._prompt_features_scaler = SimpleScaler.from_state_dict(state_dict["prompt_features_scaler_state"])

        if model._network is None:
            model._initialize_network(
                prompt_features_dim=state_dict["prompt_features_dim"],
            )
        model.network.load_state_dict(
            state_dict["network_state_dict"],
            strict=False,
        )
        model.network.to(model.device)
        
        model._optimizer_state = state_dict.get("optimizer_state")
        model._scheduler_state = state_dict.get("scheduler_state")
        model._epochs_completed = state_dict.get("epochs_completed", 0)
        
        if model._base_model_name is not None:
            model_type, _ = model._base_model_name.split("/", 1)
            base_model = model_loading.load_scoring_model_from_state_dict(model_type, state_dict["base_model_state_dict"])
            model._model_outputs_cache = ModelOutputsCache(base_model, quiet=model.print_every is not None)
        
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
            indexes=preprocessed_data.filtered_indexes,
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
            "original_indices": torch.tensor([item["original_index"] for item in batch], dtype=torch.long),  # [batch_size]
            "model_ids_a": torch.tensor([item["model_id_a"] for item in batch], dtype=torch.long),  # [batch_size]
            "model_ids_b": torch.tensor([item["model_id_b"] for item in batch], dtype=torch.long),  # [batch_size]
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
        epochs_timer: Timer,
    ) -> "TransformerEmbeddingModel.EpochResult":
        with Timer(f"epoch_{epoch}", verbosity="start+end", parent=epochs_timer) as timer:
            self.network.train()
            total_loss = 0.0
            total_accuracy = 0.0
            n_batches = 0
            diagnostic_accumulator = EpochDiagnosticsAccumulator()
            transformer_params, projection_params, scoring_head_params = self.network.get_param_groups_for_metrics()
            numeric_descs, bool_descs = get_feature_descriptions()
            prompt_feature_names = [d.name for d in numeric_descs] + [d.name for d in bool_descs]

            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)  # [batch_size, seq_len]
                attention_mask = batch["attention_mask"].to(self.device)  # [batch_size, seq_len]
                prompt_features = batch["prompt_features"].to(self.device).requires_grad_(True)  # [batch_size, prompt_features_dim]
                model_embedding_a = batch["model_embedding_a"].to(self.device).requires_grad_(True)  # [batch_size, model_embedding_dim]
                model_embedding_b = batch["model_embedding_b"].to(self.device)  # [batch_size, model_embedding_dim]
                labels = batch["labels"].to(self.device)  # [batch_size]
                model_ids_a = batch["model_ids_a"].to(self.device)  # [batch_size]
                model_ids_b = batch["model_ids_b"].to(self.device)  # [batch_size]

                optimizer.zero_grad()
                prompt_embedding = self.network.forward_encode(input_ids, attention_mask)  # [batch_size, transformer_hidden_size]
                prompt_embedding.retain_grad() # Needed to retain gradient in non-leaf tensor after backward pass
                scores_a = self.network.forward_score(
                    prompt_embedding,
                    prompt_features,
                    model_embedding_a,
                )  # [batch_size]
                scores_b = self.network.forward_score(
                    prompt_embedding,
                    prompt_features,
                    model_embedding_b,
                )  # [batch_size]

                scores_a, scores_b = self._augment_with_base_scores(scores_a, scores_b, batch["original_indices"])

                with torch.no_grad():
                    prompt_repr = self.network.prompt_emb_proj(prompt_embedding)   # [batch_size, 2 * proj_dim]
                    feat_repr = self.network.feat_proj(prompt_features)            # [batch_size, proj_dim]
                    model_repr = self.network.model_proj(model_embedding_a)        # [batch_size, 3 * proj_dim]
                    interaction = torch.cat([prompt_repr, feat_repr], dim=1) * model_repr  # [batch_size, 3 * proj_dim]
                diagnostic_accumulator.update_representation_stats({
                    "prompt_emb_proj": prompt_repr,
                    "feat_proj": feat_repr,
                    "model_proj": model_repr,
                    "interaction": interaction,
                })

                loss: torch.Tensor = compute_pairwise_ranking_loss(
                    self.ranking_loss_type, scores_a, scores_b, labels, margin=0.1
                )
                loss.backward()

                diagnostic_accumulator.update_grad_norms({
                    "transformer": transformer_params,
                    "projection": projection_params,
                    "scoring_head": scoring_head_params,
                })
                diagnostic_accumulator.update_gradient_attribution({
                    **split_tensor_with_grad(prompt_features, prompt_feature_names),
                    "prompt_embedding": prompt_embedding,
                    "model_embedding": model_embedding_a,
                })
                diagnostic_accumulator.update_score_variance(
                    torch.cat([scores_a.detach(), scores_b.detach()]),  # [2 * batch_size]
                    torch.cat([model_ids_a, model_ids_b]),              # [2 * batch_size]
                )

                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                with torch.no_grad():
                    total_accuracy += compute_pairwise_accuracy(scores_a, scores_b, labels)
                n_batches += 1

            avg_loss = total_loss / n_batches
            avg_accuracy = total_accuracy / n_batches
            additional_metrics = diagnostic_accumulator.to_dict()

            with Timer("perform_validation", verbosity="start+end", parent=timer):
                val_loss, val_accuracy = self._perform_validation(val_dataloader, timer) if val_dataloader is not None else (None, None)

            entry = TrainingHistoryEntry(
                epoch=epoch,
                total_loss=avg_loss,
                val_loss=val_loss,
                train_accuracy=avg_accuracy,
                val_accuracy=val_accuracy,
                additional_metrics=additional_metrics,
            )
            self._history_entries.append(entry)
            
            self.append_entry_to_log(entry, log_timings_from=self.last_timer)
            
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
                
                prompt_embedding = self.network.forward_encode(input_ids, attention_mask)  # [batch_size, transformer_hidden_size]
                scores_a = self.network.forward_score(
                    prompt_embedding,
                    prompt_features,
                    model_embedding_a,
                )  # [batch_size]
                scores_b = self.network.forward_score(
                    prompt_embedding,
                    prompt_features,
                    model_embedding_b,
                )  # [batch_size]

                scores_a, scores_b = self._augment_with_base_scores(scores_a, scores_b, batch["original_indices"])

                loss: torch.Tensor = compute_pairwise_ranking_loss(
                    self.ranking_loss_type, scores_a, scores_b, labels, margin=0.1
                )
                batch_accuracy = compute_pairwise_accuracy(scores_a, scores_b, labels)
                
                total_loss += loss.item()
                total_accuracy += batch_accuracy
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_accuracy = total_accuracy / n_batches
        return avg_loss, avg_accuracy

    def _augment_with_base_scores(
        self, 
        scores_a: torch.Tensor, 
        scores_b: torch.Tensor, 
        original_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._model_outputs_cache is None:
            return scores_a, scores_b

        assert len(scores_a) == len(scores_b) == len(original_indices), "Number of scores and original indices must match"

        base_scores_a, base_scores_b = self._model_outputs_cache.get_base_scores(original_indices.cpu().numpy())
        scores_a = scores_a + torch.tensor(base_scores_a, device=self.device)  # [batch_size]
        scores_b = scores_b + torch.tensor(base_scores_b, device=self.device)  # [batch_size]

        return scores_a, scores_b
    
    def _load_base_model(self) -> None:
        """Load the base model from the specification string."""
        if self._base_model_name is None or self._model_outputs_cache is not None:
            return
        
        if "/" not in self._base_model_name:
            raise ValueError("Base model must be of the form 'model_type/model_name'")
                
        model_type, model_name = self._base_model_name.split("/", 1)
        
        if self.print_every is not None:
            print(f"Loading base model: {self._base_model_name}")
        
        loaded_model = model_loading.load_scoring_model(model_type, model_name)
        self._model_outputs_cache = ModelOutputsCache(
            model=loaded_model,
            quiet=self.print_every is not None,
        )
        
        if self.print_every is not None:
            print(f"Base model loaded")

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
            indexes: list[int],
            max_length: int,
        ) -> None:
            assert len(pairs) == len(indexes), "Number of pairs and indexes must match"

            self.pairs = pairs
            self.indexes = indexes
            self.max_length = max_length
        
        def __len__(self) -> int:
            return len(self.pairs)
        
        def __getitem__(self, idx: int) -> dict:
            pair = self.pairs[idx]
            original_index = self.indexes[idx]

            # Return raw data, tokenization happens in collate_fn
            return {
                "prompt": pair.prompt,
                "prompt_features": torch.from_numpy(pair.prompt_features),
                "model_embedding_a": torch.from_numpy(pair.model_embedding_a),
                "model_embedding_b": torch.from_numpy(pair.model_embedding_b),
                "label": torch.tensor(1.0 if pair.winner_label == 0 else -1.0, dtype=torch.float32),
                "original_index": original_index,
                "model_id_a": pair.model_id_a,
                "model_id_b": pair.model_id_b,
            }

    class _TransformerNetwork(nn.Module):
        """Transformer-based network with two-tower interaction architecture."""

        def __init__(
            self,
            transformer_model_name: str,
            prompt_features_dim: int,
            model_embedding_dim: int,
            finetuning_spec: FineTuningSpec,
            hidden_dims: list[int],
            dropout: float,
            pooling_method: PoolingMethod,
            proj_dim: int = 64,
            quiet: bool = False,
        ) -> None:
            super().__init__()

            self.pooling_method = pooling_method
            self.proj_dim = proj_dim

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

            self.prompt_emb_proj = nn.Sequential(
                nn.Linear(transformer_hidden_size, 2 * proj_dim),
                nn.LeakyReLU(0.1),
            )
            self.feat_proj = nn.Sequential(
                nn.Linear(prompt_features_dim, proj_dim),
                nn.LeakyReLU(0.1),
            )
            self.model_proj = nn.Sequential(
                nn.Linear(model_embedding_dim, 3 * proj_dim),
                nn.LeakyReLU(0.1),
            )
            scoring_head_input_dim = 3 * proj_dim

            layers = []
            prev_dim = scoring_head_input_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.LeakyReLU(0.1))
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, 1))

            self.scoring_head = nn.Sequential(*layers)

        def forward_encode(
            self,
            input_ids: torch.Tensor,  # [batch_size, seq_len]
            attention_mask: torch.Tensor,  # [batch_size, seq_len]
        ) -> torch.Tensor:
            outputs: BaseModelOutputWithPast = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            return pool_embeddings(
                outputs.last_hidden_state,  # [batch_size, seq_len, transformer_hidden_size]
                attention_mask,  # [batch_size, seq_len]
                self.pooling_method,
            )  # [batch_size, transformer_hidden_size]

        def forward_score(
            self,
            prompt_embedding: torch.Tensor,  # [batch_size, transformer_hidden_size]
            prompt_features: torch.Tensor,  # [batch_size, prompt_features_dim]
            model_embedding: torch.Tensor,  # [batch_size, model_embedding_dim]
        ) -> torch.Tensor:
            prompt_repr = self.prompt_emb_proj(prompt_embedding)  # [batch_size, 2 * proj_dim]
            feat_repr = self.feat_proj(prompt_features)  # [batch_size, proj_dim]
            model_repr = self.model_proj(model_embedding)  # [batch_size, 3 * proj_dim]

            prompt_combined = torch.cat([prompt_repr, feat_repr], dim=1)  # [batch_size, 3 * proj_dim]
            interaction = prompt_combined * model_repr  # [batch_size, 3 * proj_dim]

            output: torch.Tensor = self.scoring_head(interaction)  # [batch_size, 1]

            return output.squeeze(-1)  # [batch_size]

        def forward(
            self,
            input_ids: torch.Tensor,  # [batch_size, seq_len]
            attention_mask: torch.Tensor,  # [batch_size, seq_len]
            prompt_features: torch.Tensor,  # [batch_size, prompt_features_dim]
            model_embedding: torch.Tensor,  # [batch_size, model_embedding_dim]
        ) -> torch.Tensor:
            prompt_embedding = self.forward_encode(input_ids, attention_mask)
            return self.forward_score(prompt_embedding, prompt_features, model_embedding)

        def get_param_groups_for_metrics(
            self,
        ) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
            """Return (transformer_params, projection_params, scoring_head_params) for gradient norm metrics."""
            transformer_params = []
            projection_params = []
            scoring_head_params = []

            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "transformer" in name:
                    transformer_params.append(param)
                elif "scoring_head" in name:
                    scoring_head_params.append(param)
                elif "prompt_emb_proj" in name or "feat_proj" in name or "model_proj" in name:
                    projection_params.append(param)

            return transformer_params, projection_params, scoring_head_params

    class _SensitivityAnalysis:
        """Prompt sensitivity and feature importance metrics computed on dataloaders.

        Requires a one-time encoding pass to convert raw token batches into prompt
        embeddings.  All perturbations then operate on those pre-computed embeddings,
        so the (expensive) transformer encoder is never called more than once.

        Results are accuracy drops from baseline (positive = the modification hurts
        accuracy, i.e. the model relies on that component).

        A and B sides of each pair share the same prompt, so a single embedding is
        stored per pair and used for both scoring calls.
        """

        # Encoded dataloader element: (prompt_emb, prompt_feat, model_emb_a, model_emb_b, labels)
        _EncodedBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

        @classmethod
        def compute(
            cls,
            network: "TransformerEmbeddingModel._TransformerNetwork",
            train_dataloader: "DataLoader[dict[str, torch.Tensor]]",
            val_dataloader: "DataLoader[dict[str, torch.Tensor]] | None",
            seed: int = 42,
        ) -> dict[str, float]:
            """Compute all sensitivity metrics on available dataloaders.

            Returns a flat dict of accuracy drops keyed by
            ``sensitivity/<metric>_train`` and ``sensitivity/<metric>_val``.
            Val entries are omitted when val_dataloader is None.
            """
            numeric_descs, boolean_descs = get_feature_descriptions()
            feature_names = [d.name for d in numeric_descs] + [d.name for d in boolean_descs]

            results: dict[str, float] = {}
            for suffix, raw_dataloader in [("_train", train_dataloader), ("_val", val_dataloader)]:
                if raw_dataloader is None:
                    continue

                encoded_dl = cls._encode_dataloader(network, raw_dataloader)
                global_mean_emb, global_mean_feat = cls._compute_global_means(network, encoded_dl)
                baseline = cls._compute_accuracy(network, encoded_dl)

                results[f"sensitivity/prompt{suffix}"] = baseline - cls._compute_accuracy(
                    network, encoded_dl,
                    lambda e, f: cls._shuffle_prompts(e, f, np.random.default_rng(seed)),
                )
                results[f"sensitivity/prompt_embedding{suffix}"] = baseline - cls._compute_accuracy(
                    network, encoded_dl,
                    lambda e, f: cls._set_emb_to_mean(e, f, global_mean_emb),
                )
                results[f"sensitivity/prompt_features{suffix}"] = baseline - cls._compute_accuracy(
                    network, encoded_dl,
                    lambda e, f: cls._set_feat_to_mean(e, f, global_mean_feat),
                )
                for idx, name in enumerate(feature_names):
                    results[f"sensitivity/feature/{name}{suffix}"] = baseline - cls._compute_accuracy(
                        network, encoded_dl,
                        lambda e, f: cls._permute_feature(e, f, idx, np.random.default_rng(seed)),
                    )

            return results

        @classmethod
        def _encode_dataloader(
            cls,
            network: "TransformerEmbeddingModel._TransformerNetwork",
            raw_dataloader: "DataLoader[dict[str, torch.Tensor]]",
        ) -> "DataLoader[TransformerEmbeddingModel._SensitivityAnalysis._EncodedBatch]":
            """Encode all prompts in a raw dataloader into a pre-computed embedding dataloader.

            Returns a new DataLoader with tensors
            ``(prompt_emb, prompt_feat, model_emb_a, model_emb_b, labels)``
            where ``prompt_emb`` is the transformer output (shared for both A and B).
            """
            device = next(network.parameters()).device
            all_prompt_embs: list[torch.Tensor] = []
            all_prompt_feats: list[torch.Tensor] = []
            all_model_embs_a: list[torch.Tensor] = []
            all_model_embs_b: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []

            network.eval()
            with torch.no_grad():
                for batch in raw_dataloader:
                    input_ids = batch["input_ids"].to(device)          # [batch, seq_len]
                    attention_mask = batch["attention_mask"].to(device)  # [batch, seq_len]
                    prompt_feat = batch["prompt_features"].to(device)   # [batch, prompt_feat_dim]
                    model_emb_a = batch["model_embedding_a"].to(device)  # [batch, model_emb_dim]
                    model_emb_b = batch["model_embedding_b"].to(device)  # [batch, model_emb_dim]
                    labels = batch["labels"].to(device)                  # [batch]

                    prompt_emb = network.forward_encode(input_ids, attention_mask)  # [batch, d_transformer]

                    all_prompt_embs.append(prompt_emb.cpu())
                    all_prompt_feats.append(prompt_feat.cpu())
                    all_model_embs_a.append(model_emb_a.cpu())
                    all_model_embs_b.append(model_emb_b.cpu())
                    all_labels.append(labels.cpu())

            dataset = TensorDataset(
                torch.cat(all_prompt_embs),   # [n, d_transformer]
                torch.cat(all_prompt_feats),  # [n, prompt_feat_dim]
                torch.cat(all_model_embs_a),  # [n, model_emb_dim]
                torch.cat(all_model_embs_b),  # [n, model_emb_dim]
                torch.cat(all_labels),         # [n]
            )
            batch_size = raw_dataloader.batch_size or 32
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)

        @classmethod
        def _compute_accuracy(
            cls,
            network: "TransformerEmbeddingModel._TransformerNetwork",
            dataloader: "DataLoader[TransformerEmbeddingModel._SensitivityAnalysis._EncodedBatch]",
            modify_prompt: Callable[
                [torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
            ] | None = None,
        ) -> float:
            """Compute accuracy on an encoded dataloader with an optional per-batch prompt modification."""
            device = next(network.parameters()).device
            network.eval()
            total_correct = 0.0
            total_samples = 0

            with torch.no_grad():
                for prompt_emb, prompt_feat, model_emb_a, model_emb_b, labels in dataloader:
                    prompt_emb = prompt_emb.to(device)    # [batch, d_transformer]
                    prompt_feat = prompt_feat.to(device)  # [batch, prompt_feat_dim]
                    model_emb_a = model_emb_a.to(device)  # [batch, model_emb_dim]
                    model_emb_b = model_emb_b.to(device)  # [batch, model_emb_dim]
                    labels = labels.to(device)             # [batch]

                    if modify_prompt is not None:
                        prompt_emb, prompt_feat = modify_prompt(prompt_emb, prompt_feat)

                    score_a = network.forward_score(prompt_emb, prompt_feat, model_emb_a)  # [batch]
                    score_b = network.forward_score(prompt_emb, prompt_feat, model_emb_b)  # [batch]

                    batch_size = labels.shape[0]
                    total_correct += compute_pairwise_accuracy(score_a, score_b, labels) * batch_size
                    total_samples += batch_size

            return total_correct / total_samples if total_samples > 0 else 0.0

        @classmethod
        def _compute_global_means(
            cls,
            network: "TransformerEmbeddingModel._TransformerNetwork",
            dataloader: "DataLoader[TransformerEmbeddingModel._SensitivityAnalysis._EncodedBatch]",
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Compute the dataset-level mean prompt embedding and features.

            Returns:
                (mean_emb, mean_feat) shaped [1, d_transformer] and [1, prompt_feat_dim]
            """
            device = next(network.parameters()).device
            sum_emb: torch.Tensor | None = None
            sum_feat: torch.Tensor | None = None
            total = 0

            with torch.no_grad():
                for prompt_emb, prompt_feat, _, _, _ in dataloader:
                    prompt_emb = prompt_emb.to(device)   # [batch, d_transformer]
                    prompt_feat = prompt_feat.to(device)  # [batch, prompt_feat_dim]
                    if sum_emb is None:
                        sum_emb = prompt_emb.sum(dim=0)
                        sum_feat = prompt_feat.sum(dim=0)
                    else:
                        sum_emb += prompt_emb.sum(dim=0)
                        sum_feat += prompt_feat.sum(dim=0)
                    total += prompt_emb.shape[0]

            return (sum_emb / total).unsqueeze(0), (sum_feat / total).unsqueeze(0)  # [1, d_emb], [1, d_feat]

        @staticmethod
        def _shuffle_prompts(
            emb: torch.Tensor,   # [batch, d_transformer]
            feat: torch.Tensor,  # [batch, prompt_feat_dim]
            rng: np.random.Generator,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # [batch, d_transformer], [batch, prompt_feat_dim]
            idx = torch.from_numpy(rng.permutation(emb.shape[0])).to(emb.device)
            return emb[idx], feat[idx]

        @staticmethod
        def _set_emb_to_mean(
            emb: torch.Tensor,      # [batch, d_transformer]
            feat: torch.Tensor,     # [batch, prompt_feat_dim]
            mean_emb: torch.Tensor, # [1, d_transformer]
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # [batch, d_transformer], [batch, prompt_feat_dim]
            return mean_emb.expand_as(emb), feat

        @staticmethod
        def _set_feat_to_mean(
            emb: torch.Tensor,       # [batch, d_transformer]
            feat: torch.Tensor,      # [batch, prompt_feat_dim]
            mean_feat: torch.Tensor, # [1, prompt_feat_dim]
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # [batch, d_transformer], [batch, prompt_feat_dim]
            return emb, mean_feat.expand_as(feat)

        @staticmethod
        def _permute_feature(
            emb: torch.Tensor,   # [batch, d_transformer]
            feat: torch.Tensor,  # [batch, prompt_feat_dim]
            feature_idx: int,
            rng: np.random.Generator,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # [batch, d_transformer], [batch, prompt_feat_dim]
            perm = torch.from_numpy(rng.permutation(feat.shape[0])).to(feat.device)
            feat = feat.clone()
            feat[:, feature_idx] = feat[perm, feature_idx]
            return emb, feat
