"""Triplet-based model behavior encoder using fine-tunable transformers."""

from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from src.data_models.data_models import TrainingData
from src.data_models.triplet_encoder_types import (
    PreprocessedTripletEncoderData,
    PromptResponsePair,
    TrainingTriplet,
)
from src.preprocessing.triplet_finetunable_encoder_preprocessor import TripletFinetunableEncoderPreprocessor
from src.utils.data_split import split_preprocessed_behavior_data
from src.models.triplet_model_base import TripletModelBase
from src.models.optimizers.optimizer_spec import OptimizerSpecification
from src.models.optimizers.adamw_spec import AdamWSpec


class TripletFinetunableEncoderModel(TripletModelBase[TrainingTriplet]):
    """
    Triplet-based encoder using fine-tunable transformers.
    
    This encoder uses:
    - Fine-tunable transformer (e.g., BERT) for text encoding
    - Optional trainable projection layer on top
    - Triplet margin loss for training
    """
    
    def __init__(
        self,
        transformer_model_name: str = "bert-base-uncased",
        projection_dim: int = 128,
        max_length: int = 256,
        optimizer_spec: OptimizerSpecification | None = None,
        triplet_margin: float = 0.2,
        regularization_weight: float = 0.01,
        min_model_comparisons: int = 20,
        identity_positive_ratio: float = 0.8,
        preprocessor_seed: int = 42,
        print_every: int | None = None,
    ) -> None:
        """
        Initialize the fine-tunable encoder model.
        
        Args:
            transformer_model_name: Name of the HuggingFace transformer model
            projection_dim: Dimension of projection layer
            max_length: Maximum sequence length for tokenization
            optimizer_spec: Optimizer specification (default: AdamW with LR 1e-5)
            triplet_margin: Margin for triplet loss
            regularization_weight: Weight for KL-divergence regularization loss
            min_model_comparisons: Minimum comparisons for a model to be included
            identity_positive_ratio: Ratio of identity vs performance positives
            preprocessor_seed: Random seed for preprocessor
            print_every: Print progress every N epochs (None = no printing)
        """
        super().__init__(
            triplet_margin=triplet_margin,
            regularization_weight=regularization_weight,
            min_model_comparisons=min_model_comparisons,
            identity_positive_ratio=identity_positive_ratio,
            preprocessor_seed=preprocessor_seed,
            print_every=print_every,
        )
        
        self.transformer_model_name = transformer_model_name
        self.projection_dim = projection_dim
        self.max_length = max_length
        
        # Lower learning rate is typical for fine-tuning transformers
        self.optimizer_spec = optimizer_spec if optimizer_spec is not None else AdamWSpec(learning_rate=1e-5)
        
        self.preprocessor = TripletFinetunableEncoderPreprocessor(
            min_model_comparisons=min_model_comparisons,
            identity_positive_ratio=identity_positive_ratio,
            seed=preprocessor_seed,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        self._module: TripletFinetunableEncoderModel._EncoderModule | None = None
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of the output embeddings."""
        return self.projection_dim
    
    def _get_module(self) -> nn.Module:
        """Get the neural network module."""
        return self._module
    
    def _initialize_module(self, input_dim: int) -> None:
        """Initialize the neural network module."""
        # input_dim is not used for transformer model
        self._module = self._EncoderModule(
            transformer_model_name=self.transformer_model_name,
            projection_dim=self.projection_dim,
        ).to(self.device)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for the model."""
        return self.optimizer_spec.create_optimizer(self._module)
    
    def _create_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler | None:
        """Create learning rate scheduler."""
        return self.optimizer_spec.create_scheduler(optimizer)
    
    def _preprocess_data(self, data: TrainingData) -> PreprocessedTripletEncoderData[TrainingTriplet]:
        """Preprocess training data."""
        return self.preprocessor.preprocess(data)
    
    def _split_preprocessed_data(
        self,
        preprocessed_data: PreprocessedTripletEncoderData[TrainingTriplet],
        val_fraction: float,
        seed: int,
    ) -> tuple[PreprocessedTripletEncoderData[TrainingTriplet], PreprocessedTripletEncoderData[TrainingTriplet]]:
        """Split preprocessed data into train and validation sets."""
        return split_preprocessed_behavior_data(preprocessed_data, val_fraction, seed)
    
    def _prepare_dataloader(
        self,
        preprocessed_data: PreprocessedTripletEncoderData[TrainingTriplet],
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Prepare dataloader from preprocessed triplets (text-based).
        
        Args:
            preprocessed_data: Preprocessed triplets (as TrainingTriplet objects)
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader yielding tokenized (anchor, positive, negative) tuples
        """
        dataset = self._TripletTextDataset(
            triplets=preprocessed_data.triplets,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(
        self,
        batch: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Collate function for batching tokenized triplets.
        
        Args:
            batch: List of (anchor_tokens, positive_tokens, negative_tokens) tuples
            
        Returns:
            Tuple of batched (anchor_tokens, positive_tokens, negative_tokens)
        """
        anchors, positives, negatives = zip(*batch)
        
        # Stack input_ids and attention_mask for each component
        anchor_batch = {
            "input_ids": torch.stack([a["input_ids"] for a in anchors]),
            "attention_mask": torch.stack([a["attention_mask"] for a in anchors]),
        }
        positive_batch = {
            "input_ids": torch.stack([p["input_ids"] for p in positives]),
            "attention_mask": torch.stack([p["attention_mask"] for p in positives]),
        }
        negative_batch = {
            "input_ids": torch.stack([n["input_ids"] for n in negatives]),
            "attention_mask": torch.stack([n["attention_mask"] for n in negatives]),
        }
        
        return anchor_batch, positive_batch, negative_batch
    
    def encode(
        self,
        pairs: list[PromptResponsePair],
    ) -> np.ndarray:
        """
        Encode a list of (prompt, response) pairs into embeddings.
        
        Args:
            pairs: List of prompt-response pairs  # [n_samples]
            
        Returns:
            Array of embeddings  # [n_samples, embedding_dim]
        """
        if self._module is None:
            raise RuntimeError("Module not initialized. Train the model first.")
        
        self._module.eval()
        embeddings = []
        
        with torch.no_grad():
            for pair in pairs:
                # Combine prompt and response
                text = f"{pair.prompt} [SEP] {pair.response}"
                
                # Tokenize
                tokens = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                
                # Move to device
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                
                # Encode
                embedding = self._module(tokens)  # [1, embedding_dim]
                embeddings.append(embedding.cpu().numpy())
        
        return np.concatenate(embeddings, axis=0)  # [n_samples, embedding_dim]
    
    def get_state_dict(self) -> dict[str, Any]:
        """Get state dictionary for serialization."""
        if self._module is None:
            raise RuntimeError("Module not initialized. Train the model first.")
        
        return {
            "model_type": "TripletFinetunableEncoderModel",
            "transformer_model_name": self.transformer_model_name,
            "projection_dim": self.projection_dim,
            "max_length": self.max_length,
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "triplet_margin": self.triplet_margin,
            "regularization_weight": self.regularization_weight,
            "min_model_comparisons": self.min_model_comparisons,
            "identity_positive_ratio": self.identity_positive_ratio,
            "preprocessor_seed": self.preprocessor_seed,
            "print_every": self.print_every,
            "module_state_dict": self._module.state_dict(),
            "epoch_logs": self._epoch_logs,
            "model_embeddings": self.model_embeddings,
        }
    
    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "TripletFinetunableEncoderModel":
        """Load model from state dictionary."""
        optimizer_spec = OptimizerSpecification.from_serialized(
            state_dict["optimizer_type"],
            state_dict["optimizer_params"],
        )
        
        model = cls(
            transformer_model_name=state_dict["transformer_model_name"],
            projection_dim=state_dict["projection_dim"],
            max_length=state_dict["max_length"],
            optimizer_spec=optimizer_spec,
            triplet_margin=state_dict["triplet_margin"],
            regularization_weight=state_dict["regularization_weight"],
            min_model_comparisons=state_dict["min_model_comparisons"],
            identity_positive_ratio=state_dict["identity_positive_ratio"],
            preprocessor_seed=state_dict["preprocessor_seed"],
            print_every=state_dict["print_every"],
        )
        
        model._initialize_module(0)  # input_dim not used
        model._module.load_state_dict(state_dict["module_state_dict"])
        model._model_embeddings = state_dict["model_embeddings"]
        model._epoch_logs = state_dict["epoch_logs"]
        
        return model
    
    class _TripletTextDataset(Dataset):
        """Dataset for text-based triplets that tokenizes on-the-fly."""
        
        def __init__(
            self,
            triplets: list[TrainingTriplet],
            tokenizer: AutoTokenizer,
            max_length: int,
        ) -> None:
            self.triplets = triplets
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self) -> int:
            return len(self.triplets)
        
        def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
            triplet = self.triplets[idx]
            
            # Combine prompt and response for each component
            anchor_text = f"{triplet.anchor_prompt} [SEP] {triplet.anchor_response}"
            positive_text = f"{triplet.positive_prompt} [SEP] {triplet.positive_response}"
            negative_text = f"{triplet.negative_prompt} [SEP] {triplet.negative_response}"
            
            # Tokenize each component
            anchor_tokens = self.tokenizer(
                anchor_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            positive_tokens = self.tokenizer(
                positive_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            negative_tokens = self.tokenizer(
                negative_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            # Remove batch dimension added by return_tensors="pt"
            anchor_tokens = {k: v.squeeze(0) for k, v in anchor_tokens.items()}
            positive_tokens = {k: v.squeeze(0) for k, v in positive_tokens.items()}
            negative_tokens = {k: v.squeeze(0) for k, v in negative_tokens.items()}
            
            return anchor_tokens, positive_tokens, negative_tokens
    
    class _EncoderModule(nn.Module):
        """
        Inner PyTorch module for the fine-tunable encoder model.
        
        This module uses a transformer that can be fine-tuned end-to-end.
        """
        
        def __init__(
            self,
            transformer_model_name: str,
            projection_dim: int | None,
        ) -> None:
            """
            Initialize the module.
            
            Args:
                transformer_model_name: Name of the HuggingFace transformer model
                projection_dim: Dimension of projection layer
            """
            super().__init__()
            
            self.transformer = AutoModel.from_pretrained(transformer_model_name)
            self.transformer_hidden_size = self.transformer.config.hidden_size
            self.projection_dim = projection_dim
            
            self.projection = nn.Sequential(
                nn.Linear(self.transformer_hidden_size, projection_dim),
                nn.Tanh(),
            )
        
        def forward(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
            """
            Forward pass through the transformer and optional projection.
            
            Args:
                tokens: Dictionary with 'input_ids' and 'attention_mask'
                    input_ids: [batch_size, seq_length]
                    attention_mask: [batch_size, seq_length]
                
            Returns:
                Output embeddings  # [batch_size, embedding_dim]
            """
            # Get transformer outputs
            outputs = self.transformer(**tokens)
            
            # Use [CLS] token embedding (first token)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, transformer_hidden_size]
            
            return self.projection(cls_embedding)  # [batch_size, projection_dim]

