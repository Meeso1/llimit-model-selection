"""Triplet-based model behavior encoder using frozen sentence transformers."""

from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.data_models.data_models import TrainingData
from src.data_models.triplet_encoder_types import (
    PreprocessedTripletEncoderData,
    PromptResponsePair,
    TripletEmbedding,
)
from src.preprocessing.triplet_frozen_encoder_preprocessor import TripletFrozenEncoderPreprocessor
from src.utils.data_split import split_preprocessed_behavior_data
from src.models.triplet_model_base import TripletModelBase
from src.models.optimizers.optimizer_spec import OptimizerSpecification
from src.models.optimizers.adamw_spec import AdamWSpec


class TripletFrozenEncoderModel(TripletModelBase[TripletEmbedding]):
    """
    Triplet-based encoder using frozen sentence transformers with trainable dense layers.
    
    This encoder uses:
    - Frozen text encoder (sentence transformer) for initial embeddings
    - Trainable dense network on top for learned embedding space
    - Triplet margin loss for training
    """
    
    def __init__(
        self,
        encoder_model_name: str = "all-MiniLM-L6-v2",
        hidden_dims: list[int] | None = None,
        optimizer_spec: OptimizerSpecification | None = None,
        triplet_margin: float = 0.2,
        regularization_weight: float = 0.01,
        min_model_comparisons: int = 20,
        identity_positive_ratio: float = 0.8,
        preprocessor_seed: int = 42,
        print_every: int | None = None,
    ) -> None:
        """
        Initialize the frozen encoder model.
        
        Args:
            encoder_model_name: Name of the sentence transformer model for text encoding
            hidden_dims: Dimensions of trainable dense layers after frozen transformer
            optimizer_spec: Optimizer specification (default: AdamW with LR 0.001)
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
        
        self.encoder_model_name = encoder_model_name
        self.hidden_dims = hidden_dims if hidden_dims is not None else [256, 128]
        
        if not self.hidden_dims:
            raise ValueError("hidden_dims cannot be empty - the model must have trainable layers")
        
        self.optimizer_spec = optimizer_spec if optimizer_spec is not None else AdamWSpec(learning_rate=0.001)
        
        self.preprocessor = TripletFrozenEncoderPreprocessor(
            min_model_comparisons=min_model_comparisons,
            identity_positive_ratio=identity_positive_ratio,
            embedding_model_name=encoder_model_name,
            seed=preprocessor_seed,
        )
        
        self.input_dim: int | None = None
        self._module: TripletFrozenEncoderModel._EncoderModule | None = None
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of the output embeddings."""
        return self.hidden_dims[-1]
    
    def _get_module(self) -> nn.Module:
        """Get the neural network module."""
        return self._module
    
    def _initialize_module(self, input_dim: int | None = None) -> None:
        """Initialize the neural network module."""
        if input_dim is None:
            raise ValueError("input_dim cannot be None for frozen encoder model")
        
        self.input_dim = input_dim
        self._module = self._EncoderModule(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
        ).to(self.device)
    
    def _infer_input_dim(self, first_batch: Any) -> int:
        """Infer the input dimension from the first batch."""
        if not isinstance(first_batch[0], torch.Tensor):
            raise ValueError("first_batch[0] is not a torch.Tensor")

        return first_batch[0].shape[1]
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for the model."""
        return self.optimizer_spec.create_optimizer(self._module)
    
    def _create_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler | None:
        """Create learning rate scheduler."""
        return self.optimizer_spec.create_scheduler(optimizer)
    
    def _preprocess_data(self, data: TrainingData) -> PreprocessedTripletEncoderData[TripletEmbedding]:
        """Preprocess training data."""
        return self.preprocessor.preprocess(data)
    
    def _split_preprocessed_data(
        self,
        preprocessed_data: PreprocessedTripletEncoderData[TripletEmbedding],
        val_fraction: float,
        seed: int,
    ) -> tuple[PreprocessedTripletEncoderData[TripletEmbedding], PreprocessedTripletEncoderData[TripletEmbedding]]:
        """Split preprocessed data into train and validation sets."""
        return split_preprocessed_behavior_data(preprocessed_data, val_fraction, seed)
    
    def _prepare_dataloader(
        self,
        preprocessed_data: PreprocessedTripletEncoderData[TripletEmbedding],
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Prepare dataloader from preprocessed triplets embeddings.
        
        Args:
            preprocessed_data: Preprocessed triplets embeddings
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader yielding (anchor_emb, positive_emb, negative_emb) tuples
        """    
        anchor_embeddings = torch.stack([
            torch.cat([
                torch.from_numpy(triplet.anchor_prompt),
                torch.from_numpy(triplet.anchor_response),
            ])
            for triplet in preprocessed_data.triplets
        ])  # [n_triplets, 2 * embedding_dim]
        positive_embeddings = torch.stack([
            torch.cat([
                torch.from_numpy(triplet.positive_prompt),
                torch.from_numpy(triplet.positive_response),
            ])
            for triplet in preprocessed_data.triplets
        ])  # [n_triplets, 2 * embedding_dim]
        negative_embeddings = torch.stack([
            torch.cat([
                torch.from_numpy(triplet.negative_prompt),
                torch.from_numpy(triplet.negative_response),
            ])
            for triplet in preprocessed_data.triplets
        ])  # [n_triplets, 2 * embedding_dim]
        
        dataset = TensorDataset(
            anchor_embeddings,
            positive_embeddings,
            negative_embeddings,
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    
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
        
        preprocessed_pairs = self.preprocessor.preprocess_for_inference(pairs)
        embeddings = torch.stack([
            torch.cat([
                torch.from_numpy(pair.prompt), 
                torch.from_numpy(pair.response),
            ])
            for pair in preprocessed_pairs
        ]).to(self.device)  # [n_samples, 2 * base_embedding_dim]
        
        with torch.no_grad():
            embeddings = self._module(embeddings)  # [n_samples, output_dim]

        embeddings = embeddings.cpu().numpy()  # [n_samples, embedding_dim]
        return embeddings
    
    def get_state_dict(self) -> dict[str, Any]:
        """Get state dictionary for serialization."""
        if self._module is None:
            raise RuntimeError("Module not initialized. Train the model first.")
        
        return {
            "model_type": "TripletFrozenEncoderModel",
            "encoder_model_name": self.encoder_model_name,
            "hidden_dims": self.hidden_dims,
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
            "input_dim": self.input_dim,
            "model_embeddings": self.model_embeddings,
        }
    
    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "TripletFrozenEncoderModel":
        """Load model from state dictionary."""
        optimizer_spec = OptimizerSpecification.from_serialized(
            state_dict["optimizer_type"],
            state_dict["optimizer_params"],
        )
        
        model = cls(
            encoder_model_name=state_dict["encoder_model_name"],
            hidden_dims=state_dict["hidden_dims"],
            optimizer_spec=optimizer_spec,
            triplet_margin=state_dict["triplet_margin"],
            regularization_weight=state_dict["regularization_weight"],
            min_model_comparisons=state_dict["min_model_comparisons"],
            identity_positive_ratio=state_dict["identity_positive_ratio"],
            preprocessor_seed=state_dict["preprocessor_seed"],
            print_every=state_dict["print_every"],
        )
        
        model._initialize_module(state_dict["input_dim"])
        model._module.load_state_dict(state_dict["module_state_dict"])
        model._model_embeddings = state_dict["model_embeddings"]
        model._epoch_logs = state_dict["epoch_logs"]
        
        return model
    
    class _EncoderModule(nn.Module):
        """
        Inner PyTorch module for the frozen encoder model.
        
        This is a trainable dense network that projects the frozen text encoder
        embeddings to a learned embedding space.
        """
        
        def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
            """
            Initialize the module.
            
            Args:
                input_dim: Input embedding dimension from frozen text encoder
                hidden_dims: List of hidden layer dimensions
            """
            super().__init__()
            
            self.input_dim = input_dim
            self.hidden_dims = hidden_dims
            self.output_dim = hidden_dims[-1]
            
            # Build dense network with hidden layers
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.LeakyReLU(0.1))
                layers.append(nn.Dropout(0.1))
                prev_dim = hidden_dim
            
            # Remove last dropout
            layers = layers[:-1]
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through the network.
            
            Args:
                x: Input embeddings  # [batch_size, input_dim]
                
            Returns:
                Output embeddings  # [batch_size, output_dim]
            """
            return self.network(x)

