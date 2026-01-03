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
from src.utils.accuracy import compute_embedding_accuracy
from src.utils.data_split import split_preprocessed_behavior_data
from src.utils.timer import Timer
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
    ) -> DataLoader:
        """
        Prepare dataloader from preprocessed triplets embeddings.
        
        Args:
            preprocessed_data: Preprocessed triplets embeddings
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader yielding (anchor_emb, positive_emb, negative_emb, anchor_model_ids) tuples
        """    
        class TripletDataset:
            def __init__(self, triplets):
                self.triplets = triplets
                
            def __len__(self):
                return len(self.triplets)
                
            def __getitem__(self, idx):
                triplet = self.triplets[idx]
                anchor = torch.cat([
                    torch.from_numpy(triplet.anchor_prompt),
                    torch.from_numpy(triplet.anchor_response),
                ])
                positive = torch.cat([
                    torch.from_numpy(triplet.positive_prompt),
                    torch.from_numpy(triplet.positive_response),
                ])
                negative = torch.cat([
                    torch.from_numpy(triplet.negative_prompt),
                    torch.from_numpy(triplet.negative_response),
                ])
                return anchor, positive, negative, triplet.anchor_model_id
        
        dataset = TripletDataset(preprocessed_data.triplets)
        
        def collate_fn(batch):
            anchors, positives, negatives, model_ids = zip(*batch)
            return (
                torch.stack(anchors),
                torch.stack(positives),
                torch.stack(negatives),
                list(model_ids)
            )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
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
    
    def _train_epoch(
        self,
        epoch: int,
        dataloader: DataLoader,
        val_dataloader: DataLoader | None,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epochs_timer: Timer,
    ) -> "TripletModelBase.EpochLog":
        """Train for one epoch with pre-computed embeddings."""        
        with Timer(f"epoch_{epoch}", verbosity="start+end", parent=epochs_timer) as timer:
            module = self._get_module()
            module.train()
            total_triplet_loss = 0.0
            total_reg_loss = 0.0
            total_loss = 0.0
            correct_triplets = 0
            n_batches = 0
            total_samples = 0
            
            # Store anchor embeddings and model IDs for universal accuracy
            train_anchor_embeddings = []
            train_anchor_model_ids = []
            
            for batch_anchor, batch_positive, batch_negative, batch_anchor_model_ids_list in dataloader:
                batch_anchor = batch_anchor.to(self.device)  # [batch_size, input_dim]
                batch_positive = batch_positive.to(self.device)  # [batch_size, input_dim]
                batch_negative = batch_negative.to(self.device)  # [batch_size, input_dim]
                
                batch_size_actual = len(batch_anchor)
                total_samples += batch_size_actual
                
                optimizer.zero_grad()
                
                # Forward pass
                anchor_emb = module(batch_anchor)  # [batch_size, output_dim]
                positive_emb = module(batch_positive)  # [batch_size, output_dim]
                negative_emb = module(batch_negative)  # [batch_size, output_dim]
                
                # Triplet loss
                triplet_loss = criterion(anchor_emb, positive_emb, negative_emb)
                
                # Regularization loss: KL divergence to encourage normal distribution
                reg_loss = self._compute_regularization_loss(
                    torch.cat([anchor_emb, positive_emb, negative_emb], dim=0)
                )
                
                # Total loss
                loss = triplet_loss + self.regularization_weight * reg_loss
                loss.backward()
                optimizer.step()
                
                total_triplet_loss += triplet_loss.item()
                total_reg_loss += reg_loss.item()
                total_loss += loss.item()
                
                # Compute triplet accuracy
                with torch.no_grad():
                    dist_pos = torch.norm(anchor_emb - positive_emb, p=2, dim=1)  # [batch_size]
                    dist_neg = torch.norm(anchor_emb - negative_emb, p=2, dim=1)  # [batch_size]
                    correct_triplets += (dist_pos < dist_neg).sum().item()
                    
                    # Store anchor embeddings and model IDs
                    train_anchor_embeddings.append(anchor_emb.detach())
                    train_anchor_model_ids.extend(batch_anchor_model_ids_list)
                
                n_batches += 1
            
            avg_triplet_loss = total_triplet_loss / n_batches
            avg_reg_loss = total_reg_loss / n_batches
            avg_loss = total_loss / n_batches
            triplet_accuracy = correct_triplets / total_samples
            
            model_embeddings = None

            # TODO: Make this a parameter
            compute_embeddings_every_k_epochs = 5
            if (epoch % compute_embeddings_every_k_epochs == 0) or (model_embeddings is None):
                train_anchor_embeddings_tensor = torch.cat(train_anchor_embeddings, dim=0)
                with Timer(f"compute_model_embeddings_{epoch}", verbosity="start+end", parent=timer):
                    model_embeddings = self._compute_model_embeddings_from_stored(
                        train_anchor_embeddings_tensor, train_anchor_model_ids
                    )
            
            # Compute universal accuracy for training data
            train_anchor_embeddings_tensor = torch.cat(train_anchor_embeddings, dim=0)
            train_universal_accuracy = compute_embedding_accuracy(
                sample_embeddings=train_anchor_embeddings_tensor.detach().cpu().numpy(),
                sample_model_names=train_anchor_model_ids,
                model_embeddings=model_embeddings
            )
            
            # Validation
            val_loss, val_triplet_accuracy, val_universal_accuracy = self._perform_validation(
                val_dataloader, criterion, model_embeddings, timer
            ) if val_dataloader is not None else (None, None, None)
        
        epoch_log = self.EpochLog(
            epoch=epoch,
            train_loss=avg_loss,
            train_triplet_loss=avg_triplet_loss,
            train_reg_loss=avg_reg_loss,
            train_triplet_accuracy=triplet_accuracy,
            train_universal_accuracy=train_universal_accuracy,
            val_loss=val_loss,
            val_triplet_accuracy=val_triplet_accuracy,
            val_universal_accuracy=val_universal_accuracy,
            duration=timer.elapsed_time,
        )
        self._epoch_logs.append(epoch_log)
        
        return epoch_log
    
    def _perform_validation(
        self,
        val_dataloader: DataLoader,
        criterion: nn.Module,
        model_embeddings: dict[str, np.ndarray],
        timer: Timer,
    ) -> tuple[float, float, float]:
        """Perform validation and return (loss, triplet_accuracy, universal_accuracy)."""
        module = self._get_module()
        module.eval()
        total_loss = 0.0
        correct_triplets = 0
        n_batches = 0
        total_samples = 0
        
        # Store anchor embeddings and model IDs for universal accuracy
        val_anchor_embeddings = []
        val_anchor_model_ids = []
        
        with torch.no_grad():
            for batch_anchor, batch_positive, batch_negative, batch_anchor_model_ids_list in val_dataloader:
                batch_anchor = batch_anchor.to(self.device)  # [batch_size, input_dim]
                batch_positive = batch_positive.to(self.device)  # [batch_size, input_dim]
                batch_negative = batch_negative.to(self.device)  # [batch_size, input_dim]
                
                batch_size_actual = len(batch_anchor)
                total_samples += batch_size_actual
                
                # Forward pass
                anchor_emb = module(batch_anchor)  # [batch_size, output_dim]
                positive_emb = module(batch_positive)  # [batch_size, output_dim]
                negative_emb = module(batch_negative)  # [batch_size, output_dim]
                
                # Triplet loss
                triplet_loss = criterion(anchor_emb, positive_emb, negative_emb)
                
                # Regularization loss
                reg_loss = self._compute_regularization_loss(
                    torch.cat([anchor_emb, positive_emb, negative_emb], dim=0)
                )
                
                # Total loss
                loss = triplet_loss + self.regularization_weight * reg_loss
                total_loss += loss.item()
                
                # Compute triplet accuracy
                dist_pos = torch.norm(anchor_emb - positive_emb, p=2, dim=1)  # [batch_size]
                dist_neg = torch.norm(anchor_emb - negative_emb, p=2, dim=1)  # [batch_size]
                correct_triplets += (dist_pos < dist_neg).sum().item()
                
                # Store anchor embeddings and model IDs
                val_anchor_embeddings.append(anchor_emb)
                val_anchor_model_ids.extend(batch_anchor_model_ids_list)
                
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        triplet_accuracy = correct_triplets / total_samples
        
        # Compute universal accuracy for validation data
        val_anchor_embeddings_tensor = torch.cat(val_anchor_embeddings, dim=0)
        universal_accuracy = compute_embedding_accuracy(
            sample_embeddings=val_anchor_embeddings_tensor.detach().cpu().numpy(),
            sample_model_names=val_anchor_model_ids,
            model_embeddings=model_embeddings
        )
        
        return avg_loss, triplet_accuracy, universal_accuracy
    
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
            "module_state_dict": self._module.cpu().state_dict(),
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
        model._module.load_state_dict(
            state_dict["module_state_dict"],
            map_location=model.device,
        )
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

