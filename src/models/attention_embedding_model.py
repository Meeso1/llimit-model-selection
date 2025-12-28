"""Attention-based embedding model for LLM fingerprinting."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random

from src.data_models.data_models import TrainingData
from src.data_models.attention_embedding_types import (
    PreprocessedAttentionEmbeddingData,
    ProcessedPair,
    ModelSetSample,
)
from src.data_models.triplet_encoder_types import PromptResponsePair
from src.preprocessing.attention_embedding_preprocessor import AttentionEmbeddingPreprocessor
from src.models.embedding_model_base import EmbeddingModelBase
from src.models.optimizers.optimizer_spec import OptimizerSpecification
from src.models.optimizers.adamw_spec import AdamWSpec
from src.utils.data_split import ValidationSplit, split_attention_embedding_preprocessed_data
from src.utils.timer import Timer
from src.utils.jar import Jar
from src.constants import MODELS_JAR_PATH


class PairEncoder(nn.Module):
    """Encodes individual (prompt, response) pairs."""
    
    def __init__(
        self,
        d_emb: int,
        d_scalar: int,
        h_emb: int = 256,
        h_scalar: int = 64,
        h_pair: int = 256,
        pair_mlp_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the pair encoder.
        
        Args:
            d_emb: Text embedding dimension
            d_scalar: Scalar feature dimension
            h_emb: Projected embedding dimension
            h_scalar: Projected scalar dimension
            h_pair: Pair encoding output dimension
            pair_mlp_layers: Number of MLP layers in fusion
            dropout: Dropout rate
        """
        super().__init__()
        
        # Embedding projections
        self.prompt_proj = nn.Sequential(
            nn.Linear(d_emb, h_emb),
            nn.LayerNorm(h_emb),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.response_proj = nn.Sequential(
            nn.Linear(d_emb, h_emb),
            nn.LayerNorm(h_emb),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Scalar projection
        self.scalar_proj = nn.Sequential(
            nn.Linear(d_scalar, h_scalar),
            nn.LayerNorm(h_scalar),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion MLP
        fusion_input_dim = 4 * h_emb + h_scalar
        self.fusion = self._build_mlp(
            fusion_input_dim, 
            h_pair, 
            pair_mlp_layers,
            dropout
        )
    
    def _build_mlp(
        self, 
        in_dim: int, 
        out_dim: int, 
        num_layers: int, 
        dropout: float
    ) -> nn.Module:
        """Build an MLP with the specified dimensions."""
        layers = []
        hidden_dim = (in_dim + out_dim) // 2
        
        for i in range(num_layers):
            inp = in_dim if i == 0 else hidden_dim
            outp = out_dim if i == num_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(inp, outp),
                nn.LayerNorm(outp),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        return nn.Sequential(*layers)
    
    def forward(
        self, 
        prompt_emb: torch.Tensor,  # [batch, d_emb]
        response_emb: torch.Tensor,  # [batch, d_emb]
        scalar_features: torch.Tensor  # [batch, d_scalar]
    ) -> torch.Tensor:  # [batch, h_pair]
        """
        Encode (prompt, response) pairs.
        
        Args:
            prompt_emb: Prompt embeddings
            response_emb: Response embeddings
            scalar_features: Scalar features
            
        Returns:
            Pair encodings
        """
        # Project embeddings
        prompt_h = self.prompt_proj(prompt_emb)  # [batch, h_emb]
        response_h = self.response_proj(response_emb)  # [batch, h_emb]
        
        # Compute interactions
        emb_diff = response_h - prompt_h  # [batch, h_emb]
        emb_prod = response_h * prompt_h  # [batch, h_emb]
        
        # Project scalars
        scalar_h = self.scalar_proj(scalar_features)  # [batch, h_scalar]
        
        # Fuse all features
        concat = torch.cat([
            prompt_h, response_h, emb_diff, emb_prod, scalar_h
        ], dim=-1)  # [batch, 4*h_emb + h_scalar]
        
        pair_encoding = self.fusion(concat)  # [batch, h_pair]
        return pair_encoding


class SetAggregator(nn.Module):
    """Aggregates pair encodings into a single model embedding using attention."""
    
    def __init__(
        self,
        h_pair: int = 256,
        d_out: int = 128,
        num_attention_heads: int = 4,
    ) -> None:
        """
        Initialize the set aggregator.
        
        Args:
            h_pair: Pair encoding dimension
            d_out: Output model embedding dimension
            num_attention_heads: Number of attention heads
        """
        super().__init__()
        
        self.h_pair = h_pair
        self.num_heads = num_attention_heads
        self.head_dim = h_pair // num_attention_heads
        
        assert h_pair % num_attention_heads == 0, \
            f"h_pair ({h_pair}) must be divisible by num_attention_heads ({num_attention_heads})"
        
        # Learnable query vectors: one per head, each of dimension head_dim
        # Shape: [num_heads, head_dim]
        self.queries = nn.Parameter(
            torch.randn(num_attention_heads, self.head_dim)
        )
        
        # Multi-head attention projection
        self.k_proj = nn.Linear(h_pair, h_pair)
        self.v_proj = nn.Linear(h_pair, h_pair)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(h_pair, d_out),
            nn.LayerNorm(d_out)
        )
    
    def forward(
        self, 
        pair_encodings: torch.Tensor,  # [batch, num_pairs, h_pair]
        mask: torch.Tensor | None = None  # [batch, num_pairs]
    ) -> torch.Tensor:  # [batch, d_out]
        """
        Aggregate pair encodings into model embeddings.
        
        Args:
            pair_encodings: Pair encodings  # [batch, num_pairs, h_pair]
            mask: Mask for valid pairs (True for valid, False for padding)  # [batch, num_pairs]
            
        Returns:
            Model embeddings  # [batch, d_out]
        """
        batch_size, num_pairs, _ = pair_encodings.shape
        
        # Project keys and values
        K = self.k_proj(pair_encodings)  # [batch, num_pairs, h_pair]
        V = self.v_proj(pair_encodings)  # [batch, num_pairs, h_pair]
        
        # Reshape for multi-head attention
        # [batch, num_pairs, h_pair] -> [batch, num_pairs, num_heads, head_dim]
        K = K.view(batch_size, num_pairs, self.num_heads, self.head_dim)
        K = K.permute(0, 2, 1, 3)  # [batch, num_heads, num_pairs, head_dim]
        
        V = V.view(batch_size, num_pairs, self.num_heads, self.head_dim)
        V = V.permute(0, 2, 1, 3)  # [batch, num_heads, num_pairs, head_dim]
        
        # Prepare queries: [num_heads, head_dim] -> [batch, num_heads, 1, head_dim]
        Q = self.queries.unsqueeze(0).unsqueeze(2)  # [1, num_heads, 1, head_dim]
        Q = Q.expand(batch_size, -1, -1, -1)  # [batch, num_heads, 1, head_dim]
        
        # Compute attention scores
        # Q: [batch, num_heads, 1, head_dim]
        # K^T: [batch, num_heads, head_dim, num_pairs]
        # scores: [batch, num_heads, 1, num_pairs]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, num_pairs]
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax over pairs
        attn_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, 1, num_pairs]
        
        # Weighted sum of values
        # attn_weights: [batch, num_heads, 1, num_pairs]
        # V: [batch, num_heads, num_pairs, head_dim]
        # pooled: [batch, num_heads, 1, head_dim]
        pooled = torch.matmul(attn_weights, V)
        pooled = pooled.squeeze(2)  # [batch, num_heads, head_dim]
        pooled = pooled.reshape(batch_size, self.h_pair)  # [batch, h_pair]
        
        # Output projection
        model_embedding = self.out_proj(pooled)  # [batch, d_out]
        
        return model_embedding


class AttentionEmbeddingModel(EmbeddingModelBase):
    """
    Attention-based embedding model for LLM fingerprinting.
    
    This model:
    - Encodes each (prompt, response) pair using a PairEncoder
    - Aggregates pairs from the same model using attention-based SetAggregator
    - Trains with supervised contrastive loss
    
    This is an embedding model (component) used within DnEmbeddingModel,
    not a standalone routing model.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        h_emb: int = 256,
        h_scalar: int = 64,
        h_pair: int = 256,
        d_out: int = 128,
        pair_mlp_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 0.07,
        optimizer_spec: OptimizerSpecification | None = None,
        min_model_comparisons: int = 20,
        preprocessor_seed: int = 42,
        pairs_per_model: int = 32,
        models_per_batch: int = 16,
        embeddings_per_model: int = 4,
        print_every: int | None = None,
    ) -> None:
        """
        Initialize the attention embedding model.
        
        Args:
            embedding_model_name: Name of the sentence transformer model
            h_emb: Projected embedding dimension
            h_scalar: Projected scalar dimension
            h_pair: Pair encoding output dimension
            d_out: Final model embedding dimension
            pair_mlp_layers: Number of MLP layers in pair encoder fusion
            num_attention_heads: Number of attention heads
            dropout: Dropout rate
            temperature: Temperature for contrastive loss
            optimizer_spec: Optimizer specification
            min_model_comparisons: Minimum comparisons for a model
            preprocessor_seed: Random seed for preprocessor
            pairs_per_model: Number of pairs to sample per embedding
            models_per_batch: Number of unique models per batch
            embeddings_per_model: Number of embeddings to create per model per batch (for contrastive learning)
            print_every: Print progress every N epochs (None = no printing)
        """
        self.embedding_model_name = embedding_model_name
        self.h_emb = h_emb
        self.h_scalar = h_scalar
        self.h_pair = h_pair
        self.d_out = d_out
        self.pair_mlp_layers = pair_mlp_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.temperature = temperature
        self.min_model_comparisons = min_model_comparisons
        self.preprocessor_seed = preprocessor_seed
        self.pairs_per_model = pairs_per_model
        self.models_per_batch = models_per_batch
        self.embeddings_per_model = embeddings_per_model
        self.print_every = print_every
        
        self.optimizer_spec = optimizer_spec if optimizer_spec is not None else AdamWSpec(learning_rate=1e-4)
        
        self.preprocessor = AttentionEmbeddingPreprocessor(
            embedding_model_name=embedding_model_name,
            min_model_comparisons=min_model_comparisons,
            seed=preprocessor_seed,
        )
        
        self._pair_encoder: PairEncoder | None = None
        self._set_aggregator: SetAggregator | None = None
        self._epoch_logs: list["AttentionEmbeddingModel.EpochLog"] = []
        self._model_embeddings: dict[str, np.ndarray] | None = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_timer: Timer | None = None
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of the output embeddings."""
        return self.d_out
    
    @property
    def model_embeddings(self) -> dict[str, np.ndarray]:
        """Get the model embeddings (must be initialized first)."""
        assert self._model_embeddings is not None, "Model embeddings not initialized. Train or load a model first."
        return self._model_embeddings
    
    @property
    def is_initialized(self) -> bool:
        """Check if the model has been initialized (trained or loaded)."""
        return self._pair_encoder is not None and self._set_aggregator is not None
    
    def train(
        self,
        data: TrainingData,
        validation_split: ValidationSplit | None = None,
        epochs: int = 50,
        batch_size: int = 32
    ) -> None:
        """
        Train the model.
        
        Args:
            data: Training data
            validation_split: Validation split configuration (unused for now)
            epochs: Number of epochs
            batch_size: Batch size (unused - we use models_per_batch instead)
        """
        with Timer("train_attention_embedding", verbosity="start+end") as train_timer:
            self.last_timer = train_timer
            
            # Preprocess data
            with Timer("preprocess", verbosity="start+end", parent=train_timer):
                preprocessed_data = self.preprocessor.preprocess(data)
            
            validation_split = validation_split or ValidationSplit(val_fraction=0, seed=42)
            with Timer("split_data", verbosity="start+end", parent=train_timer):
                train_preprocessed, val_preprocessed = split_attention_embedding_preprocessed_data(
                    preprocessed_data,
                    val_fraction=validation_split.val_fraction,
                    seed=validation_split.seed,
                )
                val_dataloader = self._create_dataloader(val_preprocessed, shuffle=False) if val_preprocessed is not None else None
            
            # Initialize modules if needed
            if self._pair_encoder is None:
                d_emb = train_preprocessed.samples[0].pairs[0].prompt_emb.shape[0]
                d_scalar = train_preprocessed.samples[0].pairs[0].scalar_features.shape[0]
                
                self._pair_encoder = PairEncoder(
                    d_emb=d_emb,
                    d_scalar=d_scalar,
                    h_emb=self.h_emb,
                    h_scalar=self.h_scalar,
                    h_pair=self.h_pair,
                    pair_mlp_layers=self.pair_mlp_layers,
                    dropout=self.dropout,
                ).to(self.device)
                
                self._set_aggregator = SetAggregator(
                    h_pair=self.h_pair,
                    d_out=self.d_out,
                    num_attention_heads=self.num_attention_heads,
                ).to(self.device)
            
            # Create optimizer
            optimizer = self.optimizer_spec.create_optimizer_for_multiple([self._pair_encoder, self._set_aggregator])
            scheduler = self.optimizer_spec.create_scheduler(optimizer)
            
            # Create dataloader
            dataloader = self._create_dataloader(train_preprocessed, shuffle=True)
            
            # Training loop
            with Timer("epochs", verbosity="start+end", parent=train_timer) as epochs_timer:
                for epoch in range(1, epochs + 1):
                    with Timer(f"epoch_{epoch}", verbosity="start+end", parent=epochs_timer) as epoch_timer:
                        epoch_loss, nn_accuracy, triplet_accuracy = self._train_epoch(dataloader, optimizer)
                    
                    val_loss, val_nn_accuracy, val_triplet_accuracy = self._perform_validation(val_dataloader)
                        
                    epoch_log = self.EpochLog(
                        epoch=epoch,
                        train_loss=epoch_loss,
                        duration=epoch_timer.elapsed_time,
                        nearest_neighbor_accuracy=nn_accuracy,
                        triplet_accuracy=triplet_accuracy,
                        val_loss=val_loss,
                        val_nearest_neighbor_accuracy=val_nn_accuracy,
                        val_triplet_accuracy=val_triplet_accuracy,
                    )
                    self._epoch_logs.append(epoch_log)
                        
                    self._log_epoch_result(epoch_log)
                        
                    if scheduler is not None:
                        scheduler.step()
            
            # Compute final embeddings for all models
            with Timer("compute_model_embeddings", verbosity="start+end", parent=train_timer):
                self._model_embeddings = self._compute_model_embeddings(train_preprocessed)
    
    def _create_dataloader(
        self,
        preprocessed_data: PreprocessedAttentionEmbeddingData,
        shuffle: bool = True
    ) -> DataLoader:
        """Create a dataloader for the preprocessed data."""
        dataset = self._ModelSetDataset(
            samples=preprocessed_data.samples,
            model_id_to_index=preprocessed_data.model_id_to_index,
            pairs_per_model=self.pairs_per_model,
            models_per_batch=self.models_per_batch,
            embeddings_per_model=self.embeddings_per_model,
            seed=self.preprocessor_seed,
        )
        
        return DataLoader(
            dataset,
            batch_size=1,  # Each "batch" is already a full batch of models
            shuffle=shuffle,
            collate_fn=lambda x: x[0],  # Just unwrap the single item
        )
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer
    ) -> tuple[float, float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, nearest_neighbor_accuracy, triplet_accuracy)
        """
        assert self._pair_encoder is not None
        assert self._set_aggregator is not None
        
        self._pair_encoder.train()
        self._set_aggregator.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # Accumulate embeddings and indices for accuracy computation
        all_embeddings = []
        all_model_indices = []
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Unpack batch
            prompt_embs = batch['prompt_embs'].to(self.device)  # [total_pairs, d_emb]
            response_embs = batch['response_embs'].to(self.device)  # [total_pairs, d_emb]
            scalar_features = batch['scalar_features'].to(self.device)  # [total_pairs, d_scalar]
            model_indices = batch['model_indices'].to(self.device)  # [num_models]
            pairs_per_model = batch['pairs_per_model']
            
            # Encode pairs
            pair_encodings = self._pair_encoder(
                prompt_embs, response_embs, scalar_features
            )  # [total_pairs, h_pair]
            
            # Reshape into [num_models, pairs_per_model, h_pair]
            pair_encodings = pair_encodings.view(
                len(model_indices), pairs_per_model, -1
            )
            
            # Aggregate per model
            model_embeddings: torch.Tensor = self._set_aggregator(pair_encodings)  # [num_models, d_out]
            
            # Compute contrastive loss
            loss = self._supervised_contrastive_loss(model_embeddings, model_indices)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self._pair_encoder.parameters()) + list(self._set_aggregator.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Accumulate embeddings for accuracy computation
            all_embeddings.append(model_embeddings.detach())
            all_model_indices.append(model_indices)
        
        # Compute accuracy on accumulated embeddings
        nn_accuracy, triplet_accuracy = self._compute_accuracy(
            all_embeddings, all_model_indices
        )
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, nn_accuracy, triplet_accuracy
    
    def _perform_validation(
        self,
        val_dataloader: DataLoader | None
    ) -> tuple[float | None, float | None, float | None]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (average_loss, nearest_neighbor_accuracy, triplet_accuracy)
        """
        if val_dataloader is None:
            return None, None, None
        
        assert self._pair_encoder is not None
        assert self._set_aggregator is not None
        
        self._pair_encoder.eval()
        self._set_aggregator.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        # Accumulate embeddings and indices for accuracy computation
        all_embeddings = []
        all_model_indices = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Unpack batch
                prompt_embs = batch['prompt_embs'].to(self.device)  # [total_pairs, d_emb]
                response_embs = batch['response_embs'].to(self.device)  # [total_pairs, d_emb]
                scalar_features = batch['scalar_features'].to(self.device)  # [total_pairs, d_scalar]
                model_indices = batch['model_indices'].to(self.device)  # [num_models]
                pairs_per_model = batch['pairs_per_model']
                
                # Encode pairs
                pair_encodings = self._pair_encoder(
                    prompt_embs, response_embs, scalar_features
                )  # [total_pairs, h_pair]
                
                # Reshape into [num_models, pairs_per_model, h_pair]
                pair_encodings = pair_encodings.view(
                    len(model_indices), pairs_per_model, -1
                )
                
                # Aggregate per model
                model_embeddings: torch.Tensor = self._set_aggregator(pair_encodings)  # [num_models, d_out]
                
                # Compute contrastive loss
                loss = self._supervised_contrastive_loss(model_embeddings, model_indices)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Accumulate embeddings for accuracy computation
                all_embeddings.append(model_embeddings)
                all_model_indices.append(model_indices)
        
        # Compute accuracy on accumulated embeddings
        nn_accuracy, triplet_accuracy = self._compute_accuracy(
            all_embeddings, all_model_indices
        )
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, nn_accuracy, triplet_accuracy
    
    def _supervised_contrastive_loss(
        self,
        embeddings: torch.Tensor,  # [batch_size, d_out]
        model_indices: torch.Tensor  # [batch_size]
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            embeddings: Model embeddings  # [batch_size, d_out]
            model_indices: Model indices  # [batch_size]
            
        Returns:
            Scalar loss value
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)  # [batch_size, d_out]
        
        # Compute all pairwise similarities
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature  # [batch_size, batch_size]
        
        # Create mask for positive pairs (same model, excluding self)
        model_indices = model_indices.view(-1, 1)  # [batch_size, 1]
        positive_mask = (model_indices == model_indices.T).float()  # [batch_size, batch_size]
        
        # Exclude diagonal (self-pairs) using non-inplace operation
        eye_mask = torch.eye(batch_size, device=device)  # [batch_size, batch_size]
        positive_mask = positive_mask * (1 - eye_mask)  # [batch_size, batch_size]
        
        # For numerical stability, subtract max from similarities
        logits_max, _ = similarity_matrix.max(dim=1, keepdim=True)  # [batch_size, 1]
        logits = similarity_matrix - logits_max.detach()  # [batch_size, batch_size]
        
        # Compute log-softmax over all pairs (excluding self)
        exp_logits = torch.exp(logits)  # [batch_size, batch_size]
        # Exclude self-similarity using non-inplace operation
        exp_logits = exp_logits * (1 - eye_mask)  # [batch_size, batch_size]
        
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)  # [batch_size, batch_size]
        
        # Average log-probability over positive pairs
        num_positives = positive_mask.sum(dim=1)  # [batch_size]
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)  # [batch_size]
        
        # Only compute loss for samples that have at least one positive
        valid_mask = num_positives > 0  # [batch_size]
        
        if valid_mask.sum() == 0:
            # No valid samples with positives, return zero loss
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        loss = -mean_log_prob_pos[valid_mask].mean()
        
        return loss
    
    def _compute_model_embeddings(
        self,
        preprocessed_data: PreprocessedAttentionEmbeddingData
    ) -> dict[str, np.ndarray]:
        """Compute embeddings for all models in the dataset."""
        assert self._pair_encoder is not None
        assert self._set_aggregator is not None
        
        self._pair_encoder.eval()
        self._set_aggregator.eval()
        
        embeddings = {}
        
        with torch.no_grad():
            for sample in preprocessed_data.samples:
                # Collect all pairs for this model
                prompt_embs = torch.tensor(
                    np.stack([p.prompt_emb for p in sample.pairs])
                ).to(self.device)
                response_embs = torch.tensor(
                    np.stack([p.response_emb for p in sample.pairs])
                ).to(self.device)
                scalar_features = torch.tensor(
                    np.stack([p.scalar_features for p in sample.pairs])
                ).to(self.device)
                
                # Encode pairs
                pair_encodings = self._pair_encoder(
                    prompt_embs, response_embs, scalar_features
                )  # [n_pairs, h_pair]
                
                # Add batch dimension and aggregate
                pair_encodings = pair_encodings.unsqueeze(0)  # [1, n_pairs, h_pair]
                model_embedding = self._set_aggregator(pair_encodings)  # [1, d_out]
                
                # Store embedding
                embeddings[sample.model_id] = model_embedding.squeeze(0).cpu().numpy()
        
        return embeddings
    
    def encode(
        self,
        pairs: list[PromptResponsePair],
        scaler_state: Any  # ScalerState from preprocessed data
    ) -> np.ndarray:
        """
        Encode a list of (prompt, response) pairs into a single model embedding.
        
        Args:
            pairs: List of prompt-response pairs  # [n_samples]
            scaler_state: ScalerState for normalizing features
            
        Returns:
            Single model embedding  # [d_out]
        """
        assert self._pair_encoder is not None
        assert self._set_aggregator is not None
        
        # Process pairs
        processed_pairs = []
        for pair in pairs:
            prompt_emb, response_emb, scalar_features = self.preprocessor.process_single_pair(
                pair.prompt, pair.response, scaler_state
            )
            processed_pairs.append((prompt_emb, response_emb, scalar_features))
        
        # Stack into tensors
        prompt_embs = torch.tensor(np.stack([p[0] for p in processed_pairs])).to(self.device)
        response_embs = torch.tensor(np.stack([p[1] for p in processed_pairs])).to(self.device)
        scalar_features = torch.tensor(np.stack([p[2] for p in processed_pairs])).to(self.device)
        
        self._pair_encoder.eval()
        self._set_aggregator.eval()
        
        with torch.no_grad():
            # Encode pairs
            pair_encodings = self._pair_encoder(
                prompt_embs, response_embs, scalar_features
            )  # [n_pairs, h_pair]
            
            # Add batch dimension and aggregate
            pair_encodings = pair_encodings.unsqueeze(0)  # [1, n_pairs, h_pair]
            model_embedding = self._set_aggregator(pair_encodings)  # [1, d_out]
        
        return model_embedding.squeeze(0).cpu().numpy()
    
    def _compute_accuracy(
        self,
        embeddings: list[torch.Tensor],  # [batch_size, d_out]
        model_indices: list[torch.Tensor],  # [batch_size]
        triplet_sample_ratio: float = 0.1
    ) -> tuple[float, float]:
        """
        Compute accuracy metrics for tracking training progress.
        
        Args:
            embeddings: Model embeddings  # [batch_size, d_out]
            model_indices: Model indices  # [batch_size]
            triplet_sample_ratio: Ratio of triplets to sample (relative to batch size)
            
        Returns:
            Tuple of (nearest_neighbor_accuracy, triplet_accuracy)
        """
        if len(embeddings) == 0 or len(model_indices) == 0:
            raise ValueError("No embeddings to compute accuracy on")
        
        embeddings_tensor = torch.cat(embeddings, dim=0)
        model_indices_tensor = torch.cat(model_indices, dim=0)
        
        batch_size = embeddings_tensor.shape[0]
        device = embeddings_tensor.device
        
        # L2 normalize embeddings
        embeddings_norm = F.normalize(embeddings_tensor, p=2, dim=1)  # [batch_size, d_out]
        
        # Compute pairwise distances
        distance_matrix = torch.cdist(embeddings_norm, embeddings_norm, p=2)  # [batch_size, batch_size]
        
        # 1. Nearest Neighbor Accuracy
        # For each embedding, find the nearest neighbor (excluding itself)
        # Set diagonal to infinity to exclude self-pairs
        distance_matrix_no_diag = distance_matrix + torch.eye(batch_size, device=device) * float('inf')
        
        # Find nearest neighbor for each embedding
        nearest_neighbor_indices = distance_matrix_no_diag.argmin(dim=1)  # [batch_size]
        
        # Check if nearest neighbor has the same model index
        nearest_neighbor_correct = (model_indices_tensor[nearest_neighbor_indices] == model_indices_tensor).float()
        nearest_neighbor_accuracy = nearest_neighbor_correct.mean().item()
        
        # 2. Triplet Accuracy
        # Sample triplets: for each anchor, sample a positive (same model) and negative (different model)
        num_triplets = max(1, int(batch_size * triplet_sample_ratio))
        
        triplet_correct_count = 0
        valid_triplet_count = 0
        
        for _ in range(num_triplets):
            # Sample anchor
            anchor_idx = torch.randint(0, batch_size, (1,), device=device).item()
            anchor_model = model_indices_tensor[anchor_idx]
            
            # Find all indices with same model (excluding anchor)
            positive_candidates = (model_indices_tensor == anchor_model).nonzero(as_tuple=True)[0]
            positive_candidates = positive_candidates[positive_candidates != anchor_idx]
            
            # Find all indices with different model
            negative_candidates = (model_indices_tensor != anchor_model).nonzero(as_tuple=True)[0]
            
            # Skip if we can't form a valid triplet
            if len(positive_candidates) == 0 or len(negative_candidates) == 0:
                continue
            
            # Sample positive and negative
            positive_idx = positive_candidates[torch.randint(0, len(positive_candidates), (1,), device=device)].item()
            negative_idx = negative_candidates[torch.randint(0, len(negative_candidates), (1,), device=device)].item()
            
            # Compute distances
            dist_anchor_positive = distance_matrix[anchor_idx, positive_idx]
            dist_anchor_negative = distance_matrix[anchor_idx, negative_idx]
            
            # Check if triplet is correctly ordered
            if dist_anchor_positive < dist_anchor_negative:
                triplet_correct_count += 1
            valid_triplet_count += 1
        
        triplet_accuracy = triplet_correct_count / max(valid_triplet_count, 1)
        
        return nearest_neighbor_accuracy, triplet_accuracy
    
    def _log_epoch_result(self, epoch_log: "AttentionEmbeddingModel.EpochLog") -> None:
        """Print epoch results if print_every is set."""
        if self.print_every is None:
            return
        
        if epoch_log.epoch % self.print_every != 0:
            return
        
        if epoch_log.val_loss is not None:
            loss_str = f"loss = {epoch_log.train_loss:.4f}/{epoch_log.val_loss:.4f}"
            nn_acc_str = f"nn_acc = {epoch_log.nearest_neighbor_accuracy:.4f}/{epoch_log.val_nearest_neighbor_accuracy:.4f}"
            triplet_acc_str = f"triplet_acc = {epoch_log.triplet_accuracy:.4f}/{epoch_log.val_triplet_accuracy:.4f}"
        else:
            loss_str = f"loss = {epoch_log.train_loss:.4f}"
            nn_acc_str = f"nn_acc = {epoch_log.nearest_neighbor_accuracy:.4f}"
            triplet_acc_str = f"triplet_acc = {epoch_log.triplet_accuracy:.4f}"
        
        print(
            f"Epoch {epoch_log.epoch:>4}: "
            f"{loss_str}, {nn_acc_str}, {triplet_acc_str} - "
            f"{epoch_log.duration:.2f}s"
        )
    
    def get_state_dict(self) -> dict[str, Any]:
        """Get model state dict for saving."""
        assert self._pair_encoder is not None, "Model not trained yet"
        assert self._set_aggregator is not None, "Model not trained yet"
        
        return {
            "model_type": "AttentionEmbeddingModel",
            "embedding_model_name": self.embedding_model_name,
            "h_emb": self.h_emb,
            "h_scalar": self.h_scalar,
            "h_pair": self.h_pair,
            "d_out": self.d_out,
            "pair_mlp_layers": self.pair_mlp_layers,
            "num_attention_heads": self.num_attention_heads,
            "dropout": self.dropout,
            "temperature": self.temperature,
            "min_model_comparisons": self.min_model_comparisons,
            "preprocessor_seed": self.preprocessor_seed,
            "pairs_per_model": self.pairs_per_model,
            "models_per_batch": self.models_per_batch,
            "embeddings_per_model": self.embeddings_per_model,
            "print_every": self.print_every,
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "pair_encoder_state": self._pair_encoder.state_dict(),
            "set_aggregator_state": self._set_aggregator.state_dict(),
            "epoch_logs": self._epoch_logs,
            "model_embeddings": self._model_embeddings,
        }
    
    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "AttentionEmbeddingModel":
        """Load model from state dict."""
        optimizer_spec = OptimizerSpecification.from_serialized(
            state_dict["optimizer_type"],
            state_dict["optimizer_params"],
        )
        
        model = cls(
            embedding_model_name=state_dict["embedding_model_name"],
            h_emb=state_dict["h_emb"],
            h_scalar=state_dict["h_scalar"],
            h_pair=state_dict["h_pair"],
            d_out=state_dict["d_out"],
            pair_mlp_layers=state_dict["pair_mlp_layers"],
            num_attention_heads=state_dict["num_attention_heads"],
            dropout=state_dict["dropout"],
            temperature=state_dict["temperature"],
            optimizer_spec=optimizer_spec,
            min_model_comparisons=state_dict["min_model_comparisons"],
            preprocessor_seed=state_dict["preprocessor_seed"],
            pairs_per_model=state_dict["pairs_per_model"],
            models_per_batch=state_dict["models_per_batch"],
            embeddings_per_model=state_dict.get("embeddings_per_model", 4),
            print_every=state_dict.get("print_every"),
        )
        
        # We need to know d_emb and d_scalar to initialize modules
        # These are stored implicitly in the layer weights
        d_emb = state_dict["pair_encoder_state"]["prompt_proj.0.weight"].shape[1]
        d_scalar = state_dict["pair_encoder_state"]["scalar_proj.0.weight"].shape[1]
        
        model._pair_encoder = PairEncoder(
            d_emb=d_emb,
            d_scalar=d_scalar,
            h_emb=model.h_emb,
            h_scalar=model.h_scalar,
            h_pair=model.h_pair,
            pair_mlp_layers=model.pair_mlp_layers,
            dropout=model.dropout,
        ).to(model.device)
        model._pair_encoder.load_state_dict(state_dict["pair_encoder_state"])
        
        model._set_aggregator = SetAggregator(
            h_pair=model.h_pair,
            d_out=model.d_out,
            num_attention_heads=model.num_attention_heads,
        ).to(model.device)
        model._set_aggregator.load_state_dict(state_dict["set_aggregator_state"])
        
        # Load epoch logs and embeddings
        model._epoch_logs = state_dict["epoch_logs"]
        model._model_embeddings = state_dict["model_embeddings"]
        
        return model
    
    # load_from_state_dict is inherited from EmbeddingModelBase
    
    def save(self, name: str) -> None:
        """Save the model to disk."""
        jar = Jar(str(MODELS_JAR_PATH))
        jar.add(name, self.get_state_dict())
    
    @classmethod
    def load(cls, name: str) -> "AttentionEmbeddingModel":
        """Load the model from disk."""
        jar = Jar(str(MODELS_JAR_PATH))
        return cls.load_state_dict(jar.get(name))
    
    @dataclass
    class EpochLog:
        """Log entry for a single training epoch."""
        epoch: int
        train_loss: float
        duration: float
        nearest_neighbor_accuracy: float
        triplet_accuracy: float
        val_loss: float | None = None
        val_nearest_neighbor_accuracy: float | None = None
        val_triplet_accuracy: float | None = None
    
    class _ModelSetDataset(Dataset):
        """
        Dataset that samples sets of pairs from different models.
        
        For contrastive learning, we need multiple embeddings from the same model
        in each batch. We achieve this by sampling fewer unique models but creating
        multiple embeddings per model (using different subsets of pairs).
        """
        
        def __init__(
            self,
            samples: list[ModelSetSample],
            model_id_to_index: dict[str, int],
            pairs_per_model: int,
            models_per_batch: int,
            seed: int,
            embeddings_per_model: int = 4,  # Number of embeddings to create per model
        ) -> None:
            """
            Initialize the dataset.
            
            Args:
                samples: List of model samples
                model_id_to_index: Mapping from model ID to index
                pairs_per_model: Number of pairs per embedding
                models_per_batch: Number of unique models per batch
                seed: Random seed
                embeddings_per_model: Number of embeddings to create per model (for contrastive learning)
            """
            self.samples = samples
            self.model_id_to_index = model_id_to_index
            self.pairs_per_model = pairs_per_model
            self.models_per_batch = models_per_batch
            self.embeddings_per_model = embeddings_per_model
            self.seed = seed
            
            # Group samples by model
            self.samples_by_model = {s.model_id: s for s in samples}
            self.model_ids = list(self.samples_by_model.keys())
            
            # Set random seed
            self.rng = random.Random(seed)
        
        def __len__(self) -> int:
            """Number of batches per epoch."""
            return max(1, len(self.model_ids) // self.models_per_batch) * 10  # Multiple batches per epoch
        
        def __getitem__(self, idx: int) -> dict:
            """
            Get a batch of model sets.
            
            For each model, we create multiple embeddings using different subsets of pairs.
            This ensures we have positive pairs for contrastive learning.
            """
            # Sample random models for this batch
            num_models = min(self.models_per_batch, len(self.model_ids))
            batch_models = self.rng.sample(self.model_ids, num_models)
            
            batch_data = {
                'prompt_embs': [],
                'response_embs': [],
                'scalar_features': [],
                'model_indices': [],
            }
            
            # For each model, create multiple embeddings with different pair subsets
            for model_id in batch_models:
                sample = self.samples_by_model[model_id]
                model_index = self.model_id_to_index[model_id]
                
                # Create multiple embeddings for this model
                for _ in range(self.embeddings_per_model):
                    # Sample different pairs each time
                    if len(sample.pairs) >= self.pairs_per_model:
                        selected = self.rng.sample(sample.pairs, self.pairs_per_model)
                    else:
                        selected = self.rng.choices(sample.pairs, k=self.pairs_per_model)
                    
                    for pair in selected:
                        batch_data['prompt_embs'].append(pair.prompt_emb)
                        batch_data['response_embs'].append(pair.response_emb)
                        batch_data['scalar_features'].append(pair.scalar_features)
                    
                    # Add model index for this embedding
                    batch_data['model_indices'].append(model_index)
            
            # Stack into tensors
            return {
                'prompt_embs': torch.tensor(np.stack(batch_data['prompt_embs']), dtype=torch.float32),
                'response_embs': torch.tensor(np.stack(batch_data['response_embs']), dtype=torch.float32),
                'scalar_features': torch.tensor(np.stack(batch_data['scalar_features']), dtype=torch.float32),
                'model_indices': torch.tensor(batch_data['model_indices'], dtype=torch.long),
                'pairs_per_model': self.pairs_per_model,
            }
