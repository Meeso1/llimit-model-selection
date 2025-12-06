from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

from src.data_models.data_models import TrainingData
from src.data_models.behavior_encoder_types import PreprocessedBehaviorEncoderData, PromptResponsePair
from src.preprocessing.behavior_embedding_preprocessor import BehaviorEmbeddingPreprocessor
from src.utils.data_split import ValidationSplit, split_preprocessed_behavior_data
from src.utils.timer import Timer
from src.models.optimizers.optimizer_spec import OptimizerSpecification
from src.models.optimizers.adamw_spec import AdamWSpec


class ModelBehaviorEncoder:
    """
    Model behavior encoder using Siamese-style architecture with triplet loss.
    
    This encoder learns to map (prompt, response) pairs to an embedding space where:
    - Models with similar performance are close together
    - Each model has a distinctive signature
    - Winning responses are separated from losing responses
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
        Initialize the model behavior encoder.
        
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
        self.encoder_model_name = encoder_model_name
        self.hidden_dims = hidden_dims if hidden_dims is not None else [256, 128]
        
        if not self.hidden_dims:
            raise ValueError("hidden_dims cannot be empty - the model must have trainable layers")
        
        self.optimizer_spec = optimizer_spec if optimizer_spec is not None else AdamWSpec(learning_rate=0.001)
        self.triplet_margin = triplet_margin
        self.regularization_weight = regularization_weight
        self.min_model_comparisons = min_model_comparisons
        self.identity_positive_ratio = identity_positive_ratio
        self.preprocessor_seed = preprocessor_seed
        self.print_every = print_every
        
        self._text_encoder: SentenceTransformer | None = None
        self.preprocessor = BehaviorEmbeddingPreprocessor(
            min_model_comparisons=min_model_comparisons,
            identity_positive_ratio=identity_positive_ratio,
            seed=preprocessor_seed,
        )
        
        self._module: ModelBehaviorEncoder._BehaviorEncoderModule | None = None
        self._epoch_logs: list["ModelBehaviorEncoder.EpochLog"] = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_timer: Timer | None = None
    
    @property
    def text_encoder(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._text_encoder is None:
            self._text_encoder = SentenceTransformer(self.encoder_model_name)
        return self._text_encoder
    
    @property
    def module(self) -> "_BehaviorEncoderModule":
        """Get the neural network module (must be initialized first)."""
        if self._module is None:
            raise RuntimeError("Module not initialized. Train or load a model first.")
        return self._module
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of the output embeddings."""
        return self.hidden_dims[-1]
    
    def _initialize_module(self) -> None:
        """Initialize the neural network module."""
        base_dim = self.text_encoder.get_sentence_embedding_dimension()
        self._module = self._BehaviorEncoderModule(
            input_dim=base_dim,
            hidden_dims=self.hidden_dims,
        ).to(self.device)
    
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
        with Timer("train_behavior_encoder", verbosity="start+end") as train_timer:
            self.last_timer = train_timer
            
            with Timer("preprocess", verbosity="start+end", parent=train_timer):
                preprocessed_data = self.preprocessor.preprocess(data)
            
            with Timer("split_data", verbosity="start+end", parent=train_timer):
                if validation_split is not None:
                    train_preprocessed, val_preprocessed = split_preprocessed_behavior_data(
                        preprocessed_data,
                        val_fraction=validation_split.val_fraction,
                        seed=validation_split.seed,
                    )
                else:
                    train_preprocessed = preprocessed_data
                    val_preprocessed = None
            
            with Timer("prepare_dataloaders", verbosity="start+end", parent=train_timer):
                train_dataloader = self._prepare_dataloader(train_preprocessed, batch_size, shuffle=True)
                val_dataloader = self._prepare_dataloader(val_preprocessed, batch_size, shuffle=False) if val_preprocessed is not None else None
            
            if self._module is None:
                self._initialize_module()
            
            optimizer = self.optimizer_spec.create_optimizer(self.module)
            scheduler = self.optimizer_spec.create_scheduler(optimizer)
            
            criterion = nn.TripletMarginLoss(margin=self.triplet_margin, p=2)
            
            with Timer("epochs", verbosity="start+end", parent=train_timer) as epochs_timer:
                for epoch in range(1, epochs + 1):
                    epoch_log = self._train_epoch(
                        epoch,
                        train_dataloader,
                        val_dataloader,
                        optimizer,
                        criterion,
                        epochs_timer,
                    )
                    
                    self._log_epoch_result(epoch_log)
                    
                    if scheduler is not None:
                        scheduler.step()
    
    def encode(
        self,
        pairs: list[PromptResponsePair],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode a list of (prompt, response) pairs into embeddings.
        
        Args:
            pairs: List of prompt-response pairs  # [n_samples]
            batch_size: Batch size for encoding
            
        Returns:
            Array of embeddings  # [n_samples, embedding_dim]
        """
        # Concatenate prompt and response
        texts = [f"{pair.prompt} [SEP] {pair.response}" for pair in pairs]
        
        # Encode texts using sentence transformer
        text_embeddings = self.text_encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )  # [n_samples, base_dim]
        
        # Apply trainable dense layers if module exists
        if self._module is not None:
            text_embeddings = text_embeddings.to(self.device)
            with torch.no_grad():
                embeddings = self.module(text_embeddings)  # [n_samples, output_dim]
            embeddings = embeddings.cpu()
        else:
            embeddings = text_embeddings
        
        return embeddings.numpy()  # [n_samples, embedding_dim]
    
    def compute_model_embedding(
        self,
        pairs: list[PromptResponsePair],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Compute a single, representative embedding for a model by averaging.
        
        Args:
            pairs: List of prompt-response pairs from the model  # [n_samples]
            batch_size: Batch size for encoding
            
        Returns:
            Single averaged embedding  # [embedding_dim]
        """
        embeddings = self.encode(pairs, batch_size)  # [n_samples, embedding_dim]
        return np.mean(embeddings, axis=0)  # [embedding_dim]
    
    def get_state_dict(self) -> dict[str, Any]:
        """Get state dictionary for serialization."""
        if self._module is None:
            raise RuntimeError("Module not initialized. Train the model first.")
        
        return {
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
            "module_state_dict": self.module.state_dict(),
            "epoch_logs": self._epoch_logs,
        }
    
    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "ModelBehaviorEncoder":
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
        
        model._initialize_module()
        model.module.load_state_dict(state_dict["module_state_dict"])
        model._epoch_logs = state_dict["epoch_logs"]
        
        return model
    
    def _prepare_dataloader(
        self,
        preprocessed_data: PreprocessedBehaviorEncoderData,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Prepare dataloader from preprocessed triplets.
        
        Encodes all texts using the frozen sentence transformer and caches the results.
        This is efficient since the text encoder is frozen during training.
        
        Args:
            preprocessed_data: Preprocessed triplets
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader yielding (anchor_emb, positive_emb, negative_emb) tuples
        """
        # Collect all texts to encode
        all_prompts = []
        all_responses = []
        
        for triplet in preprocessed_data.triplets:
            all_prompts.extend([triplet.anchor_prompt, triplet.positive_prompt, triplet.negative_prompt])
            all_responses.extend([triplet.anchor_response, triplet.positive_response, triplet.negative_response])
        
        # Concatenate prompts and responses
        all_texts = [f"{p} [SEP] {r}" for p, r in zip(all_prompts, all_responses)]
        
        # Encode all texts using the frozen sentence transformer
        # This is done once and cached for the entire training process
        all_embeddings = self.text_encoder.encode(
            all_texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )  # [n_texts, base_dim]
        
        # Split into anchor, positive, negative
        anchor_embeddings = all_embeddings[0::3]  # [n_triplets, base_dim]
        positive_embeddings = all_embeddings[1::3]  # [n_triplets, base_dim]
        negative_embeddings = all_embeddings[2::3]  # [n_triplets, base_dim]
        
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
    
    def _train_epoch(
        self,
        epoch: int,
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epochs_timer: Timer,
    ) -> "ModelBehaviorEncoder.EpochLog":
        """Train for one epoch."""
        with Timer(f"epoch_{epoch}", verbosity="start+end", parent=epochs_timer) as timer:
            self.module.train()
            total_triplet_loss = 0.0
            total_reg_loss = 0.0
            total_loss = 0.0
            correct_triplets = 0
            n_batches = 0
            total_samples = 0
            
            for batch_anchor, batch_positive, batch_negative in dataloader:
                batch_anchor = batch_anchor.to(self.device)  # [batch_size, base_dim]
                batch_positive = batch_positive.to(self.device)  # [batch_size, base_dim]
                batch_negative = batch_negative.to(self.device)  # [batch_size, base_dim]
                
                batch_size_actual = len(batch_anchor)
                total_samples += batch_size_actual
                
                optimizer.zero_grad()
                
                # Forward pass through projection
                anchor_emb = self.module(batch_anchor)  # [batch_size, output_dim]
                positive_emb = self.module(batch_positive)  # [batch_size, output_dim]
                negative_emb = self.module(batch_negative)  # [batch_size, output_dim]
                
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
                
                n_batches += 1
            
            avg_triplet_loss = total_triplet_loss / n_batches
            avg_reg_loss = total_reg_loss / n_batches
            avg_loss = total_loss / n_batches
            triplet_accuracy = correct_triplets / total_samples
            
            # Validation
            val_loss, val_triplet_accuracy = self._perform_validation(
                val_dataloader, criterion, timer
            ) if val_dataloader is not None else (None, None)
            
            epoch_log = self.EpochLog(
                epoch=epoch,
                train_loss=avg_loss,
                train_triplet_loss=avg_triplet_loss,
                train_reg_loss=avg_reg_loss,
                train_triplet_accuracy=triplet_accuracy,
                val_loss=val_loss,
                val_triplet_accuracy=val_triplet_accuracy,
                duration=timer.elapsed_time,
            )
            self._epoch_logs.append(epoch_log)
            
            return epoch_log
    
    def _perform_validation(
        self,
        val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
        timer: Timer,
    ) -> tuple[float, float]:
        """Perform validation and return (loss, triplet_accuracy)."""
        self.module.eval()
        total_loss = 0.0
        correct_triplets = 0
        n_batches = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_anchor, batch_positive, batch_negative in val_dataloader:
                batch_anchor = batch_anchor.to(self.device)  # [batch_size, base_dim]
                batch_positive = batch_positive.to(self.device)  # [batch_size, base_dim]
                batch_negative = batch_negative.to(self.device)  # [batch_size, base_dim]
                
                batch_size_actual = len(batch_anchor)
                total_samples += batch_size_actual
                
                # Forward pass
                anchor_emb = self.module(batch_anchor)  # [batch_size, output_dim]
                positive_emb = self.module(batch_positive)  # [batch_size, output_dim]
                negative_emb = self.module(batch_negative)  # [batch_size, output_dim]
                
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
                
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        triplet_accuracy = correct_triplets / total_samples
        return avg_loss, triplet_accuracy
    
    def _compute_regularization_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence regularization loss.
        
        Encourages embeddings to follow a standard normal distribution.
        
        Args:
            embeddings: Batch of embeddings  # [batch_size, embedding_dim]
            
        Returns:
            Scalar regularization loss
        """
        # Compute mean and log variance across the batch
        mean = embeddings.mean(dim=0)  # [embedding_dim]
        log_var = torch.log(embeddings.var(dim=0) + 1e-8)  # [embedding_dim]
        
        # KL divergence from N(0, 1)
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return kl_loss / embeddings.size(0)  # Normalize by batch size
    
    def _log_epoch_result(self, epoch_log: "ModelBehaviorEncoder.EpochLog") -> None:
        """Print epoch results if print_every is set."""
        if self.print_every is None:
            return
        
        if epoch_log.epoch % self.print_every != 0:
            return
        
        if epoch_log.val_loss is None or epoch_log.val_triplet_accuracy is None:
            print(
                f"Epoch {epoch_log.epoch:>4}: "
                f"loss = {epoch_log.train_loss:.4f} "
                f"(triplet: {epoch_log.train_triplet_loss:.4f}, reg: {epoch_log.train_reg_loss:.4f}), "
                f"triplet_acc = {(epoch_log.train_triplet_accuracy*100):.2f}% - "
                f"{epoch_log.duration:.2f}s"
            )
        else:
            print(
                f"Epoch {epoch_log.epoch:>4}: "
                f"loss = {epoch_log.train_loss:.4f}/{epoch_log.val_loss:.4f}, "
                f"triplet_acc = {(epoch_log.train_triplet_accuracy*100):.2f}%/{(epoch_log.val_triplet_accuracy*100):.2f}% - "
                f"{epoch_log.duration:.2f}s"
            )
    
    @dataclass
    class EpochLog:
        """Log entry for a single training epoch."""
        epoch: int
        train_loss: float
        train_triplet_loss: float
        train_reg_loss: float
        train_triplet_accuracy: float
        val_loss: float | None
        val_triplet_accuracy: float | None
        duration: float
    
    class _BehaviorEncoderModule(nn.Module):
        """
        Inner PyTorch module for the behavior encoder.
        
        This is a trainable dense network that projects the frozen text encoder
        embeddings to a learned embedding space.
        """
        
        def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
            """
            Initialize the module.
            
            Args:
                input_dim: Input embedding dimension from text encoder
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
