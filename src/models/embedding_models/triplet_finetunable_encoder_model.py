"""Triplet-based model behavior encoder using fine-tunable transformers."""

from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast

from src.data_models.data_models import TrainingData
from src.data_models.triplet_encoder_types import (
    PreprocessedTripletEncoderData,
    PromptResponsePair,
    TrainingTriplet,
)
from src.preprocessing.triplet_finetunable_encoder_preprocessor import TripletFinetunableEncoderPreprocessor
from src.utils.accuracy import compute_embedding_accuracy
from src.utils.data_split import split_preprocessed_behavior_data
from src.utils.timer import Timer
from src.utils.torch_utils import state_dict_to_cpu
from src.utils.transformer_pooling_utils import detect_pooling_method, pool_embeddings, PoolingMethod
from src.models.embedding_models.triplet_model_base import TripletModelBase
from src.models.optimizers.optimizer_spec import OptimizerSpecification
from src.models.optimizers.adamw_spec import AdamWSpec
from src.models.finetuning_specs.finetuning_spec_union import FineTuningSpec
from src.models.finetuning_specs.finetuning_spec import FineTuningSpecification
from src.models.finetuning_specs.last_layers_spec import LastLayersSpec


class TripletFinetunableEncoderModel(TripletModelBase[TrainingTriplet]):
    """
    Triplet-based encoder using fine-tunable transformers.

    This encoder uses:
    - Transformer (e.g., a sentence-transformer model) for text encoding
    - A configurable fine-tuning strategy (LoRA, last-layers, full, etc.)
    - Optional trainable projection layer on top
    - Triplet margin loss for training
    """

    def __init__(
        self,
        transformer_model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        finetuning_spec: FineTuningSpec | None = None,
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
            transformer_model_name: HuggingFace model name
            finetuning_spec: Fine-tuning strategy (default: LastLayersSpec(num_unfrozen_layers=1))
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
        self.finetuning_spec = finetuning_spec if finetuning_spec is not None else LastLayersSpec(num_unfrozen_layers=1)
        self.projection_dim = projection_dim
        self.max_length = max_length
        self.optimizer_spec = optimizer_spec if optimizer_spec is not None else AdamWSpec(learning_rate=1e-5)

        self.preprocessor = TripletFinetunableEncoderPreprocessor(
            min_model_comparisons=min_model_comparisons,
            identity_positive_ratio=identity_positive_ratio,
            seed=preprocessor_seed,
        )

        self._tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            transformer_model_name,
            use_fast=("deberta" not in transformer_model_name),
        )
        self._pooling_method: PoolingMethod = detect_pooling_method(transformer_model_name)

        if print_every is not None:
            print(f"Detected pooling method for {transformer_model_name}: {self._pooling_method}")

        self._module: TripletFinetunableEncoderModel._EncoderModule | None = None

    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of the output embeddings."""
        return self.projection_dim

    @property
    def embedding_type(self) -> str:
        """Get the type of the embedding model."""
        return "finetunable"

    def _get_module(self) -> nn.Module:
        """Get the neural network module."""
        return self._module

    def _initialize_module(self, input_dim: int | None = None) -> None:
        """Initialize the neural network module."""
        self._module = self._EncoderModule(
            transformer_model_name=self.transformer_model_name,
            finetuning_spec=self.finetuning_spec,
            projection_dim=self.projection_dim,
            pooling_method=self._pooling_method,
            quiet=self.print_every is None,
        ).to(self.device)

    def _infer_input_dim(self, first_batch: Any) -> int | None:
        return None

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
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]:
        """
        Prepare dataloader from preprocessed triplets (text-based).

        Args:
            preprocessed_data: Preprocessed triplets (as TrainingTriplet objects)
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Returns:
            DataLoader yielding (anchor, positive, negative, anchor_model_ids) tuples
        """
        dataset = self._TripletTextDataset(
            triplets=preprocessed_data.triplets,
            tokenizer=self._tokenizer,
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
        batch: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], str]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], list[str]]:
        """
        Collate function for batching tokenized triplets.

        Args:
            batch: List of (anchor_tokens, positive_tokens, negative_tokens, anchor_model_id) tuples

        Returns:
            Tuple of batched (anchor_tokens, positive_tokens, negative_tokens, anchor_model_ids)
        """
        anchors, positives, negatives, anchor_model_ids = zip(*batch)

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

        return anchor_batch, positive_batch, negative_batch, list(anchor_model_ids)

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
                text = f"{pair.prompt} [SEP] {pair.response}"

                tokens = self._tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}

                embedding = self._module(tokens)  # [1, embedding_dim]
                embeddings.append(embedding.cpu().numpy())

        return np.concatenate(embeddings, axis=0)  # [n_samples, embedding_dim]

    def _train_epoch(
        self,
        epoch: int,
        dataloader: DataLoader,
        val_dataloader: DataLoader | None,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epochs_timer: Timer,
    ) -> "TripletModelBase.EpochLog":
        """Train for one epoch with tokenized text."""
        with Timer(f"epoch_{epoch}", verbosity="start+end", parent=epochs_timer) as timer:
            module = self._get_module()
            module.train()
            total_triplet_loss = 0.0
            total_reg_loss = 0.0
            total_loss = 0.0
            correct_triplets = 0
            n_batches = 0
            total_samples = 0

            train_anchor_embeddings = []
            train_anchor_model_ids = []

            for batch_anchor, batch_positive, batch_negative, batch_anchor_model_ids in dataloader:
                batch_anchor = {k: v.to(self.device) for k, v in batch_anchor.items()}  # dict with input_ids, attention_mask
                batch_positive = {k: v.to(self.device) for k, v in batch_positive.items()}
                batch_negative = {k: v.to(self.device) for k, v in batch_negative.items()}

                batch_size_actual = batch_anchor['input_ids'].size(0)
                total_samples += batch_size_actual

                optimizer.zero_grad()

                anchor_emb = module(batch_anchor)  # [batch_size, output_dim]
                positive_emb = module(batch_positive)  # [batch_size, output_dim]
                negative_emb = module(batch_negative)  # [batch_size, output_dim]

                triplet_loss = criterion(anchor_emb, positive_emb, negative_emb)

                reg_loss = self._compute_regularization_loss(
                    torch.cat([anchor_emb, positive_emb, negative_emb], dim=0)
                )

                loss = triplet_loss + self.regularization_weight * reg_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
                optimizer.step()

                total_triplet_loss += triplet_loss.item()
                total_reg_loss += reg_loss.item()
                total_loss += loss.item()

                with torch.no_grad():
                    dist_pos = torch.norm(anchor_emb - positive_emb, p=2, dim=1)  # [batch_size]
                    dist_neg = torch.norm(anchor_emb - negative_emb, p=2, dim=1)  # [batch_size]
                    correct_triplets += (dist_pos < dist_neg).sum().item()

                train_anchor_embeddings.append(anchor_emb.detach())
                train_anchor_model_ids.extend(batch_anchor_model_ids)

                n_batches += 1

            avg_triplet_loss = total_triplet_loss / n_batches
            avg_reg_loss = total_reg_loss / n_batches
            avg_loss = total_loss / n_batches
            triplet_accuracy = correct_triplets / total_samples

            train_anchor_embeddings_tensor = torch.cat(train_anchor_embeddings, dim=0)
            with Timer(f"compute_model_embeddings_{epoch}", verbosity="start+end", parent=timer):
                model_embeddings = self._compute_model_embeddings_from_stored(
                    train_anchor_embeddings_tensor, train_anchor_model_ids
                )

            train_universal_accuracy = compute_embedding_accuracy(
                sample_embeddings=train_anchor_embeddings_tensor.detach().cpu().numpy(),
                sample_model_names=train_anchor_model_ids,
                model_embeddings=model_embeddings
            )

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

        val_anchor_embeddings = []
        val_anchor_model_ids = []

        with torch.no_grad():
            for batch_anchor, batch_positive, batch_negative, batch_anchor_model_ids in val_dataloader:
                batch_anchor = {k: v.to(self.device) for k, v in batch_anchor.items()}
                batch_positive = {k: v.to(self.device) for k, v in batch_positive.items()}
                batch_negative = {k: v.to(self.device) for k, v in batch_negative.items()}

                batch_size_actual = batch_anchor['input_ids'].size(0)
                total_samples += batch_size_actual

                anchor_emb = module(batch_anchor)  # [batch_size, output_dim]
                positive_emb = module(batch_positive)  # [batch_size, output_dim]
                negative_emb = module(batch_negative)  # [batch_size, output_dim]

                triplet_loss = criterion(anchor_emb, positive_emb, negative_emb)

                reg_loss = self._compute_regularization_loss(
                    torch.cat([anchor_emb, positive_emb, negative_emb], dim=0)
                )

                loss = triplet_loss + self.regularization_weight * reg_loss
                total_loss += loss.item()

                dist_pos = torch.norm(anchor_emb - positive_emb, p=2, dim=1)  # [batch_size]
                dist_neg = torch.norm(anchor_emb - negative_emb, p=2, dim=1)  # [batch_size]
                correct_triplets += (dist_pos < dist_neg).sum().item()

                val_anchor_embeddings.append(anchor_emb)
                val_anchor_model_ids.extend(batch_anchor_model_ids)

                n_batches += 1

        avg_loss = total_loss / n_batches
        triplet_accuracy = correct_triplets / total_samples

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
            "model_type": "TripletFinetunableEncoderModel",
            "transformer_model_name": self.transformer_model_name,
            "finetuning_method": self.finetuning_spec.method,
            "finetuning_spec": self.finetuning_spec.model_dump(),
            "pooling_method": self._pooling_method,
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
            "module_state_dict": state_dict_to_cpu(self._module.state_dict()),
            "model_embeddings": self.model_embeddings,
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "TripletFinetunableEncoderModel":
        """Load model from state dictionary."""
        optimizer_spec = OptimizerSpecification.from_serialized(
            state_dict["optimizer_type"],
            state_dict["optimizer_params"],
        )

        finetuning_spec = FineTuningSpecification.from_serialized(
            state_dict["finetuning_method"],
            state_dict["finetuning_spec"],
        )

        model = cls(
            transformer_model_name=state_dict["transformer_model_name"],
            finetuning_spec=finetuning_spec,
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

        # Restore the serialized pooling method so the module is built with the
        # same pooling that was used during training (the __init__ auto-detects it,
        # but we override here in case the saved value differs from detection).
        model._pooling_method = state_dict["pooling_method"]
        model._module = TripletFinetunableEncoderModel._EncoderModule(
            transformer_model_name=state_dict["transformer_model_name"],
            finetuning_spec=finetuning_spec,
            projection_dim=state_dict["projection_dim"],
            pooling_method=state_dict["pooling_method"],
            quiet=True,
        )
        model._module.load_state_dict(state_dict["module_state_dict"])
        model._module.to(model.device)
        model._model_embeddings = state_dict["model_embeddings"]

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

        def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], str]:
            triplet = self.triplets[idx]

            anchor_text = f"{triplet.anchor_prompt} [SEP] {triplet.anchor_response}"
            positive_text = f"{triplet.positive_prompt} [SEP] {triplet.positive_response}"
            negative_text = f"{triplet.negative_prompt} [SEP] {triplet.negative_response}"

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

            anchor_tokens = {k: v.squeeze(0) for k, v in anchor_tokens.items()}
            positive_tokens = {k: v.squeeze(0) for k, v in positive_tokens.items()}
            negative_tokens = {k: v.squeeze(0) for k, v in negative_tokens.items()}

            return anchor_tokens, positive_tokens, negative_tokens, triplet.anchor_model_id

    class _EncoderModule(nn.Module):
        """
        Inner PyTorch module for the fine-tunable encoder model.

        The fine-tuning strategy (frozen layers, LoRA, etc.) is fully delegated
        to the provided FineTuningSpec, which mirrors the approach used by
        TransformerEmbeddingModel._TransformerNetwork.
        """

        def __init__(
            self,
            transformer_model_name: str,
            finetuning_spec: FineTuningSpec,
            projection_dim: int,
            pooling_method: PoolingMethod,
            quiet: bool = False,
        ) -> None:
            """
            Initialize the module.

            Args:
                transformer_model_name: HuggingFace model name
                finetuning_spec: Fine-tuning strategy to apply to the transformer
                projection_dim: Dimension of the projection layer
                pooling_method: Token pooling strategy ('mean', 'cls', 'last_token')
                quiet: Whether to suppress fine-tuning summary prints
            """
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

            self.projection = nn.Sequential(
                nn.Linear(transformer_hidden_size, projection_dim),
                nn.Tanh(),
            )

        def forward(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
            """
            Forward pass through the transformer and projection.

            Args:
                tokens: Dictionary with 'input_ids' and 'attention_mask'
                    input_ids: [batch_size, seq_length]
                    attention_mask: [batch_size, seq_length]

            Returns:
                Output embeddings  # [batch_size, projection_dim]
            """
            outputs: BaseModelOutputWithPast = self.transformer(**tokens)

            pooled = pool_embeddings(
                outputs.last_hidden_state,  # [batch_size, seq_len, transformer_hidden_size]
                tokens["attention_mask"],   # [batch_size, seq_len]
                self.pooling_method,
            )  # [batch_size, transformer_hidden_size]

            return self.projection(pooled)  # [batch_size, projection_dim]
