"""Preprocessor for embedding prompts using sentence transformers."""

import hashlib
from sentence_transformers import SentenceTransformer
import torch

from src.constants import PREPROCESSED_DATA_JAR_PATH
from src.data_models.data_models import TrainingData
from src.data_models.dense_network_types import (
    PreprocessedInferenceInput,
    PreprocessedPromptPair,
    PreprocessedTrainingData,
)
from src.utils.jar import Jar
from src.utils.string_encoder import StringEncoder
from src.utils.timer import Timer
from src.preprocessing.utils import filter_out_ties, filter_out_both_bad, filter_out_empty_entries, filter_out_rare_models, create_encoder


class PromptEmbeddingPreprocessor:
    """
    Preprocessor that embeds prompts using sentence transformers.
    
    This preprocessor:
    - Filters out ties and both_bad from training data
    - Embeds user prompts using a sentence transformer model
    - Caches preprocessed data for efficiency
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        min_model_comparisons: int = 20,
    ) -> None:
        """
        Initialize the preprocessor.
        
        Args:
            embedding_model_name: Name of the sentence transformer model to use
            min_model_comparisons: Minimum number of comparisons for a model to be included
        """
        self.embedding_model_name = embedding_model_name
        self.min_model_comparisons = min_model_comparisons
        self.version = "v1"
        self.jar = Jar(str(PREPROCESSED_DATA_JAR_PATH))
        self._model: SentenceTransformer | None = None
        self.last_timer: Timer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def preprocess(
        self,
        data: TrainingData,
    ) -> PreprocessedTrainingData:
        """
        Preprocess training data by embedding prompts and filtering.
        
        Args:
            data: Raw training data
            
        Returns:
            Preprocessed training data with embeddings and model encoder
        """
        with Timer("preprocess", verbosity="start+end") as timer:
            self.last_timer = timer
            with Timer("generate_cache_key", verbosity="start+end", parent=timer):
                cache_key = self._generate_cache_key(data)
            
            if cache_key in self.jar:
                return self.jar.get(cache_key)
            
            with Timer("filter_entries_and_fit_encoder", verbosity="start+end", parent=timer):
                filtered_data, model_encoder = self._filter_data_and_fit_encoder(data)
            
            with Timer("embed_prompts", verbosity="start+end", parent=timer):
                prompts = [entry.user_prompt for entry in filtered_data.entries]
                embeddings = self._embed_prompts(prompts).cpu()  # [n_prompts, embedding_dim]
            
            pairs: list[PreprocessedPromptPair] = []
            for i, entry in enumerate(filtered_data.entries):
                winner_label = 0 if entry.winner == "model_a" else 1
                pairs.append(
                    PreprocessedPromptPair(
                        prompt_embedding=embeddings[i].numpy(),
                        model_id_a=model_encoder.encode(entry.model_a),
                        model_id_b=model_encoder.encode(entry.model_b),
                        winner_label=winner_label,
                    )
                )
            
            embedding_dim = pairs[0].prompt_embedding.shape[0]
            preprocessed_data = PreprocessedTrainingData(
                pairs=pairs,
                embedding_dim=embedding_dim,
                model_encoder=model_encoder,
            )
            
            self.jar.add(cache_key, preprocessed_data)
            
            return preprocessed_data

    def _filter_data_and_fit_encoder(self, data: TrainingData) -> tuple[TrainingData, StringEncoder]:
        filtered_data = filter_out_ties(data)
        filtered_data = filter_out_both_bad(filtered_data)
        filtered_data = filter_out_empty_entries(filtered_data)
        filtered_data = filter_out_rare_models(filtered_data, self.min_model_comparisons)
        if len(filtered_data.entries) == 0:
            raise ValueError(
                "No valid training data after filtering. "
                f"All models were filtered out (min_model_comparisons={self.min_model_comparisons}). "
                "Try lowering min_model_comparisons or providing more training data."
            )

        model_encoder = create_encoder(filtered_data)
        
        return filtered_data, model_encoder

    def _generate_cache_key(self, data: TrainingData) -> str:
        """
        Generate cache key for a dataset.
        
        The cache key is based on:
        - Preprocessor version
        - Hash of dataset's entries
        
        Args:
            data: Training dataset
            
        Returns:
            Cache key string
        """
        # Create a hash based on dataset content
        # Use session IDs and timestamps as a proxy for dataset identity
        hasher = hashlib.sha256()
        hasher.update(str(len(data.entries)).encode())
        
        for entry in data.entries:
            hasher.update(entry.timestamp.encode())
        
        dataset_signature = hasher.hexdigest()[:16]
        return f"prompt_embedding/{self.version}/{self.embedding_model_name}-{dataset_signature}-{self.min_model_comparisons}"

    def preprocess_for_inference(
        self,
        prompts: list[str],
        model_names: list[str],
        model_encoder: StringEncoder,
    ) -> PreprocessedInferenceInput:
        """
        Preprocess prompts and model names for inference.
        
        Args:
            prompts: List of prompts to embed
            model_names: List of model names to score
            model_encoder: Fitted model encoder from training
            
        Returns:
            PreprocessedInferenceInput with embeddings and model IDs
        """
        prompt_embeddings = self._embed_prompts(prompts).cpu()  # [n_prompts, embedding_dim]
        model_ids = model_encoder.encode(model_names)
        
        return PreprocessedInferenceInput(
            prompt_embeddings=prompt_embeddings.numpy(),
            model_ids=model_ids,
        )

    def _embed_prompts(self, prompts: list[str]) -> torch.Tensor:
        """
        Embed a list of prompts using the sentence transformer model.
        
        Args:
            prompts: List of prompts to embed
            
        Returns:
            Tensor of embeddings  # [n_prompts, embedding_dim]
        """
        embeddings = self.model.encode(
            prompts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return embeddings
