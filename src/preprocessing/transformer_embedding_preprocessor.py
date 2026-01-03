"""Preprocessor for transformer embedding model."""

import hashlib
import numpy as np

from src.data_models.data_models import TrainingData
from src.data_models.transformer_embedding_types import (
    PreprocessedInferenceInput,
    PreprocessedPromptPair,
    PreprocessedTrainingData,
)
from src.utils.jars import Jars
from src.utils.string_encoder import StringEncoder
from src.utils.timer import Timer
from src.preprocessing.utils import filter_out_ties, filter_out_both_bad, filter_out_empty_entries, filter_out_rare_models, create_encoder
from src.preprocessing.scoring_feature_extraction import extract_all_prompt_features


class TransformerEmbeddingPreprocessor:
    """
    Preprocessor that prepares data for transformer embedding model.
    
    This preprocessor:
    - Filters out ties and both_bad from training data
    - Extracts prompt features
    - Stores raw prompts (not embeddings - transformer will encode them)
    - Caches preprocessed data for efficiency
    """

    def __init__(
        self,
        min_model_comparisons: int = 20,
    ) -> None:
        """
        Initialize the preprocessor.
        
        Args:
            min_model_comparisons: Minimum number of comparisons for a model to be included
        """
        self.min_model_comparisons = min_model_comparisons
        self.version = "v1"
        self.last_timer: Timer | None = None

    def preprocess(
        self,
        data: TrainingData,
    ) -> PreprocessedTrainingData:
        """
        Preprocess training data by extracting features and filtering.
        
        Args:
            data: Raw training data
            
        Returns:
            Preprocessed training data with prompts, features, and model encoder
        """
        with Timer("preprocess", verbosity="start+end") as timer:
            self.last_timer = timer
            with Timer("generate_cache_key", verbosity="start+end", parent=timer):
                cache_key = self._generate_cache_key(data)
            
            if cache_key in Jars.preprocessed_data:
                return Jars.preprocessed_data.get(cache_key)
            
            with Timer("filter_entries_and_fit_encoder", verbosity="start+end", parent=timer):
                filtered_data, model_encoder = self._filter_data_and_fit_encoder(data)
            
            with Timer("extract_prompt_features", verbosity="start+end", parent=timer) as prompt_features_timer:
                prompt_features_list = [
                    extract_all_prompt_features(
                        entry.user_prompt,
                        entry.conversation_history,
                        timer=prompt_features_timer
                    )
                    for entry in filtered_data.entries
                ]
            
            pairs: list[PreprocessedPromptPair] = []
            for entry, prompt_features in zip(filtered_data.entries, prompt_features_list):
                winner_label = 0 if entry.winner == "model_a" else 1
                # Note: model_embeddings will be added later by the model
                pairs.append(
                    PreprocessedPromptPair(
                        prompt=entry.user_prompt,
                        prompt_features=prompt_features,
                        model_embedding_a=np.array([]),  # Placeholder, will be filled by model
                        model_embedding_b=np.array([]),  # Placeholder, will be filled by model
                        model_id_a=model_encoder.encode(entry.model_a),
                        model_id_b=model_encoder.encode(entry.model_b),
                        winner_label=winner_label,
                    )
                )
            
            prompt_features_dim = pairs[0].prompt_features.shape[0]
            preprocessed_data = PreprocessedTrainingData(
                pairs=pairs,
                prompt_features_dim=prompt_features_dim,
                model_encoder=model_encoder,
            )
            
            Jars.preprocessed_data.add(cache_key, preprocessed_data)
            
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
        - Min model comparisons threshold
        
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
        return f"transformer_embedding/{self.version}/{dataset_signature}-{self.min_model_comparisons}"

    def preprocess_for_inference(
        self,
        prompts: list[str],
        model_names: list[str],
        model_embeddings: dict[str, np.ndarray],
    ) -> PreprocessedInferenceInput:
        """
        Preprocess prompts and model names for inference.
        
        Args:
            prompts: List of prompts (raw text)
            model_names: List of model names to score
            model_embeddings: Dictionary mapping model names to embeddings
            
        Returns:
            PreprocessedInferenceInput with prompts, features, and model embeddings
        """
        
        prompt_features_list = [
            extract_all_prompt_features(prompt, [])  # TODO: Decide what to do with conversation history in inference
            for prompt in prompts
        ]
        
        model_embeddings_array = np.array([
            model_embeddings[model_name] if model_name in model_embeddings else model_embeddings["default"]
            for model_name in model_names
        ])  # [n_models, model_embedding_dim]
        
        return PreprocessedInferenceInput(
            prompts=prompts,
            prompt_features=np.stack(prompt_features_list),  # [n_prompts, prompt_features_dim]
            model_embeddings=model_embeddings_array,  # [n_models, model_embedding_dim]
        )

