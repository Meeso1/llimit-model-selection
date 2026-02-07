"""Preprocessor for length prediction models."""

import hashlib
import numpy as np

from src.data_models.data_models import TrainingData, EvaluationEntry
from src.data_models.length_prediction.length_prediction_data_models import (
    PreprocessedLengthPredictionSample,
    PreprocessedLengthPredictionTrainingData,
    PreprocessedLengthPredictionInferenceInput,
)
from src.preprocessing.prompt_embedding_preprocessor import PromptEmbeddingPreprocessor
from src.utils.jars import Jars
from src.utils.string_encoder import StringEncoder
from src.utils.timer import Timer
from src.preprocessing.simple_scaler import SimpleScaler


class LengthPredictionPreprocessor:
    """
    Preprocessor for length prediction models.
    
    This preprocessor:
    - Uses PromptEmbeddingPreprocessor internally for prompt embeddings and features
    - Extracts response lengths from evaluation entries
    - Scales response lengths to a friendlier range [0, 1]
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
        self.prompt_embedding_preprocessor = PromptEmbeddingPreprocessor(
            embedding_model_name=embedding_model_name,
            min_model_comparisons=min_model_comparisons,
        )
        
        self.version = "v1"
        self.last_timer: Timer | None = None

    def preprocess(
        self,
        data: TrainingData,
    ) -> PreprocessedLengthPredictionTrainingData:
        """
        Preprocess training data by embedding prompts and extracting response lengths.
        
        Args:
            data: Raw training data (same as for scoring models)
            
        Returns:
            Preprocessed training data with embeddings and response lengths (without model embeddings)
        """
        with Timer("preprocess", verbosity="start+end") as timer:
            self.last_timer = timer
            with Timer("generate_cache_key", verbosity="start+end", parent=timer):
                cache_key = self._generate_cache_key(data)
            
            if cache_key in Jars.preprocessed_data:
                return Jars.preprocessed_data.get(cache_key)
                
            with Timer("prompt_embedding_preprocessing", verbosity="start+end", parent=timer):
                preprocessed = self.prompt_embedding_preprocessor.preprocess(data)
            
            filtered_entries = [data.entries[i] for i in preprocessed.filtered_indexes]
            
            with Timer("compute_response_lengths", verbosity="start+end", parent=timer):
                response_lengths_a, response_lengths_b = self._compute_response_lengths(filtered_entries)
                
            # Fit scaler on all lengths and transform
            scaler = SimpleScaler().fit(np.array(response_lengths_a + response_lengths_b))
            
            response_lengths_a_scaled = scaler.transform(np.array(response_lengths_a))  # [n_pairs]
            response_lengths_b_scaled = scaler.transform(np.array(response_lengths_b))  # [n_pairs]
            
            assert len(preprocessed.pairs) == len(filtered_entries)
            
            samples = [
                PreprocessedLengthPredictionSample(
                    prompt_embedding=pair.prompt_embedding,
                    prompt_features=pair.prompt_features,
                    model_id_a=pair.model_id_a,
                    model_id_b=pair.model_id_b,
                    response_length_a=response_lengths_a_scaled[i],
                    response_length_b=response_lengths_b_scaled[i],
                )
                for i, pair in enumerate(preprocessed.pairs)
            ]
            
            preprocessed_data = PreprocessedLengthPredictionTrainingData(
                samples=samples,
                embedding_dim=preprocessed.embedding_dim,
                prompt_features_dim=preprocessed.prompt_features_dim,
                model_encoder=preprocessed.model_encoder,
                filtered_indexes=preprocessed.filtered_indexes,
                prompt_features_scaler_state=preprocessed.scaler_state,
                output_scaler_state=scaler.get_state_dict(),
            )
            
            Jars.preprocessed_data.add(cache_key, preprocessed_data)
            
            return preprocessed_data

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
        hasher = hashlib.sha256()
        hasher.update(str(len(data.entries)).encode())
        
        for entry in data.entries:
            hasher.update(entry.timestamp.encode())
        
        dataset_signature = hasher.hexdigest()[:16]
        return f"length_prediction/{self.version}-{self.prompt_embedding_preprocessor.version}/{self.prompt_embedding_preprocessor.embedding_model_name}-{dataset_signature}-{self.prompt_embedding_preprocessor.min_model_comparisons}"

    def preprocess_for_inference(
        self,
        prompts: list[str],
        model_names: list[str],
        model_encoder: StringEncoder,
        scaler: SimpleScaler,
    ) -> PreprocessedLengthPredictionInferenceInput:
        """
        Preprocess prompts and model names for inference.
        
        Args:
            prompts: List of prompts to embed
            model_names: List of model names to predict for
            model_encoder: Fitted model encoder from training
            scaler: Fitted scaler from training
            
        Returns:
            PreprocessedLengthPredictionInferenceInput with embeddings and features
        """
        
        preprocessed = self.prompt_embedding_preprocessor.preprocess_for_inference(
            prompts, 
            model_names, 
            model_encoder,
            scaler
        )
        
        return PreprocessedLengthPredictionInferenceInput(
            prompt_embeddings=preprocessed.prompt_embeddings,  # [n_prompts, embedding_dim]
            prompt_features=preprocessed.prompt_features,  # [n_prompts, prompt_features_dim]
        )

    def _compute_response_lengths(
        self, 
        entries: list[EvaluationEntry]
    ) -> tuple[list[float], list[float]]:
        """
        Compute approximate response lengths in tokens for model A and model B.
        
        Uses a simple heuristic: word count * 1.3 (reasonable approximation for English text).
        This is faster than using a tokenizer and sufficient for relative length prediction.
        
        Args:
            entries: List of evaluation entries
            
        Returns:
            Tuple of (lengths_a, lengths_b) where each is a list of estimated token counts
        """
        lengths_a = []
        lengths_b = []
        
        for entry in entries:
            # Approximate token count: words * 1.3
            # (most English words are 1 token, but some are split into multiple tokens)
            length_a = len(entry.model_a_response.split()) * 1.3
            length_b = len(entry.model_b_response.split()) * 1.3
            
            lengths_a.append(float(length_a))
            lengths_b.append(float(length_b))
        
        return lengths_a, lengths_b
