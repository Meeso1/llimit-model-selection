"""Preprocessor for the attention-based embedding model."""

import hashlib
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from src.data_models.data_models import TrainingData
from src.data_models.attention_embedding_types import (
    ProcessedPair,
    ModelSetSample,
    PreprocessedAttentionEmbeddingData,
    ScalerState,
)
from src.preprocessing.model_embedding_feature_extraction import extract_all_scalar_features
from src.preprocessing.simple_scaler import SimpleScaler
from src.utils.jars import Jars
from src.utils.timer import Timer
from src.preprocessing.utils import filter_out_empty_entries, filter_out_rare_models


class AttentionEmbeddingPreprocessor:
    """
    Preprocessor for the attention-based embedding model.
    
    This preprocessor:
    - Filters out rare models (< min_model_comparisons)
    - Filters out invalid entries (empty prompts/responses)
    - Embeds prompts and responses using a sentence transformer
    - Extracts stylometric and interaction features
    - Normalizes scalar features using SimpleScaler
    - Groups pairs by model
    - Caches preprocessed data for efficiency
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        min_model_comparisons: int = 20,
        seed: int = 42,
    ) -> None:
        """
        Initialize the preprocessor.
        
        Args:
            embedding_model_name: Name of the sentence transformer model
            min_model_comparisons: Minimum number of comparisons for a model to be included
            seed: Random seed for reproducibility
        """
        self.embedding_model_name = embedding_model_name
        self.min_model_comparisons = min_model_comparisons
        self.seed = seed
        self.version = "v2"
        self._embedding_model: SentenceTransformer | None = None
        self.last_timer: Timer | None = None
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model
    
    def preprocess(
        self,
        data: TrainingData,
    ) -> PreprocessedAttentionEmbeddingData:
        """
        Preprocess training data.
        
        Args:
            data: Raw training data
            
        Returns:
            Preprocessed data with model sets and fitted scaler
        """
        with Timer("preprocess_attention_embedding", verbosity="start+end") as timer:
            self.last_timer = timer
            
            with Timer("generate_cache_key", verbosity="start+end", parent=timer):
                cache_key = self._generate_cache_key(data)
            
            if cache_key in Jars.preprocessed_data:
                return Jars.preprocessed_data.get(cache_key)
            
            with Timer("filter_data", verbosity="start+end", parent=timer):
                filtered_data, indexes = self._filter_data(data)
            
            with Timer("extract_features", verbosity="start+end", parent=timer):
                pairs_by_model, scaler_state = self._extract_features(filtered_data, indexes, timer)
            
            # Create model ID mapping
            model_ids = sorted(pairs_by_model.keys())
            model_id_to_index = {model_id: i for i, model_id in enumerate(model_ids)}
            
            # Create ModelSetSamples
            samples = []
            for model_id, (pairs, indexes) in pairs_by_model.items():
                samples.append(ModelSetSample(pairs=pairs, model_id=model_id, indexes=indexes))
            
            preprocessed_data = PreprocessedAttentionEmbeddingData(
                samples=samples,
                model_id_to_index=model_id_to_index,
                scaler_state=scaler_state,
            )
            
            Jars.preprocessed_data.add(cache_key, preprocessed_data)
            
            return preprocessed_data
    
    def _filter_data(self, data: TrainingData) -> tuple[TrainingData, list[int]]:
        filtered_data, indexes = filter_out_rare_models(data, self.min_model_comparisons)
        filtered_data, indexes = filter_out_empty_entries(filtered_data, indexes)
        if len(filtered_data.entries) == 0:
            raise ValueError(
                "No valid training data after filtering. "
                f"All models were filtered out (min_model_comparisons={self.min_model_comparisons}). "
                "Try lowering min_model_comparisons or providing more training data."
            )
        
        return filtered_data, indexes
    
    def _extract_features(
        self, 
        data: TrainingData,
        indexes: list[int],
        timer: Timer
    ) -> tuple[dict[str, tuple[list[ProcessedPair], list[int]]], ScalerState]:
        """
        Extract all features for each (prompt, response) pair.
        
        Args:
            data: Filtered training data
            timer: Timer for profiling
            
        Returns:
            Tuple of (dictionary mapping model_id to list of ProcessedPairs, scaler state)
        """
        # Collect all unique (prompt, response) pairs with their models
        unique_pairs: dict[tuple[str, str], str] = {}  # (prompt, response) -> model_id
        
        for entry in data.entries:
            unique_pairs[(entry.user_prompt, entry.model_a_response)] = entry.model_a
            unique_pairs[(entry.user_prompt, entry.model_b_response)] = entry.model_b
        
        # Extract embeddings
        with Timer("embed_texts", verbosity="start+end", parent=timer):
            prompts = [pair[0] for pair in unique_pairs.keys()]
            responses = [pair[1] for pair in unique_pairs.keys()]
            
            prompt_embeddings = self._embed_texts(prompts)  # [n_pairs, d_emb]
            response_embeddings = self._embed_texts(responses)  # [n_pairs, d_emb]
        
        # Extract scalar features
        with Timer("extract_scalar_features", verbosity="start+end", parent=timer) as features_timer:
            scalar_features_list = []
            for i, (prompt, response) in enumerate(unique_pairs.keys()):
                scalar_features = extract_all_scalar_features(
                    prompt_embeddings[i],
                    response_embeddings[i],
                    prompt,
                    response,
                    timer=features_timer
                )
                scalar_features_list.append(scalar_features)
        
        # Normalize scalar features
        with Timer("normalize_features", verbosity="start+end", parent=timer):
            scalar_features_array = np.stack(scalar_features_list)  # [n_pairs, n_features]
            scaler = SimpleScaler()
            normalized_scalar_features = scaler.fit_transform(scalar_features_array)
            scaler_state = ScalerState(
                mean=scaler.mean,
                scale=scaler.scale,
            )
        
        # Group by model
        pairs_by_model: dict[str, list[ProcessedPair]] = defaultdict(list)
        indexes_by_model: dict[str, list[int]] = defaultdict(list)
        
        for i, (index, ((prompt, response), model_id)) in enumerate(zip(indexes, unique_pairs.items())):
            processed_pair = ProcessedPair(
                prompt_emb=prompt_embeddings[i],
                response_emb=response_embeddings[i],
                scalar_features=normalized_scalar_features[i],
                model_id=model_id,
            )
            pairs_by_model[model_id].append(processed_pair)
            indexes_by_model[model_id].append(index)
            
        result = {
            model_id: (pairs, indexes_by_model[model_id])
            for model_id, pairs in pairs_by_model.items()
        }
        return result, scaler_state
    
    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts using the sentence transformer.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings  # [n_texts, d_emb]
        """
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True,
        )
        return embeddings.cpu().numpy()
    
    def _generate_cache_key(self, data: TrainingData) -> str:
        """
        Generate cache key for a dataset.
        
        Args:
            data: Training dataset
            
        Returns:
            Cache key string
        """
        hasher = hashlib.sha256()
        hasher.update(str(len(data.entries)).encode())
        hasher.update(str(self.min_model_comparisons).encode())
        hasher.update(str(self.seed).encode())
        hasher.update(self.embedding_model_name.encode())
        
        for entry in data.entries:
            hasher.update(entry.timestamp.encode())
        
        dataset_signature = hasher.hexdigest()[:16]
        params_str = f"{self.embedding_model_name}-{self.min_model_comparisons}-{self.seed}"
        return f"attention_embedding/{self.version}/{params_str}-{dataset_signature}"
    
    def process_single_pair(
        self, 
        prompt: str, 
        response: str,
        scaler_state: ScalerState
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a single (prompt, response) pair for inference.
        
        Args:
            prompt: Prompt text
            response: Response text
            scaler_state: Fitted scaler state
            
        Returns:
            Tuple of (prompt_emb, response_emb, scalar_features)
        """
        # Extract embeddings
        prompt_emb = self._embed_texts([prompt])[0]
        response_emb = self._embed_texts([response])[0]
        
        # Extract scalar features
        scalar_features = extract_all_scalar_features(
            prompt_emb, response_emb, prompt, response
        )
        
        # Normalize using provided scaler state
        scaler = SimpleScaler()
        scaler.load_state_dict({
            'mean': scaler_state.mean,
            'scale': scaler_state.scale,
        })
        scalar_features_normalized = scaler.transform(
            scalar_features.reshape(1, -1)
        )[0]
        
        return prompt_emb, response_emb, scalar_features_normalized
