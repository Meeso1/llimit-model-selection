"""Preprocessor for response predictive model with response embeddings and features."""

import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from src.data_models.data_models import TrainingData
from src.data_models.response_predictive_types import (
    ResponsePredictivePair,
    PreprocessedTrainingData,
    ResponsePredictiveInferenceData,
)
from src.preprocessing.simple_scaler import SimpleScaler
from src.utils.jars import Jars
from src.utils.string_encoder import StringEncoder
from src.utils.timer import Timer
from src.preprocessing.utils import (
    filter_out_ties,
    filter_out_both_bad,
    filter_out_empty_entries,
    filter_out_rare_models,
    create_encoder,
)
from src.preprocessing.scoring_feature_extraction import extract_and_transform_all_prompt_features
from src.preprocessing.model_embedding_feature_extraction import extract_response_features_for_many


class ResponsePredictivePreprocessor:
    """
    Preprocessor that embeds prompts and responses using sentence transformers.
    
    This preprocessor:
    - Filters out ties and both_bad from training data
    - Embeds user prompts and responses using a sentence transformer model
    - Extracts scalar features for responses (interaction, lexical, structural)
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
        Preprocess training data by embedding prompts/responses and extracting features.
        
        Args:
            data: Raw training data
            
        Returns:
            Preprocessed training data with embeddings, features, and scalers
        """
        with Timer("preprocess", verbosity="start+end") as timer:
            self.last_timer = timer
            with Timer("generate_cache_key", verbosity="start+end", parent=timer):
                cache_key = self._generate_cache_key(data)
            
            if cache_key in Jars.preprocessed_data:
                return Jars.preprocessed_data.get(cache_key)
            
            with Timer("filter_entries_and_fit_encoder", verbosity="start+end", parent=timer):
                filtered_data, model_encoder, filtered_indexes = self._filter_data_and_fit_encoder(data)
            
            with Timer("extract_prompt_features", verbosity="start+end", parent=timer) as prompt_features_timer:
                prompt_features_list, prompt_scaler = extract_and_transform_all_prompt_features(
                    [entry.user_prompt for entry in filtered_data.entries],
                    [entry.conversation_history for entry in filtered_data.entries],
                    timer=prompt_features_timer
                )
            
            with Timer("embed_prompts", verbosity="start+end", parent=timer):
                prompts = [entry.user_prompt for entry in filtered_data.entries]
                prompt_embeddings = self._embed_texts(prompts).cpu()  # [n_prompts, embedding_dim]
            
            with Timer("embed_responses", verbosity="start+end", parent=timer):
                responses_a = [entry.model_a_response for entry in filtered_data.entries]
                responses_b = [entry.model_b_response for entry in filtered_data.entries]
                response_embeddings_a = self._embed_texts(responses_a).cpu()  # [n_prompts, embedding_dim]
                response_embeddings_b = self._embed_texts(responses_b).cpu()  # [n_prompts, embedding_dim]
            
            with Timer("extract_and_scale_response_features", verbosity="start+end", parent=timer) as response_features_timer:
                # Extract features for both A and B, fitting scaler on all entries
                all_prompt_embeddings = np.concatenate([prompt_embeddings.numpy(), prompt_embeddings.numpy()])  # [2*n, d_emb]
                all_response_embeddings = np.concatenate([response_embeddings_a.numpy(), response_embeddings_b.numpy()])  # [2*n, d_emb]
                all_prompts = prompts + prompts
                all_responses = responses_a + responses_b
                
                all_features_list, response_scaler = extract_response_features_for_many(
                    all_prompt_embeddings,
                    all_response_embeddings,
                    all_prompts,
                    all_responses,
                    scaler=None,
                    timer=response_features_timer,
                )
                
                # Split back into A and B
                n = len(prompts)
                response_features_list_a = all_features_list[:n]
                response_features_list_b = all_features_list[n:]
            
            pairs: list[ResponsePredictivePair] = []
            for i, entry in enumerate(filtered_data.entries):
                winner_label = 0 if entry.winner == "model_a" else 1
                pairs.append(
                    ResponsePredictivePair(
                        prompt_embedding=prompt_embeddings[i].numpy(),
                        prompt_features=prompt_features_list[i],
                        response_embedding_a=response_embeddings_a[i].numpy(),
                        response_embedding_b=response_embeddings_b[i].numpy(),
                        response_features_a=response_features_list_a[i],
                        response_features_b=response_features_list_b[i],
                        model_id_a=model_encoder.encode(entry.model_a),
                        model_id_b=model_encoder.encode(entry.model_b),
                        winner_label=winner_label,
                    )
                )
            
            prompt_features_dim = pairs[0].prompt_features.shape[0]
            response_features_dim = pairs[0].response_features_a.shape[0]
            
            preprocessed_data = PreprocessedTrainingData(
                pairs=pairs,
                prompt_features_dim=prompt_features_dim,
                response_features_dim=response_features_dim,
                model_encoder=model_encoder,
                filtered_indexes=filtered_indexes,
                prompt_scaler_state=prompt_scaler.get_state_dict(),
                response_scaler_state=response_scaler.get_state_dict(),
            )
            
            Jars.preprocessed_data.add(cache_key, preprocessed_data)
            
            return preprocessed_data

    def _filter_data_and_fit_encoder(self, data: TrainingData) -> tuple[TrainingData, StringEncoder, list[int]]:
        """Filter data and create model encoder."""
        filtered_data, indexes = filter_out_rare_models(data, self.min_model_comparisons)
        filtered_data, indexes = filter_out_ties(filtered_data, indexes)
        filtered_data, indexes = filter_out_both_bad(filtered_data, indexes)
        filtered_data, indexes = filter_out_empty_entries(filtered_data, indexes)
        
        if len(filtered_data.entries) == 0:
            raise ValueError(
                "No valid training data after filtering. "
                f"All models were filtered out (min_model_comparisons={self.min_model_comparisons}). "
                "Try lowering min_model_comparisons or providing more training data."
            )

        model_encoder = create_encoder(filtered_data)
        
        return filtered_data, model_encoder, indexes

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
        
        for entry in data.entries:
            hasher.update(entry.timestamp.encode())
        
        dataset_signature = hasher.hexdigest()[:16]
        return f"response_predictive/{self.version}/{self.embedding_model_name}-{dataset_signature}-{self.min_model_comparisons}"

    def preprocess_for_inference(
        self,
        prompts: list[str],
        model_names: list[str],
        prompt_scaler: SimpleScaler,
        model_embeddings: dict[str, np.ndarray],
    ) -> ResponsePredictiveInferenceData:
        """
        Preprocess prompts and model names for inference.
        
        Args:
            prompts: List of prompts to embed
            model_names: List of model names to score
            prompt_scaler: Fitted prompt feature scaler from training
            model_embeddings: Dictionary mapping model names to embeddings
            
        Returns:
            ResponsePredictiveInferenceData with embeddings and features
        """
        prompt_features_list, _ = extract_and_transform_all_prompt_features(
            prompts,
            [[] for _ in prompts],
            prompt_scaler,
        )
        prompt_embeddings = self._embed_texts(prompts).cpu()  # [n_prompts, embedding_dim]
        
        # Get model embeddings
        model_embs = np.stack([
            model_embeddings[name] if name in model_embeddings else model_embeddings.get("default", np.zeros(len(next(iter(model_embeddings.values())))))
            for name in model_names
        ])  # [n_models, model_embedding_dim]
        
        return ResponsePredictiveInferenceData(
            prompt_embeddings=prompt_embeddings.numpy(),  # [n_prompts, embedding_dim]
            prompt_features=np.stack(prompt_features_list),  # [n_prompts, prompt_features_dim]
            model_embeddings=model_embs,  # [n_models, model_embedding_dim]
        )

    def _embed_texts(self, texts: list[str]) -> torch.Tensor:
        """
        Embed a list of texts using the sentence transformer model.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Tensor of embeddings  # [n_texts, embedding_dim]
        """
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return embeddings
