import hashlib
import numpy as np
from timeit import Timer
from src.data_models.data_models import TrainingData
from src.data_models.gradient_boosting_types import PreprocessedTrainingData, PreprocessedPromptPair, PreprocessedInferenceInput
from src.preprocessing.prompt_embedding_preprocessor import PromptEmbeddingPreprocessor
from src.utils.jars import Jars
from src.utils.string_encoder import StringEncoder


class PromptEmbeddingWithCategoriesPreprocessor:
    def __init__(
        self, 
        embedding_model_name: str = "all-MiniLM-L6-v2", 
        min_model_comparisons: int = 20,
    ) -> None:
        self.prompt_embedding_preprocessor = PromptEmbeddingPreprocessor(
            embedding_model_name=embedding_model_name,
            min_model_comparisons=min_model_comparisons,
        )
        
        self.version = "v1"
        self.category_dim = 11  # 1 (creative_writing) + 7 (criteria) + 2 (if) + 1 (math)
        self.last_timer: Timer | None = None

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
            
            if cache_key in Jars.preprocessed_data:
                return Jars.preprocessed_data.get(cache_key)
            
            with Timer("filter_entries_and_fit_encoder", verbosity="start+end", parent=timer):
                # We call filtering here, so that no data is filtered out in `preprocess()` call - this way we can zip() with unprocessed data to add categories
                filtered_data, _ = self.prompt_embedding_preprocessor._filter_data_and_fit_encoder(data)
            
            with Timer("prompt_embedding_preprocessing", verbosity="start+end", parent=timer):
                preprocessed = self.prompt_embedding_preprocessor.preprocess(filtered_data)
            
            with Timer("add_categories", verbosity="start+end", parent=timer):
                categories = self._compute_categories(filtered_data)
                
            assert len(preprocessed.pairs) == len(filtered_data.entries)
            
            pairs = [
                PreprocessedPromptPair(
                    prompt_embedding=pair.prompt_embedding,
                    prompt_features=pair.prompt_features,
                    prompt_categories=category,
                    model_embedding_a=np.array([]), # Placeholder, will be filled by model
                    model_embedding_b=np.array([]), # Placeholder, will be filled by model
                    model_id_a=pair.model_id_a,
                    model_id_b=pair.model_id_b,
                    winner_label=pair.winner_label,
                )
                for category, pair in zip(categories, preprocessed.pairs)
            ]
            preprocessed_data = PreprocessedTrainingData(
                pairs=pairs,
                embedding_dim=preprocessed.embedding_dim,
                prompt_features_dim=preprocessed.prompt_features_dim,
                prompt_categories_dim=len(categories[0]),
                model_encoder=preprocessed.model_encoder,
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
        # Use session IDs and timestamps as a proxy for dataset identity
        hasher = hashlib.sha256()
        hasher.update(str(len(data.entries)).encode())
        
        for entry in data.entries:
            hasher.update(entry.timestamp.encode())
        
        dataset_signature = hasher.hexdigest()[:16]
        return f"prompt_embedding_with_categories/{self.version}/{self.embedding_model_name}-{dataset_signature}-{self.min_model_comparisons}"

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
            PreprocessedInferenceInput with embeddings, features, and model IDs
        """
        
        preprocessed = self.prompt_embedding_preprocessor.preprocess_for_inference(prompts, model_names, model_encoder)
        categories = [np.zeros(self.category_dim) for _ in prompts] # TODO: How to get categories for inference?
        return PreprocessedInferenceInput(
            prompt_embeddings=preprocessed.prompt_embeddings, # [n_prompts, embedding_dim]
            prompt_features=preprocessed.prompt_features,  # [n_prompts, prompt_features_dim]
            prompt_categories=np.stack(categories), # [n_prompts, prompt_categories_dim]
            model_embeddings=preprocessed.model_embeddings, # [n_models, model_embedding_dim]
        )
        
    def _compute_categories(self, data: TrainingData) -> list[np.ndarray]:
        """
        Compute categories for a list of prompts.
        
        Args:
            data: Training data to compute categories for
            
        Returns:
            List of categories for each prompt [prompt_categories_dim]
        """
        categories = []
        
        for entry in data.entries:
            if entry.category_tag is None:
                # If no category tag, use zeros
                categories.append(np.zeros(self.category_dim))
            else:
                # Convert CategoryTag to numpy array
                category_array = self._category_tag_to_array(entry.category_tag)
                categories.append(category_array)
        
        return categories
    
    def _category_tag_to_array(self, tag) -> np.ndarray:
        """
        Convert a CategoryTag to a numpy array.
        
        Structure: [creative_writing(1), criteria(7), if(2), math(1)]
        Total: 11 dimensions
        
        Args:
            tag: CategoryTag to convert
            
        Returns:
            Numpy array of shape [11]
        """
        result = np.zeros(self.category_dim)
        idx = 0
        
        # Creative writing (1 feature)
        result[0] = float(tag.creative_writing_v0_1.creative_writing)
        idx += 1
        
        # Criteria (7 features)
        result[1] = float(tag.criteria_v0_1.complexity)
        result[2] = float(tag.criteria_v0_1.creativity)
        result[3] = float(tag.criteria_v0_1.domain_knowledge)
        result[4] = float(tag.criteria_v0_1.problem_solving)
        result[5] = float(tag.criteria_v0_1.real_world)
        result[6] = float(tag.criteria_v0_1.specificity)
        result[7] = float(tag.criteria_v0_1.technical_accuracy)
        
        # IF (2 features: bool + score)
        result[8] = float(tag.if_v0_1.if_)
        result[9] = float(tag.if_v0_1.score)
        
        # Math (1 feature)
        result[10] = float(tag.math_v0_1.math)
        
        return result
