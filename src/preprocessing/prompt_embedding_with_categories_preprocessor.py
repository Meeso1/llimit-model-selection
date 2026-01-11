import hashlib
import numpy as np
from src.data_models.data_models import CategoryTag, EvaluationEntry, TrainingData
from src.data_models.gradient_boosting_types import PreprocessedTrainingData, PreprocessedPromptPair, PreprocessedInferenceInput
from src.preprocessing.prompt_embedding_preprocessor import PromptEmbeddingPreprocessor
from src.utils.jars import Jars
from src.utils.string_encoder import StringEncoder
from src.utils.timer import Timer


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
        self.category_dim = 12  # 2 (creative_writing: bool + score) + 7 (criteria) + 2 (if: bool + score) + 1 (math)
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
                
            with Timer("prompt_embedding_preprocessing", verbosity="start+end", parent=timer):
                preprocessed = self.prompt_embedding_preprocessor.preprocess(data)
            
            filtered_entries = [data.entries[i] for i in preprocessed.filtered_indexes]
            
            with Timer("add_categories", verbosity="start+end", parent=timer):
                categories = self._compute_categories(filtered_entries)
                
            assert len(preprocessed.pairs) == len(filtered_entries)
            
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
                prompt_categories_dim=self.category_dim,
                model_encoder=preprocessed.model_encoder,
                filtered_indexes=preprocessed.filtered_indexes,
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
        return f"prompt_embedding_with_categories/{self.version}-{self.prompt_embedding_preprocessor.version}/{self.prompt_embedding_preprocessor.embedding_model_name}-{dataset_signature}-{self.prompt_embedding_preprocessor.min_model_comparisons}"

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
        
    def _compute_categories(self, entries: list[EvaluationEntry]) -> list[np.ndarray]:
        """
        Compute categories for a list of prompts.
        
        Args:
            entries: List of entries to compute categories for
            
        Returns:
            List of categories for each prompt [prompt_categories_dim]
        """
        categories = []
        
        for entry in entries:
            if entry.category_tag is None:
                # If no category tag, use zeros
                categories.append(np.zeros(self.category_dim))
            else:
                # Convert CategoryTag to numpy array
                category_array = self._category_tag_to_array(entry.category_tag)
                categories.append(category_array)
        
        return categories
    
    def _category_tag_to_array(self, tag: CategoryTag) -> np.ndarray:
        """
        Convert a CategoryTag to a numpy array.
        
        Structure: [creative_writing_bool(1), creative_writing_score(1), criteria(7), if_bool(1), if_score(1), math(1)]
        Total: 12 dimensions
        
        Args:
            tag: CategoryTag to convert
            
        Returns:
            Numpy array of shape [12]
        """
        result = np.full(self.category_dim, fill_value=np.nan)
        
        # Creative writing (2 features: bool + score)
        result[0] = float(tag.creative_writing_v0_1.creative_writing)
        if tag.creative_writing_v0_1.score == "yes":
            result[1] = 1.0
        elif tag.creative_writing_v0_1.score == "no":
            result[1] = 0.0
        else:  # None
            result[1] = 0.5
        
        # Criteria (7 features)
        result[2] = float(tag.criteria_v0_1.complexity)
        result[3] = float(tag.criteria_v0_1.creativity)
        result[4] = float(tag.criteria_v0_1.domain_knowledge)
        result[5] = float(tag.criteria_v0_1.problem_solving)
        result[6] = float(tag.criteria_v0_1.real_world)
        result[7] = float(tag.criteria_v0_1.specificity)
        result[8] = float(tag.criteria_v0_1.technical_accuracy)
        
        # IF (2 features: bool + score normalized 0-5 -> 0.0-1.0)
        result[9] = float(tag.if_v0_1.if_)
        if tag.if_v0_1.score is not None:
            result[10] = float(tag.if_v0_1.score) / 5.0
        else:
            result[10] = 0.5
        
        # Math (1 feature)
        result[11] = float(tag.math_v0_1.math)
        
        assert np.isnan(result).sum() == 0, f"{len(result) - np.isnan(result).sum()} NaN values in category array"
        
        return result
