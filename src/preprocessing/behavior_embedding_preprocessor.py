from dataclasses import dataclass
import hashlib
import random
from collections import Counter
import numpy as np

from sentence_transformers import SentenceTransformer

from src.constants import PREPROCESSED_DATA_JAR_PATH
from src.data_models.data_models import EvaluationEntry, TrainingData
from src.data_models.behavior_encoder_types import (
    PromptResponsePair,
    PromptResponsePairEmbedding,
    TrainingTriplet,
    PreprocessedBehaviorEncoderData,
    TripletEmbedding,
)
from src.utils.jar import Jar
from src.utils.timer import Timer


class BehaviorEmbeddingPreprocessor:
    """
    Preprocessor for the ModelBehaviorEncoder.
    
    This preprocessor:
    - Filters rare models (< min_model_comparisons)
    - Filters invalid entries (empty prompts/responses)
    - Constructs training triplets using a strategic selection process
    - Incorporates tie and both_bad entries
    - Caches preprocessed data for efficiency
    """
    
    def __init__(
        self,
        min_model_comparisons: int = 20,
        identity_positive_ratio: float = 0.8,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        seed: int = 42,
    ) -> None:
        """
        Initialize the preprocessor.
        
        Args:
            min_model_comparisons: Minimum number of comparisons for a model to be included
            identity_positive_ratio: Ratio of identity positives (vs performance positives)
            seed: Random seed for reproducible triplet construction
        """
        self.min_model_comparisons = min_model_comparisons
        self.identity_positive_ratio = identity_positive_ratio
        self.seed = seed
        self.embedding_model_name = embedding_model_name
        self.version = "v1"
        self.jar = Jar(str(PREPROCESSED_DATA_JAR_PATH))
        self.last_timer: Timer | None = None
    
    def preprocess(
        self,
        data: TrainingData,
    ) -> PreprocessedBehaviorEncoderData:
        """
        Preprocess training data by constructing triplets.
        
        Args:
            data: Raw training data
            
        Returns:
            Preprocessed data with training triplets
        """
        with Timer("preprocess_behavior", verbosity="start+end") as timer:
            self.last_timer = timer
            
            with Timer("generate_cache_key", verbosity="start+end", parent=timer):
                cache_key = self._generate_cache_key(data)
            
            if cache_key in self.jar:
                return self.jar.get(cache_key)
            
            with Timer("filter", verbosity="start+end", parent=timer):
                filtered_data = self._filter_data(data)

            with Timer("make_triplets", verbosity="start+end", parent=timer):
                triplets = self._construct_triplets(filtered_data)

            with Timer("generate_embeddings", verbosity="start+end", parent=timer):
                embeddings = self._generate_embeddings(triplets)
            
            preprocessed_data = PreprocessedBehaviorEncoderData(triplets=embeddings)
            self.jar.add(cache_key, preprocessed_data)
            
            return preprocessed_data
    
    def preprocess_for_inference(
        self,
        pairs: list[PromptResponsePair]
    ) -> list[PromptResponsePairEmbedding]:
        embedding_model = SentenceTransformer(self.embedding_model_name)
        result = []
        cache = {}
        for pair in pairs:
            result.append(PromptResponsePairEmbedding(
                prompt=self._embed_or_get_cached(pair.prompt, embedding_model, cache),
                response=self._embed_or_get_cached(pair.response, embedding_model, cache)
            ))
        return result

    def _filter_data(self, data: TrainingData) -> TrainingData:
        """
        Filter out invalid and rare model entries.
        
        Args:
            data: Raw training data
            
        Returns:
            Filtered training data
        """
        # Filter out entries with empty prompts or responses
        valid_entries = [
            entry for entry in data.entries
            if (entry.user_prompt.strip() 
                and entry.model_a_response.strip() 
                and entry.model_b_response.strip())
        ]
        
        # Count model appearances
        model_counts = Counter()
        for entry in valid_entries:
            model_counts[entry.model_a] += 1
            model_counts[entry.model_b] += 1
        
        # Filter out rare models
        frequent_models = {
            model for model, count in model_counts.items()
            if count >= self.min_model_comparisons
        }
        
        filtered_entries = [
            entry for entry in valid_entries
            if entry.model_a in frequent_models and entry.model_b in frequent_models
        ]

        # Filter out `both_bad` - we don't use these currently
        no_both_bad = [
            entry for entry in filtered_entries
            if entry.winner != "both_bad"
        ]
        
        return TrainingData(entries=no_both_bad)

    def _construct_triplets(self, data: TrainingData) -> list[TrainingTriplet]:
        all_examples = self._make_all_examples(data.entries)
        all_winning_examples = self._make_all_winning_examples(data.entries)
        examples_by_model = self._make_examples_by_model(data.entries)

        triplets = []
        triplets.extend(self._construct_triplets_from_win_lose_pairs(
            [e for e in data.entries if e.winner in ["model_a", "model_b"]],
            examples_by_model,
            all_winning_examples
        ))
        triplets.extend(self._construct_triplets_from_tie_pairs(
            [e for e in data.entries if e.winner == "tie"],
            all_examples
        ))

        return triplets

    def _make_all_examples(self, pairs: list[EvaluationEntry]) -> list["BehaviorEmbeddingPreprocessor.ModelExample"]:
        result = []
        for pair in pairs:
            result.append(BehaviorEmbeddingPreprocessor.ModelExample(
                prompt=pair.user_prompt,
                response=pair.model_a_response,
                model=pair.model_a
            ))
            result.append(BehaviorEmbeddingPreprocessor.ModelExample(
                prompt=pair.user_prompt,
                response=pair.model_b_response,
                model=pair.model_b
            ))
        return result

    def _make_all_winning_examples(self, pairs: list[EvaluationEntry]) -> list["BehaviorEmbeddingPreprocessor.ModelExample"]:
        result = []
        for pair in pairs:
            if pair.winner == "model_a" or pair.winner == "tie":
                result.append(BehaviorEmbeddingPreprocessor.ModelExample(
                    prompt=pair.user_prompt,
                    response=pair.model_a_response,
                    model=pair.model_a
                ))
            if pair.winner == "model_b" or pair.winner == "tie":
                result.append(BehaviorEmbeddingPreprocessor.ModelExample(
                    prompt=pair.user_prompt,
                    response=pair.model_b_response,
                    model=pair.model_b
                ))

        return result

    def _make_examples_by_model(self, pairs: list[EvaluationEntry]) -> dict[str, list["BehaviorEmbeddingPreprocessor.ModelExample"]]:
        result = {}
        for pair in pairs:
            if pair.model_a not in result:
                result[pair.model_a] = []
            if pair.model_b not in result:
                result[pair.model_b] = []
            result[pair.model_a].append(BehaviorEmbeddingPreprocessor.ModelExample(
                prompt=pair.user_prompt,
                response=pair.model_a_response,
                model=pair.model_a
            ))
            result[pair.model_b].append(BehaviorEmbeddingPreprocessor.ModelExample(
                prompt=pair.user_prompt,
                response=pair.model_b_response,
                model=pair.model_b
            ))

        return result

    def _construct_triplets_from_win_lose_pairs(
        self, 
        pairs: list[EvaluationEntry], 
        examples_by_model: dict[str, list["BehaviorEmbeddingPreprocessor.ModelExample"]],
        all_winning_examples: list["BehaviorEmbeddingPreprocessor.ModelExample"]
    ) -> list[TrainingTriplet]:       
        triplets = []
        for pair in pairs:
            anchor_prompt = pair.user_prompt

            # 1st triplet: winning as anchor, losing as negative
            # Choose positive as either something else from the same model, or any winning example
            first_anchor_response = pair.model_a_response \
                if pair.winner == "model_a" \
                else pair.model_b_response
            first_anchor_model = pair.model_a \
                if pair.winner == "model_a" \
                else pair.model_b

            first_negative_prompt = pair.user_prompt
            first_negative_response = pair.model_b_response \
                if pair.winner == "model_a" \
                else pair.model_a_response

            if random.random() < self.identity_positive_ratio:
                first_positive_example = random.choice(examples_by_model[first_anchor_model])
            else:
                first_positive_example = random.choice(all_winning_examples)

            triplets.append(TrainingTriplet(
                anchor_prompt=anchor_prompt,
                anchor_response=first_anchor_response,
                positive_prompt=first_positive_example.prompt,
                positive_response=first_positive_example.response,
                negative_prompt=first_negative_prompt,
                negative_response=first_negative_response
            ))

            # 2nd triplet: losing as anchor, winning as negative
            # Choose positive as something else from the same model
            second_anchor_response = pair.model_a_response \
                if pair.winner == "model_b" \
                else pair.model_b_response
            second_anchor_model = pair.model_a \
                if pair.winner == "model_b" \
                else pair.model_b

            second_negative_prompt = pair.user_prompt
            second_negative_response = pair.model_a_response \
                if pair.winner == "model_b" \
                else pair.model_b_response

            second_positive_example = random.choice(examples_by_model[second_anchor_model])
            triplets.append(TrainingTriplet(
                anchor_prompt=anchor_prompt,
                anchor_response=second_anchor_response,
                positive_prompt=second_positive_example.prompt,
                positive_response=second_positive_example.response,
                negative_prompt=second_negative_prompt,
                negative_response=second_negative_response
            ))

        return triplets

    def _construct_triplets_from_tie_pairs(
        self, 
        pairs: list[EvaluationEntry], 
        all_examples: list["BehaviorEmbeddingPreprocessor.ModelExample"]
    ) -> list[TrainingTriplet]:
        triplets = []
        for pair in pairs:
            anchor_prompt = pair.user_prompt

            # 1st triplet: tie as anchor, winning as positive
            # Choose negative as any example
            first_anchor_response = pair.model_a_response
            first_positive_prompt = pair.user_prompt
            first_positive_response = pair.model_b_response

            negative_example = random.choice(all_examples)
            triplets.append(TrainingTriplet(
                anchor_prompt=anchor_prompt,
                anchor_response=first_anchor_response,
                positive_prompt=first_positive_prompt,
                positive_response=first_positive_response,
                negative_prompt=negative_example.prompt,
                negative_response=negative_example.response
            ))
                
            # 2nd triplet: tie as anchor, losing as positive
            # Same negative as in first triplet
            second_anchor_response = pair.model_b_response
            second_positive_prompt = pair.user_prompt
            second_positive_response = pair.model_a_response

            triplets.append(TrainingTriplet(
                anchor_prompt=anchor_prompt,
                anchor_response=second_anchor_response,
                positive_prompt=second_positive_prompt,
                positive_response=second_positive_response,
                negative_prompt=negative_example.prompt,
                negative_response=negative_example.response
            ))

    def _generate_embeddings(self, triplets: list[TrainingTriplet]) -> list[TripletEmbedding]:
        embedding_model = SentenceTransformer(self.embedding_model_name)

        result = []
        cache = {}
        for triplet in triplets:
            result.append(TripletEmbedding(
                anchor_prompt=self._embed_or_get_cached(triplet.anchor_prompt, embedding_model, cache),
                anchor_response=self._embed_or_get_cached(triplet.anchor_response, embedding_model, cache),
                positive_prompt=self._embed_or_get_cached(triplet.positive_prompt, embedding_model, cache),
                positive_response=self._embed_or_get_cached(triplet.positive_response, embedding_model, cache),
                negative_prompt=self._embed_or_get_cached(triplet.negative_prompt, embedding_model, cache),
                negative_response=self._embed_or_get_cached(triplet.negative_response, embedding_model, cache)
            ))
        
        return result

    # TODO: Instead of doing that, call the model.encode() once with all texts
    def _embed_or_get_cached(self, text: str, model: SentenceTransformer, cache: dict[str, np.ndarray]) -> np.ndarray:
        if text in cache:
            return cache[text]
        
        embedding = model.encode(text).cpu().numpy()
        cache[text] = embedding
        return embedding

    def _generate_cache_key(self, data: TrainingData) -> str:
        """
        Generate cache key for a dataset.
        
        The cache key is based on:
        - Preprocessor version and parameters
        - Hash of dataset's entries
        
        Args:
            data: Training dataset
            
        Returns:
            Cache key string
        """
        hasher = hashlib.sha256()
        hasher.update(str(len(data.entries)).encode())
        hasher.update(str(self.min_model_comparisons).encode())
        hasher.update(str(self.identity_positive_ratio).encode())
        hasher.update(str(self.seed).encode())
        
        for entry in data.entries:
            hasher.update(entry.timestamp.encode())
        
        dataset_signature = hasher.hexdigest()[:16]
        params_str = f"{self.min_model_comparisons}-{self.identity_positive_ratio}-{self.seed}"
        return f"behavior_embedding/{self.version}/{params_str}-{dataset_signature}"

    @dataclass
    class ModelExample:
        prompt: str
        response: str
        model: str
