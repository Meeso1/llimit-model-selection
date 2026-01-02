from dataclasses import dataclass
import hashlib
import random

from src.data_models.data_models import EvaluationEntry, TrainingData
from src.data_models.triplet_encoder_types import (
    TrainingTriplet,
    PreprocessedTripletEncoderData,
)
from src.preprocessing.utils import filter_out_both_bad, filter_out_empty_entries, filter_out_rare_models
from src.utils.jars import Jars
from src.utils.timer import Timer


class TripletFinetunableEncoderPreprocessor:
    """
    Preprocessor for the TripletFinetunableEncoderModel.
    
    This preprocessor:
    - Filters rare models (< min_model_comparisons)
    - Filters invalid entries (empty prompts/responses)
    - Constructs training triplets using a strategic selection process
    - Incorporates tie and both_bad entries
    - Does NOT pre-compute embeddings (the model will tokenize text directly)
    - Caches preprocessed data for efficiency
    """
    
    def __init__(
        self,
        min_model_comparisons: int = 20,
        identity_positive_ratio: float = 0.8,
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
        self.version = "v2"
        self.last_timer: Timer | None = None
    
    def preprocess(
        self,
        data: TrainingData,
    ) -> PreprocessedTripletEncoderData[TrainingTriplet]:
        """
        Preprocess training data by constructing triplets.
        
        Args:
            data: Raw training data
            
        Returns:
            Preprocessed data with training triplets (as raw text, not embeddings)
        """
        with Timer("preprocess_triplet_finetunable", verbosity="start+end") as timer:
            self.last_timer = timer
            
            with Timer("generate_cache_key", verbosity="start+end", parent=timer):
                cache_key = self._generate_cache_key(data)
            
            if cache_key in Jars.preprocessed_data:
                return Jars.preprocessed_data.get(cache_key)
            
            with Timer("filter", verbosity="start+end", parent=timer):
                filtered_data = self._filter_data(data)

            with Timer("make_triplets", verbosity="start+end", parent=timer):
                triplets = self._construct_triplets(filtered_data)
            
            preprocessed_data = PreprocessedTripletEncoderData(triplets=triplets)
            Jars.preprocessed_data.add(cache_key, preprocessed_data)
            
            return preprocessed_data

    def _filter_data(self, data: TrainingData) -> TrainingData:
        """
        Filter out invalid and rare model entries.
        
        Args:
            data: Raw training data
            
        Returns:
            Filtered training data
        """
        filtered_data = filter_out_rare_models(data, self.min_model_comparisons)
        filtered_data = filter_out_empty_entries(filtered_data)
        filtered_data = filter_out_both_bad(filtered_data)
        if len(filtered_data.entries) == 0:
            raise ValueError(
                "No valid training data after filtering. "
                f"All models were filtered out (min_model_comparisons={self.min_model_comparisons}). "
                "Try lowering min_model_comparisons or providing more training data."
            )
        
        return filtered_data

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

    def _make_all_examples(self, pairs: list[EvaluationEntry]) -> list["TripletFinetunableEncoderPreprocessor.ModelExample"]:
        result = []
        for pair in pairs:
            result.append(TripletFinetunableEncoderPreprocessor.ModelExample(
                prompt=pair.user_prompt,
                response=pair.model_a_response,
                model=pair.model_a
            ))
            result.append(TripletFinetunableEncoderPreprocessor.ModelExample(
                prompt=pair.user_prompt,
                response=pair.model_b_response,
                model=pair.model_b
            ))
        
        return result

    def _make_all_winning_examples(self, pairs: list[EvaluationEntry]) -> list["TripletFinetunableEncoderPreprocessor.ModelExample"]:
        result = []
        for pair in pairs:
            if pair.winner == "model_a" or pair.winner == "tie":
                result.append(TripletFinetunableEncoderPreprocessor.ModelExample(
                    prompt=pair.user_prompt,
                    response=pair.model_a_response,
                    model=pair.model_a
                ))
            if pair.winner == "model_b" or pair.winner == "tie":
                result.append(TripletFinetunableEncoderPreprocessor.ModelExample(
                    prompt=pair.user_prompt,
                    response=pair.model_b_response,
                    model=pair.model_b
                ))

        return result

    def _make_examples_by_model(self, pairs: list[EvaluationEntry]) -> dict[str, list["TripletFinetunableEncoderPreprocessor.ModelExample"]]:
        result = {}
        for pair in pairs:
            if pair.model_a not in result:
                result[pair.model_a] = []
            if pair.model_b not in result:
                result[pair.model_b] = []
            result[pair.model_a].append(TripletFinetunableEncoderPreprocessor.ModelExample(
                prompt=pair.user_prompt,
                response=pair.model_a_response,
                model=pair.model_a
            ))
            result[pair.model_b].append(TripletFinetunableEncoderPreprocessor.ModelExample(
                prompt=pair.user_prompt,
                response=pair.model_b_response,
                model=pair.model_b
            ))

        return result

    def _construct_triplets_from_win_lose_pairs(
        self, 
        pairs: list[EvaluationEntry], 
        examples_by_model: dict[str, list["TripletFinetunableEncoderPreprocessor.ModelExample"]],
        all_winning_examples: list["TripletFinetunableEncoderPreprocessor.ModelExample"]
    ) -> list[TrainingTriplet]:  
        rng = random.Random(self.seed)

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

            if rng.random() < self.identity_positive_ratio:
                first_positive_example = rng.choice(examples_by_model[first_anchor_model])
            else:
                first_positive_example = rng.choice(all_winning_examples)

            triplets.append(TrainingTriplet(
                anchor_prompt=anchor_prompt,
                anchor_response=first_anchor_response,
                anchor_model_id=first_anchor_model,
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

            second_positive_example = rng.choice(examples_by_model[second_anchor_model])
            triplets.append(TrainingTriplet(
                anchor_prompt=anchor_prompt,
                anchor_response=second_anchor_response,
                anchor_model_id=second_anchor_model,
                positive_prompt=second_positive_example.prompt,
                positive_response=second_positive_example.response,
                negative_prompt=second_negative_prompt,
                negative_response=second_negative_response
            ))

        return triplets

    def _construct_triplets_from_tie_pairs(
        self, 
        pairs: list[EvaluationEntry], 
        all_examples: list["TripletFinetunableEncoderPreprocessor.ModelExample"]
    ) -> list[TrainingTriplet]:
        triplets = []
        for pair in pairs:
            anchor_prompt = pair.user_prompt

            # 1st triplet: A as anchor, B as positive
            # Choose negative as any example
            first_anchor_response = pair.model_a_response
            first_positive_prompt = pair.user_prompt
            first_positive_response = pair.model_b_response

            negative_example = random.Random(self.seed).choice(all_examples)
            triplets.append(TrainingTriplet(
                anchor_prompt=anchor_prompt,
                anchor_response=first_anchor_response,
                anchor_model_id=pair.model_a,
                positive_prompt=first_positive_prompt,
                positive_response=first_positive_response,
                negative_prompt=negative_example.prompt,
                negative_response=negative_example.response
            ))
                
            # 2nd triplet: B as anchor, A as positive
            # Same negative as in first triplet
            second_anchor_response = pair.model_b_response
            second_positive_prompt = pair.user_prompt
            second_positive_response = pair.model_a_response

            triplets.append(TrainingTriplet(
                anchor_prompt=anchor_prompt,
                anchor_response=second_anchor_response,
                anchor_model_id=pair.model_b,
                positive_prompt=second_positive_prompt,
                positive_response=second_positive_response,
                negative_prompt=negative_example.prompt,
                negative_response=negative_example.response
            ))

        return triplets

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
        return f"triplet_finetunable_encoder/{self.version}/{params_str}-{dataset_signature}"

    @dataclass
    class ModelExample:
        prompt: str
        response: str
        model: str

