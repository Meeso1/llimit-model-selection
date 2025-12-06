import hashlib
import random
from collections import Counter

from src.constants import PREPROCESSED_DATA_JAR_PATH
from src.data_models.data_models import TrainingData
from src.data_models.behavior_encoder_types import (
    BehaviorEncoderTrainingTriplet,
    PreprocessedBehaviorEncoderData,
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
            
            with Timer("filter_and_organize", verbosity="start+end", parent=timer):
                filtered_data = self._filter_data(data)
                organized_data = self._organize_by_model(filtered_data)
            
            with Timer("construct_triplets", verbosity="start+end", parent=timer):
                triplets = self._construct_triplets(filtered_data, organized_data)
            
            preprocessed_data = PreprocessedBehaviorEncoderData(triplets=triplets)
            self.jar.add(cache_key, preprocessed_data)
            
            return preprocessed_data
    
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
        
        return TrainingData(entries=filtered_entries)
    
    def _organize_by_model(
        self,
        data: TrainingData,
    ) -> dict[str, list[tuple[str, str]]]:
        """
        Organize (prompt, response) pairs by model.
        
        Args:
            data: Filtered training data
            
        Returns:
            Dictionary mapping model names to lists of (prompt, response) pairs
        """
        model_pairs: dict[str, list[tuple[str, str]]] = {}
        
        for entry in data.entries:
            # Add model_a's pair
            if entry.model_a not in model_pairs:
                model_pairs[entry.model_a] = []
            model_pairs[entry.model_a].append((entry.user_prompt, entry.model_a_response))
            
            # Add model_b's pair
            if entry.model_b not in model_pairs:
                model_pairs[entry.model_b] = []
            model_pairs[entry.model_b].append((entry.user_prompt, entry.model_b_response))
        
        return model_pairs
    
    def _construct_triplets(
        self,
        data: TrainingData,
        organized_data: dict[str, list[tuple[str, str]]],
    ) -> list[BehaviorEncoderTrainingTriplet]:
        """
        Construct training triplets using the strategic selection process.
        
        For each pairwise comparison:
        - Anchor: winning (prompt, response) pair
        - Negative: losing (prompt, response) pair from the same comparison
        - Positive: Either identity positive (same model, different prompt) or
                   performance positive (different model, also winning)
        
        Special cases:
        - Ties: pairs can be positives for each other
        - both_bad: pairs can serve as negatives
        
        Args:
            data: Filtered training data
            organized_data: Model-organized (prompt, response) pairs
            
        Returns:
            List of training triplets
        """
        rng = random.Random(self.seed)
        triplets: list[BehaviorEncoderTrainingTriplet] = []
        
        # Collect all winning pairs and tie/both_bad entries for positive/negative selection
        all_winning_pairs: list[tuple[str, str, str]] = []  # (prompt, response, model)
        tie_entries: list[tuple[str, str, str]] = []
        both_bad_entries: list[tuple[str, str, str]] = []
        
        for entry in data.entries:
            if entry.winner == "model_a":
                all_winning_pairs.append((entry.user_prompt, entry.model_a_response, entry.model_a))
            elif entry.winner == "model_b":
                all_winning_pairs.append((entry.user_prompt, entry.model_b_response, entry.model_b))
            elif entry.winner == "tie":
                tie_entries.append((entry.user_prompt, entry.model_a_response, entry.model_a))
                tie_entries.append((entry.user_prompt, entry.model_b_response, entry.model_b))
            elif entry.winner == "both_bad":
                both_bad_entries.append((entry.user_prompt, entry.model_a_response, entry.model_a))
                both_bad_entries.append((entry.user_prompt, entry.model_b_response, entry.model_b))
        
        # Second pass: construct triplets from entries with winners
        for entry in data.entries:
            if entry.winner not in ["model_a", "model_b"]:
                continue
            
            if entry.winner == "model_a":
                anchor_prompt = entry.user_prompt
                anchor_response = entry.model_a_response
                anchor_model = entry.model_a
                neg_prompt = entry.user_prompt
                neg_response = entry.model_b_response
            else:  # model_b
                anchor_prompt = entry.user_prompt
                anchor_response = entry.model_b_response
                anchor_model = entry.model_b
                neg_prompt = entry.user_prompt
                neg_response = entry.model_a_response
            
            # Select positive based on strategy
            use_identity = rng.random() < self.identity_positive_ratio
            
            if use_identity:
                # Identity positive: different (prompt, response) from same model as anchor
                model_pairs = organized_data.get(anchor_model, [])
                # Exclude the anchor itself
                different_pairs = [
                    (p, r) for p, r in model_pairs
                    if not (p == anchor_prompt and r == anchor_response)
                ]
                
                if different_pairs:
                    pos_prompt, pos_response = rng.choice(different_pairs)
                else:
                    # Fallback to any winning pair
                    pos_prompt, pos_response, _ = rng.choice(all_winning_pairs)
            else:
                # Performance positive: winning pair from different model
                different_model_winners = [
                    (p, r) for p, r, m in all_winning_pairs if m != anchor_model
                ]
                
                pos_prompt, pos_response = rng.choice(different_model_winners)
            
            triplets.append(BehaviorEncoderTrainingTriplet(
                anchor_prompt=anchor_prompt,
                anchor_response=anchor_response,
                positive_prompt=pos_prompt,
                positive_response=pos_response,
                negative_prompt=neg_prompt,
                negative_response=neg_response,
            ))
        
        # Construct triplets from tie pairs
        for i, (anchor_prompt, anchor_response, anchor_model) in enumerate(tie_entries):
            # Use another tie pair as positive
            other_ties = [
                (p, r) for j, (p, r, m) in enumerate(tie_entries)
                if i != j
            ]
            
            if not other_ties:
                continue
            
            pos_prompt, pos_response = rng.choice(other_ties)
            
            # Use both_bad as negative (or skip if none available)
            if both_bad_entries:
                neg_prompt, neg_response, _ = rng.choice(both_bad_entries)
            else:
                continue
            
            triplets.append(BehaviorEncoderTrainingTriplet(
                anchor_prompt=anchor_prompt,
                anchor_response=anchor_response,
                positive_prompt=pos_prompt,
                positive_response=pos_response,
                negative_prompt=neg_prompt,
                negative_response=neg_response,
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
        return f"behavior_embedding/{self.version}/{params_str}-{dataset_signature}"

