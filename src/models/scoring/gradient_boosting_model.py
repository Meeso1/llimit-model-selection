"""Gradient boosting model for prompt routing."""

from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
import os
from collections import Counter
from pydantic import TypeAdapter
import xgboost as xgb
import tempfile

from src.models.scoring.scoring_model_base import ScoringModelBase
from src.data_models.data_models import TrainingData, InputData, OutputData
from src.data_models.gradient_boosting_types import PreprocessedPromptPair, PreprocessedTrainingData, PromptRoutingOutput
from src.models.embedding_specs.embedding_spec_union import EmbeddingSpec
from src.models.embedding_models.embedding_model_base import EmbeddingModelBase
from src.preprocessing.prompt_embedding_with_categories_preprocessor import PromptEmbeddingWithCategoriesPreprocessor
from src.preprocessing.simple_scaler import SimpleScaler
from src.utils.string_encoder import StringEncoder
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.timer import Timer
from src.utils.data_split import ValidationSplit, split_gradient_boosting_preprocessed_data
from src.models.model_outputs_cache import ModelOutputsCache
from src.models import model_loading
from src.utils.best_model_tracker import BestModelTracker
import warnings


FeatureType = Literal["prompt_features", "prompt_embedding", "prompt_categories", "model_embedding"]


def margin_ranking_objective(preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
    """
    Custom objective function similar to PyTorch's MarginRankingLoss.
    
    Trains the model to predict relative scores between pairs of models.
    During inference, models are scored independently.
    
    Data structure:
        - Samples ordered as pairs: [sample_a_1, sample_b_1, sample_a_2, sample_b_2, ...]
        - Labels: 1.0 if sample at position wins, 0.0 if it loses
        - For each pair (a, b): label_a=1.0, label_b=0.0 if a wins
        
    Loss: max(0, -direction * (pred_a - pred_b) + margin)
    where direction = 1 if a wins, -1 if b wins
    
    Args:
        preds: Current predictions [n_samples]
        dtrain: Training data with labels and optional weights
        
    Returns:
        grad: Gradient vector [n_samples]
        hess: Hessian vector [n_samples]
    """
    labels = dtrain.get_label()  # [n_samples]
    weights = dtrain.get_weight()  # [n_samples] or empty array if no weights
    margin = 0.1
    
    # Ensure even number of samples (pairs)
    if len(preds) % 2 != 0:
        raise ValueError(f"Must have even number of samples (pairs), got {len(preds)}")
    
    # Reshape to pairs: [n_pairs, 2]
    preds_pairs = preds.reshape(-1, 2)  # [n_pairs, 2]
    labels_pairs = labels.reshape(-1, 2)  # [n_pairs, 2]
    
    # For each pair, extract predictions and determine direction
    pred_a = preds_pairs[:, 0]  # [n_pairs]
    pred_b = preds_pairs[:, 1]  # [n_pairs]
    
    # Label of first element in pair determines direction
    # 1.0 means a wins (direction=1), 0.0 means b wins (direction=-1)
    label_a = labels_pairs[:, 0]  # [n_pairs]
    direction = np.where(label_a == 1.0, 1.0, -1.0)  # [n_pairs]
    
    # Compute margin ranking loss components
    diff = pred_a - pred_b  # [n_pairs]
    loss_input = -direction * diff + margin  # [n_pairs]
    
    # Gradient computation (derivative of hinge loss)
    # If loss_input > 0: gradient w.r.t pred_a is -direction, w.r.t pred_b is +direction
    # If loss_input <= 0: gradient is 0
    mask = (loss_input > 0).astype(float)  # [n_pairs]
    
    grad_a = -direction * mask  # [n_pairs]
    grad_b = direction * mask   # [n_pairs]
    
    # Hessian (second derivative) - for hinge loss, it's 0 except at boundaries
    # Use small constant for numerical stability
    hess_a = np.ones_like(grad_a) * 0.01  # [n_pairs]
    hess_b = np.ones_like(grad_b) * 0.01  # [n_pairs]
    
    # Interleave gradients and hessians back to match input shape
    grad = np.empty(len(preds), dtype=float)
    hess = np.empty(len(preds), dtype=float)
    
    grad[0::2] = grad_a  # Every even index
    grad[1::2] = grad_b  # Every odd index
    hess[0::2] = hess_a
    hess[1::2] = hess_b
    
    # Apply sample weights if provided
    if len(weights) > 0:
        grad = grad * weights
        hess = hess * weights
    
    return grad, hess


def pairwise_accuracy_metric(preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]:
    """
    Custom evaluation metric: accuracy of pairwise comparisons.
    
    Measures how often the model correctly predicts which model in a pair
    should have a higher score.
    
    Args:
        preds: Current predictions [n_samples]
        dtrain: Training data with labels
        
    Returns:
        Tuple of (metric_name, accuracy_value)
    """
    labels = dtrain.get_label()  # [n_samples]
    
    # Reshape to pairs
    preds_pairs = preds.reshape(-1, 2)  # [n_pairs, 2]
    labels_pairs = labels.reshape(-1, 2)  # [n_pairs, 2]
    
    pred_a = preds_pairs[:, 0]  # [n_pairs]
    pred_b = preds_pairs[:, 1]  # [n_pairs]
    label_a = labels_pairs[:, 0]  # [n_pairs]
    
    # Predicted winner is whichever has higher score
    # True winner is label_a == 1.0
    correct = ((pred_a > pred_b) == (label_a == 1.0)).mean()
    
    return 'pairwise_acc', float(correct)


class GradientBoostingModel(ScoringModelBase):
    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        balance_model_samples: bool = True,
        input_features: list[FeatureType] = ["prompt_features", "model_embedding", "prompt_embedding"],
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_spec: EmbeddingSpec | None = None,
        load_embedding_model_from: str | None = None,
        min_model_comparisons: int = 20,
        embedding_model_epochs: int = 10,
        base_model_name: str | None = None,
        run_name: str | None = None,
        print_every: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(run_name)

        if len(input_features) == 0:
            raise ValueError("input_features must contain at least one feature type")
        
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        
        self.input_features = list(set(input_features))
        
        self.balance_model_samples = balance_model_samples
        self.embedding_model_name = embedding_model_name
        self.print_every = print_every
        self.min_model_comparisons = min_model_comparisons
        self.embedding_model_epochs = embedding_model_epochs
        self.seed = seed
        
        self.embedding_spec = embedding_spec        
        self.embedding_model: EmbeddingModelBase | None = None
        
        self.preprocessor = PromptEmbeddingWithCategoriesPreprocessor(
            embedding_model_name=embedding_model_name,
            min_model_comparisons=min_model_comparisons,
        )
        
        self._base_model_name: str | None = base_model_name
        self._model_outputs_cache: ModelOutputsCache | None = None
        
        self._embedding_model_source: str | None = load_embedding_model_from
        self._prompt_embedding_dim: int | None = None
        self._prompt_features_dim: int | None = None
        self._prompt_categories_dim: int | None = None
        self._model_embedding_dim: int | None = None
        self._history_entries: list[TrainingHistoryEntry] = []
        self._xgb_model: xgb.Booster | None = None
        self._prompt_features_scaler: SimpleScaler | None = None
        self._best_model_tracker = BestModelTracker()
        
        self.last_timer: Timer | None = None

    @property
    def model_embeddings(self) -> dict[str, np.ndarray]:
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not created. Train or load a model first.")

        return self.embedding_model.model_embeddings

    def _initialize_dimensions(
        self,
        prompt_embedding_dim: int,
        prompt_features_dim: int,
        model_embedding_dim: int,
        prompt_categories_dim: int,
    ) -> None:
        self._prompt_embedding_dim = prompt_embedding_dim
        self._prompt_features_dim = prompt_features_dim
        self._prompt_categories_dim = prompt_categories_dim
        self._model_embedding_dim = model_embedding_dim

    def get_config_for_logging(self) -> dict[str, Any]:
        """Get configuration dictionary for Weights & Biases logging."""
        return {
            "model_type": "gradient_boosting",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "balance_model_samples": self.balance_model_samples,
            "embedding_model_name": self.embedding_model_name,
            "preprocessor_version": self.preprocessor.version,
            "embedding_type": self.embedding_model.embedding_type,
            "embedding_spec": self.embedding_spec.model_dump() if self.embedding_spec is not None else None,
            "min_model_comparisons": self.min_model_comparisons,
            "embedding_model_epochs": self.embedding_model_epochs,
            "input_features": self.input_features,
            "base_model": self._base_model_name,
        }

    # TODO: Track best state
    def train(
        self,
        data: TrainingData,
        validation_split: ValidationSplit | None = None,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        """
        Train the gradient boosting model.
        
        Args:
            data: Training data with prompts and comparisons
            validation_split: Validation split configuration
            epochs: Number of boosting rounds (trees to add)
            batch_size: Used for embedding model training
        """
        if "prompt_categories" in self.input_features:
            missing_categories = sum(1 for entry in data.entries if entry.category_tag is None)
            if missing_categories > 0:
                warnings.warn(
                    f"prompt_categories in input_features but {missing_categories}/{len(data.entries)} "
                    f"({missing_categories/len(data.entries)*100:.1f}%) training entries have no category_tag. "
                    f"These will use zero vectors for categories.",
                    UserWarning
                )
        
        with Timer("train", verbosity="start+end") as train_timer:
            self.last_timer = train_timer
            
            with Timer("load_base_model", verbosity="start+end", parent=train_timer):
                self._load_base_model()
            
            with Timer("encode_prompts", verbosity="start+end", parent=train_timer):
                preprocessed_without_model_embeddings = self.preprocessor.preprocess(data)
            
            self._prompt_features_scaler = SimpleScaler.from_state_dict(preprocessed_without_model_embeddings.scaler_state)
            
            with Timer("train_embedding_model", verbosity="start+end", parent=train_timer):
                if self.embedding_model is None:
                    if self._embedding_model_source is not None:
                        self.embedding_model = model_loading.load_embedding_model_from_model(self._embedding_model_source)
                    elif self.embedding_spec is not None:
                        self.embedding_model = self.embedding_spec.create_model(
                            min_model_comparisons=self.min_model_comparisons,
                            preprocessor_seed=self.seed,
                            print_every=self.print_every,
                        )
                    else:
                        raise RuntimeError("No embedding model available and no way to create one")
                    
                if not self.embedding_model.is_initialized:
                    self.embedding_model.train(
                        data, 
                        validation_split=validation_split, 
                        epochs=self.embedding_model_epochs, 
                        batch_size=batch_size,
                    )
            
            with Timer("prepare_preprocessed_data", verbosity="start+end", parent=train_timer):
                preprocessed_pairs = [
                    PreprocessedPromptPair(
                        prompt_embedding=pair.prompt_embedding,
                        prompt_features=pair.prompt_features,
                        prompt_categories=pair.prompt_categories,
                        model_embedding_a=self.model_embeddings[preprocessed_without_model_embeddings.model_encoder.decode(pair.model_id_a)],
                        model_embedding_b=self.model_embeddings[preprocessed_without_model_embeddings.model_encoder.decode(pair.model_id_b)],
                        model_id_a=pair.model_id_a,
                        model_id_b=pair.model_id_b,
                        winner_label=pair.winner_label,
                    )
                    for pair in preprocessed_without_model_embeddings.pairs
                ]
                preprocessed_data = PreprocessedTrainingData(
                    pairs=preprocessed_pairs,
                    prompt_features_dim=preprocessed_without_model_embeddings.prompt_features_dim,
                    filtered_indexes=preprocessed_without_model_embeddings.filtered_indexes,
                    model_encoder=preprocessed_without_model_embeddings.model_encoder,
                    embedding_dim=preprocessed_without_model_embeddings.embedding_dim,
                    prompt_categories_dim=preprocessed_without_model_embeddings.prompt_categories_dim,
                    scaler_state=preprocessed_without_model_embeddings.scaler_state,
                )
            
            with Timer("cache_base_model_predictions", verbosity="start+end", parent=train_timer):
                if self._model_outputs_cache is not None:
                    self._model_outputs_cache.compute_and_cache(
                        entries=[data.entries[i] for i in preprocessed_data.filtered_indexes],
                        indexes=preprocessed_data.filtered_indexes,
                        timer=train_timer,
                    )
            
            self._initialize_dimensions(
                prompt_embedding_dim=preprocessed_data.embedding_dim,
                prompt_features_dim=preprocessed_without_model_embeddings.prompt_features_dim,
                model_embedding_dim=self.embedding_model.embedding_dim,
                prompt_categories_dim=preprocessed_without_model_embeddings.prompt_categories_dim,
            )
            
            self.init_logger_if_needed()
            
            with Timer("split_preprocessed_data", verbosity="start+end", parent=train_timer):
                preprocessed_train, preprocessed_val = split_gradient_boosting_preprocessed_data(
                    preprocessed_data,
                    val_fraction=validation_split.val_fraction if validation_split is not None else 0,
                    seed=validation_split.seed if validation_split is not None else 42,
                )
            
            with Timer("prepare_xgboost_data", verbosity="start+end", parent=train_timer):
                dtrain = self._prepare_xgboost_data(
                    preprocessed_train, 
                    use_balancing=self.balance_model_samples
                )
                dval = self._prepare_xgboost_data(preprocessed_val, use_balancing=False) if preprocessed_val is not None else None
            
            params = {
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': 1.0,  # Fixed at 1.0 to preserve pairwise structure
                'colsample_bytree': self.colsample_bytree,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'seed': self.seed,
                'disable_default_eval_metric': 1,
            }
            
            with Timer("boosting_rounds", verbosity="start+end", parent=train_timer) as rounds_timer:
                self._xgb_model = None
                for epoch in range(1, epochs + 1):
                    result = self._train_epoch(
                        epoch, 
                        params, 
                        dtrain, 
                        dval, 
                        rounds_timer,
                    )
                    
                    self._log_epoch_result(result)
                    
                    self._best_model_tracker.record_state(
                        accuracy=result.val_accuracy if result.val_accuracy is not None else result.train_accuracy,
                        state_dict=self.get_state_dict(),
                        epoch=epoch
                    )
            
            # Revert to best model parameters if available
            if self._best_model_tracker.has_best_state:
                print(f"\nReverting to best model parameters from epoch {self._best_model_tracker.best_epoch} (accuracy={self._best_model_tracker.best_accuracy:.4f})")
                self.load_state_dict(self._best_model_tracker.best_state_dict, instance=self)
            
            final_metrics = {
                "best_epoch": self._best_model_tracker.best_epoch,
                "best_accuracy": self._best_model_tracker.best_accuracy,
                "total_epochs": epochs,
            }

        self.finish_logger_if_needed(
            final_metrics=final_metrics,
            log_timings_from=self.last_timer,
        )

    def predict(
        self,
        X: InputData,
        batch_size: int = 0,
    ) -> OutputData:
        """
        Predict scores for the given prompts and models.
        
        Args:
            X: Input data with prompts and model_names
            batch_size: Batch size for prediction (not used for XGBoost)
            
        Returns:
            PromptRoutingOutput with scores for each model
        """
        if self._xgb_model is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        with Timer("predict", verbosity="start+end") as predict_timer:
            self.last_timer = predict_timer
            
            with Timer("predict_base_model", verbosity="start+end", parent=predict_timer):
                base_result = self._model_outputs_cache.predict(X, batch_size=batch_size) \
                    if self._model_outputs_cache is not None else None
            
            with Timer("preprocess_input", verbosity="start+end", parent=predict_timer):
                encoded_prompts = self.preprocessor.preprocess_for_inference(
                    prompts=X.prompts,
                    model_names=X.model_names,
                    model_encoder=StringEncoder(),  # We don't need model IDs here
                    scaler=self._prompt_features_scaler,
                )
            
            with Timer("prepare_features", verbosity="start+end", parent=predict_timer):
                # For each prompt-model combination, create feature vector
                all_features = []  # [n_prompts * n_models, feature_dim]
                prompt_indices = []  # Track which prompt each row belongs to
                model_names_flat = []  # Track which model each row is for
                
                for prompt_idx, (prompt_emb, prompt_feat, prompt_categories) in enumerate(
                    zip(encoded_prompts.prompt_embeddings, encoded_prompts.prompt_features, encoded_prompts.prompt_categories)
                ):
                    for model_name in X.model_names:
                        model_emb = self.model_embeddings[model_name]
                        # Concatenate: [prompt_embedding, prompt_features, model_embedding, categories] (missing some based on input_features)
                        features = self._create_features(prompt_emb, prompt_feat, model_emb, prompt_categories)
                        all_features.append(features)
                        prompt_indices.append(prompt_idx)
                        model_names_flat.append(model_name)
                
                X_pred = np.array(all_features)  # [n_prompts * n_models, feature_dim]
            
            with Timer("xgboost_predict", verbosity="start+end", parent=predict_timer):
                dpred = xgb.DMatrix(X_pred)
                raw_scores = self._xgb_model.predict(dpred)  # [n_prompts * n_models]
            
            with Timer("organize_scores", verbosity="start+end", parent=predict_timer):
                # Organize scores by model
                scores_dict: dict[str, list[float]] = {model: [] for model in X.model_names}
                
                for score, prompt_idx, model_name in zip(raw_scores, prompt_indices, model_names_flat):
                    scores_dict[model_name].append(score)
                
                # Convert to numpy arrays
                scores_dict_np: dict[str, np.ndarray] = {
                    model: np.array(scores)  # [n_prompts]
                    for model, scores in scores_dict.items()
                }
                
                # Add base model scores if available
                if base_result is not None:
                    accumulated = {}
                    for model_name, computed_scores in scores_dict_np.items():
                        base_scores = base_result.scores.get(model_name, np.zeros_like(computed_scores))
                        accumulated[model_name] = computed_scores + base_scores
                    scores_dict_np = accumulated
            
            return PromptRoutingOutput(_scores=scores_dict_np)

    def get_history(self) -> TrainingHistory:
        """Get training history."""
        return TrainingHistory.from_entries(self._history_entries)

    def get_state_dict(self) -> dict[str, Any]:
        """
        Get state dictionary for saving the model.
        
        Returns:
            State dictionary containing all model parameters and configuration
        """
        if self._xgb_model is None or self._prompt_features_scaler is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        if not self.embedding_model.is_initialized:
            raise RuntimeError("Embedding model not initialized")
        
        # Serialize XGBoost model to bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            tmp_path = tmp.name
        
        try:
            self._xgb_model.save_model(tmp_path)
            with open(tmp_path, 'rb') as f:
                xgb_model_bytes = f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        return {
            "balance_model_samples": self.balance_model_samples,
            "embedding_model_name": self.embedding_model_name,
            "print_every": self.print_every,
            "preprocessor_version": self.preprocessor.version,
            "prompt_embedding_dim": self._prompt_embedding_dim,
            "prompt_features_dim": self._prompt_features_dim,
            "prompt_categories_dim": self._prompt_categories_dim,
            "model_embedding_dim": self._model_embedding_dim,
            "xgb_model_bytes": xgb_model_bytes,
            "prompt_features_scaler_state": self._prompt_features_scaler.get_state_dict(),
            "embedding_type": self.embedding_model.embedding_type,
            "embedding_spec": self.embedding_spec.model_dump() if self.embedding_spec is not None else None,
            "min_model_comparisons": self.min_model_comparisons,
            "embedding_model_epochs": self.embedding_model_epochs,
            "embedding_model_state_dict": self.embedding_model.get_state_dict(),
            "seed": self.seed,
            # XGBoost hyperparameters
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            
            "input_features": self.input_features,
            
            "base_model_name": self._base_model_name,
            "base_model_state_dict": self._model_outputs_cache.model.get_state_dict() if self._model_outputs_cache is not None else None,
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any], instance: "GradientBoostingModel | None" = None) -> "GradientBoostingModel":
        """
        Load model from state dictionary.
        
        Args:
            state_dict: State dictionary from get_state_dict()
            instance: Optional existing model instance to load into
            
        Returns:
            Loaded model instance
        """        
        if instance is not None:
            if not isinstance(instance, cls):
                raise TypeError(f"instance must be of type {cls.__name__}, got {type(instance).__name__}")
            model = instance
        else:
            # Parse embedding spec using Pydantic TypeAdapter
            embedding_spec_adapter = TypeAdapter(EmbeddingSpec)
            embedding_spec = embedding_spec_adapter.validate_python(state_dict["embedding_spec"]) \
                if state_dict["embedding_spec"] is not None else None
            
            model = cls(
                balance_model_samples=state_dict["balance_model_samples"],
                embedding_model_name=state_dict["embedding_model_name"],
                embedding_spec=embedding_spec,
                min_model_comparisons=state_dict["min_model_comparisons"],
                embedding_model_epochs=state_dict["embedding_model_epochs"],
                max_depth=state_dict["max_depth"],
                learning_rate=state_dict["learning_rate"],
                colsample_bytree=state_dict["colsample_bytree"],
                reg_alpha=state_dict["reg_alpha"],
                reg_lambda=state_dict["reg_lambda"],
                base_model_name=state_dict.get("base_model_name", None),
                print_every=state_dict["print_every"],
                seed=state_dict["seed"],
                input_features=state_dict.get("input_features", ["prompt_features", "model_embedding", "prompt_embedding"]),
            )
        
        model.embedding_model = EmbeddingModelBase.load_from_state_dict(state_dict["embedding_model_state_dict"])
        model._prompt_features_scaler = SimpleScaler.from_state_dict(state_dict["prompt_features_scaler_state"])

        if model._prompt_embedding_dim is None:
            model._initialize_dimensions(
                prompt_embedding_dim=state_dict["prompt_embedding_dim"],
                prompt_features_dim=state_dict["prompt_features_dim"],
                model_embedding_dim=state_dict["model_embedding_dim"],
                prompt_categories_dim=state_dict.get("prompt_categories_dim", 0),
            )
        
        # Load XGBoost model from bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            tmp.write(state_dict["xgb_model_bytes"])
            tmp.flush()
            tmp_path = tmp.name
        
        try:
            model._xgb_model = xgb.Booster()
            model._xgb_model.load_model(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        # Load base model if present
        if model._base_model_name is not None and state_dict.get("base_model_state_dict") is not None:
            model_type, _ = model._base_model_name.split("/", 1)
            base_model = model_loading.load_scoring_model_from_state_dict(model_type, state_dict["base_model_state_dict"])
            model._model_outputs_cache = ModelOutputsCache(base_model, quiet=model.print_every is None)
        
        return model

    def _prepare_xgboost_data(
        self, 
        preprocessed_data: PreprocessedTrainingData, 
        use_balancing: bool,
    ) -> xgb.DMatrix:
        """
        Prepare data for XGBoost training with pairwise structure.
        
        For each comparison pair (model A vs model B), creates two samples:
        - Sample for model A: features=[prompt_emb, prompt_feat, model_emb_a], label=1 if A wins else 0
        - Sample for model B: features=[prompt_emb, prompt_feat, model_emb_b], label=1 if B wins else 0
        
        Samples are ordered as pairs [a_1, b_1, a_2, b_2, ...] for the custom objective function.
        
        Args:
            preprocessed_data: Preprocessed training data with comparison pairs
            use_balancing: Whether to apply sample balancing based on model frequency
            
        Returns:
            DMatrix with features [n_samples, feature_dim], labels [n_samples], and optional weights
        """
        features_list = []  # [n_samples, feature_dim]
        labels_list = []  # [n_samples]
        model_ids_list = []  # For balancing
        indexes_list = []  # Track original indexes for each pair
        
        for pair_idx, pair in zip(preprocessed_data.filtered_indexes, preprocessed_data.pairs):
            # Create feature vector for each model in the comparison
            # Format: [prompt_embedding, prompt_features, model_embedding]
            
            (features_a, features_b), (label_a, label_b) = self._create_sample_pair(pair)
            features_list.extend([features_a, features_b])
            labels_list.extend([label_a, label_b])
            model_ids_list.extend([pair.model_id_a, pair.model_id_b])
            indexes_list.append(pair_idx)
        
        X = np.array(features_list)  # [n_samples, feature_dim]
        y = np.array(labels_list)  # [n_samples]
        sample_weights = self._compute_sample_weights(model_ids_list) if use_balancing else None # [n_samples]
        
        dmatrix = xgb.DMatrix(X, label=y, weight=sample_weights) if sample_weights is not None else xgb.DMatrix(X, label=y)
        
        # Set base margins (base model scores) if available
        if self._model_outputs_cache is not None:
            base_margins = self._get_base_margins(indexes_list)
            dmatrix.set_base_margin(base_margins)
        
        return dmatrix
    
    def _create_sample_pair(self, pair: PreprocessedPromptPair) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        # Sample 1: Model A
        features_a = self._create_features(pair.prompt_embedding, pair.prompt_features, pair.model_embedding_a, pair.prompt_categories)
        label_a = 1.0 if pair.winner_label == 0 else 0.0
        
        # Sample 2: Model B
        features_b = self._create_features(pair.prompt_embedding, pair.prompt_features, pair.model_embedding_b, pair.prompt_categories)
        label_b = 1.0 if pair.winner_label == 1 else 0.0
        
        return (features_a, features_b), (label_a, label_b)
    
    def _create_features(
        self, 
        prompt_embedding: np.ndarray, 
        prompt_features: np.ndarray, 
        model_embedding: np.ndarray,
        prompt_categories: np.ndarray,
    ) -> np.ndarray:
        result = []
        
        if "prompt_features" in self.input_features:
            result.append(prompt_features)
        if "model_embedding" in self.input_features:
            result.append(model_embedding)
        if "prompt_embedding" in self.input_features:
            result.append(prompt_embedding)
        if "prompt_categories" in self.input_features:
            result.append(prompt_categories)
            
        if len(result) == 0:
            raise ValueError("No input features specified")
        
        return np.concatenate(result)

    def _compute_sample_weights(
        self,
        model_ids: list[int],
    ) -> np.ndarray:
        """
        Compute sample weights to balance model representation.
        
        Args:
            model_ids: List of model IDs  # [n_samples]
            
        Returns:
            Sample weights [n_samples]
        """
        # Count how many times each model appears
        model_counts = Counter(model_ids)
        
        # Compute weight for each model (inverse frequency)
        model_weights = {
            model_id: 1.0 / count
            for model_id, count in model_counts.items()
        }
        
        # Assign weight to each sample
        sample_weights = np.array([model_weights[model_id] for model_id in model_ids])
        
        # Normalize weights to have mean of 1.0
        sample_weights = sample_weights / sample_weights.mean()
        
        return sample_weights

    def _compute_margin_loss(self, preds: np.ndarray, dmatrix: xgb.DMatrix) -> float:
        """
        Compute margin ranking loss for logging.
        
        Args:
            preds: Predictions [n_samples]
            dmatrix: Data matrix with labels
            
        Returns:
            Average margin ranking loss
        """
        labels = dmatrix.get_label()  # [n_samples]
        margin = 0.1
        
        # Reshape to pairs
        preds_pairs = preds.reshape(-1, 2)  # [n_pairs, 2]
        labels_pairs = labels.reshape(-1, 2)  # [n_pairs, 2]
        
        pred_a = preds_pairs[:, 0]  # [n_pairs]
        pred_b = preds_pairs[:, 1]  # [n_pairs]
        label_a = labels_pairs[:, 0]  # [n_pairs]
        
        # Compute direction: 1 if a wins, -1 if b wins
        direction = np.where(label_a == 1.0, 1.0, -1.0)  # [n_pairs]
        
        # Margin ranking loss: max(0, -direction * (pred_a - pred_b) + margin)
        diff = pred_a - pred_b  # [n_pairs]
        loss = np.maximum(0, -direction * diff + margin)  # [n_pairs]
        
        return float(loss.mean())

    def _train_epoch(
        self, 
        epoch: int,
        params: dict[str, Any],
        dtrain: xgb.DMatrix,
        dval: xgb.DMatrix | None,
        rounds_timer: Timer,
    ) -> "GradientBoostingModel.EpochResult":
        """
        Train one boosting round (add one tree).
        
        Args:
            epoch: Current epoch/round number
            params: XGBoost parameters
            dtrain: Training data
            dval: Validation data (optional)
            rounds_timer: Parent timer for all rounds
            
        Returns:
            EpochResult with metrics
        """
        with Timer(f"round_{epoch}", verbosity="start+end", parent=rounds_timer) as timer:
            # Train one more tree
            evals_result: dict[str, dict[str, list[float]]] = {}
            evals = [(dtrain, 'train')]
            if dval is not None:
                evals.append((dval, 'val'))
            
            self._xgb_model = xgb.train(
                params,
                dtrain,
                num_boost_round=1,  # Add just one tree
                xgb_model=self._xgb_model,  # Continue from previous model
                obj=margin_ranking_objective,  # Custom objective
                custom_metric=pairwise_accuracy_metric,  # Custom metric
                evals=evals,
                evals_result=evals_result,
                verbose_eval=False,
            )
            
            # Get accuracy from custom metric
            train_accuracy = evals_result['train']['pairwise_acc'][0]
            
            # Compute margin ranking loss manually for logging
            # Note: XGBoost predictions already include base margins
            train_pred = self._xgb_model.predict(dtrain)  # [n_train_samples]
            train_loss = self._compute_margin_loss(train_pred, dtrain)
            
            # Validation metrics
            val_loss = None
            val_accuracy = None
            if dval is not None:
                val_accuracy = evals_result['val']['pairwise_acc'][0]
                val_pred = self._xgb_model.predict(dval)  # [n_val_samples]
                val_loss = self._compute_margin_loss(val_pred, dval)
            
            entry = TrainingHistoryEntry(
                epoch=epoch,
                total_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
            )
            self._history_entries.append(entry)
            
            self.append_entry_to_log(entry, log_timings_from=self.last_timer)
        
        return self.EpochResult(
            epoch=epoch,
            total_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            duration=timer.elapsed_time,
        )
    
    def _load_base_model(self) -> None:
        """Load the base model from the specification string."""
        if self._base_model_name is None or self._model_outputs_cache is not None:
            return
        
        if "/" not in self._base_model_name:
            raise ValueError("Base model must be of the form 'model_type/model_name'")
                
        model_type, model_name = self._base_model_name.split("/", 1)
        
        if self.print_every is not None:
            print(f"Loading base model: {self._base_model_name}")
        
        loaded_model = model_loading.load_scoring_model(model_type, model_name)
        self._model_outputs_cache = ModelOutputsCache(
            model=loaded_model,
            quiet=self.print_every is None,
        )
        
        if self.print_every is not None:
            print(f"Base model loaded")
    
    def _get_base_margins(
        self,
        indexes: list[int],
    ) -> np.ndarray:
        """
        Get base model scores as margins for XGBoost.
        
        Args:
            indexes: Original data indexes for each pair [n_pairs]
            
        Returns:
            Base margins [n_samples] where samples are ordered as pairs [a_1, b_1, a_2, b_2, ...]
        """
        if self._model_outputs_cache is None:
            raise RuntimeError("Model outputs cache not initialized")
        
        # Get base scores for all pairs
        base_scores_a, base_scores_b = self._model_outputs_cache.get_base_scores(indexes)
        
        # Convert to numpy array and interleave [a_1, b_1, a_2, b_2, ...]
        base_margins = np.empty(len(indexes) * 2, dtype=np.float32)
        base_margins[0::2] = base_scores_a  # Every even index
        base_margins[1::2] = base_scores_b  # Every odd index
        
        return base_margins

    def _log_epoch_result(self, result: "GradientBoostingModel.EpochResult") -> None:
        if self.print_every is None:
            return
        
        if not result.epoch % self.print_every == 0:
            return
        
        if result.val_loss is None or result.val_accuracy is None:
            print(f"Round {result.epoch:>4}: loss = {result.total_loss:.4f}, accuracy = {(result.train_accuracy*100):.2f}% - {result.duration:.2f}s")
        else:
            print(f"Round {result.epoch:>4}: loss = {result.total_loss:.4f}/{result.val_loss:.4f}, accuracy = {(result.train_accuracy*100):.2f}%/{(result.val_accuracy*100):.2f}% - {result.duration:.2f}s")

    @dataclass
    class EpochResult:
        epoch: int
        total_loss: float
        val_loss: float | None
        train_accuracy: float
        val_accuracy: float | None
        duration: float
