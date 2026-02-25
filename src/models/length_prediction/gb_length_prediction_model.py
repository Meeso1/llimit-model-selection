"""Gradient boosting model for response length prediction."""

from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
import os
from pydantic import TypeAdapter
import xgboost as xgb
import tempfile

from src.models.length_prediction.length_prediction_model_base import LengthPredictionModelBase
from src.data_models.data_models import TrainingData, InputData
from src.data_models.length_prediction.length_prediction_data_models import (
    LengthPredictionOutputData,
    PreprocessedLengthPredictionTrainingDataWithEmbeddings,
)
from src.models.embedding_specs.embedding_spec_union import EmbeddingSpec
from src.models.embedding_models.embedding_model_base import EmbeddingModelBase
from src.preprocessing.length_prediction_preprocessor import LengthPredictionPreprocessor
from src.preprocessing.simple_scaler import SimpleScaler
from src.utils.string_encoder import StringEncoder
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.timer import Timer
from src.utils.data_split import ValidationSplit, split_length_prediction_preprocessed_data
from src.utils.best_model_tracker import BestModelTracker
from src.utils.length_prediction_metrics import compute_length_prediction_metrics
from src.models import model_loading


FeatureType = Literal["prompt_features", "prompt_embedding", "model_embedding"]


class GbLengthPredictionModel(LengthPredictionModelBase):
    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        colsample_bytree: float = 1.0,
        colsample_bylevel: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        input_features: list[FeatureType] = ["prompt_features", "model_embedding", "prompt_embedding"],
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_spec: EmbeddingSpec | None = None,
        load_embedding_model_from: str | None = None,
        min_model_comparisons: int = 20,
        embedding_model_epochs: int = 10,
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
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        
        self.input_features = list(set(input_features))
        
        self.embedding_model_name = embedding_model_name
        self.print_every = print_every
        self.min_model_comparisons = min_model_comparisons
        self.embedding_model_epochs = embedding_model_epochs
        self.seed = seed
        
        self.embedding_spec = embedding_spec        
        self.embedding_model: EmbeddingModelBase | None = None
        
        self.preprocessor = LengthPredictionPreprocessor(
            embedding_model_name=embedding_model_name,
            min_model_comparisons=min_model_comparisons,
        )
        
        self._embedding_model_source: str | None = load_embedding_model_from
        self._prompt_embedding_dim: int | None = None
        self._prompt_features_dim: int | None = None
        self._model_embedding_dim: int | None = None
        self._scaler: SimpleScaler | None = None
        self._prompt_features_scaler: SimpleScaler | None = None
        self._history_entries: list[TrainingHistoryEntry] = []
        self._xgb_model: xgb.Booster | None = None
        self._best_model_tracker = BestModelTracker()
        self._model_avg_lengths: dict[str, float] = {}
        
        self.last_timer: Timer | None = None

    @property
    def model_embeddings(self) -> dict[str, np.ndarray]:
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not created. Train or load a model first.")

        return self.embedding_model.model_embeddings

    @property
    def scaler(self) -> SimpleScaler:
        if self._scaler is None:
            raise RuntimeError("Scaler not initialized. Train or load a model first.")
        return self._scaler

    def _initialize_dimensions(
        self,
        prompt_embedding_dim: int,
        prompt_features_dim: int,
        model_embedding_dim: int,
    ) -> None:
        self._prompt_embedding_dim = prompt_embedding_dim
        self._prompt_features_dim = prompt_features_dim
        self._model_embedding_dim = model_embedding_dim

    def get_config_for_logging(self) -> dict[str, Any]:
        """Get configuration dictionary for Weights & Biases logging."""
        return {
            "model_type": "gb_length_prediction",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "embedding_model_name": self.embedding_model_name,
            "preprocessor_version": self.preprocessor.version,
            "embedding_type": self.embedding_model.embedding_type,
            "embedding_spec": self.embedding_spec.model_dump() if self.embedding_spec is not None else None,
            "min_model_comparisons": self.min_model_comparisons,
            "embedding_model_epochs": self.embedding_model_epochs,
            "input_features": self.input_features,
        }

    def train(
        self,
        data: TrainingData,
        validation_split: ValidationSplit | None = None,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        """
        Train the gradient boosting model for length prediction.
        
        Args:
            data: Training data with prompts and comparisons
            validation_split: Validation split configuration
            epochs: Number of boosting rounds (trees to add)
            batch_size: Used for embedding model training
        """
        with Timer("train", verbosity="start+end") as train_timer:
            self.last_timer = train_timer
            
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
            
            self.init_logger_if_needed() # Must be called after embedding model is initialized
            
            with Timer("encode_prompts", verbosity="start+end", parent=train_timer):
                preprocessed_without_embeddings = self.preprocessor.preprocess(data)
            
            with Timer("prepare_preprocessed_data", verbosity="start+end", parent=train_timer):
                preprocessed_data = preprocessed_without_embeddings.add_model_embeddings(
                    self.model_embeddings,
                    self.embedding_model.embedding_dim,
                )
            
            self._scaler = SimpleScaler.from_state_dict(preprocessed_data.output_scaler_state)
            self._prompt_features_scaler = SimpleScaler.from_state_dict(preprocessed_data.prompt_features_scaler_state)
            
            self._initialize_dimensions(
                prompt_embedding_dim=preprocessed_data.embedding_dim,
                prompt_features_dim=preprocessed_data.prompt_features_dim,
                model_embedding_dim=self.embedding_model.embedding_dim,
            )
            
            with Timer("split_preprocessed_data", verbosity="start+end", parent=train_timer):
                preprocessed_train, preprocessed_val = split_length_prediction_preprocessed_data(
                    preprocessed_data,
                    val_fraction=validation_split.val_fraction if validation_split is not None else 0,
                    seed=validation_split.seed if validation_split is not None else 42,
                )
            
            self._model_avg_lengths = self._compute_model_avg_lengths(preprocessed_train)
            
            with Timer("prepare_xgboost_data", verbosity="start+end", parent=train_timer):
                dtrain = self._prepare_xgboost_data(preprocessed_train, self._model_avg_lengths)
                dval = self._prepare_xgboost_data(preprocessed_val, self._model_avg_lengths) if preprocessed_val is not None else None
            
            params = {
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': 1.0,
                'colsample_bytree': self.colsample_bytree,
                'colsample_bylevel': self.colsample_bylevel,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'seed': self.seed,
                'objective': 'reg:squarederror',  # MSE for regression
                'eval_metric': 'rmse',
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
                    
                    # Track best model based on validation accuracy (or train if no validation)
                    accuracy_to_track = result.val_accuracy if result.val_accuracy is not None else result.train_accuracy
                    
                    self._best_model_tracker.record_state(
                        accuracy=accuracy_to_track,
                        state_dict=self.get_state_dict(),
                        epoch=epoch,
                    )
            
            # Revert to best model parameters if available
            if self._best_model_tracker.has_best_state:
                accuracy_type = "val" if validation_split is not None and validation_split.val_fraction > 0 else "train"
                if self.print_every is not None:
                    print(f"\nReverting to best model parameters from epoch {self._best_model_tracker.best_epoch} "
                          f"({accuracy_type}_accuracy={self._best_model_tracker.best_accuracy:.4f})")
                
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
    ) -> LengthPredictionOutputData:
        """
        Predict response lengths for the given prompts and models.
        
        Args:
            X: Input data with prompts and model_names
            batch_size: Batch size for prediction (not used for XGBoost)
            
        Returns:
            LengthPredictionOutputData with predicted lengths for each model
        """
        if self._xgb_model is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        with Timer("predict", verbosity="start+end") as predict_timer:
            self.last_timer = predict_timer
            
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
                
                for prompt_idx, (prompt_emb, prompt_feat) in enumerate(
                    zip(encoded_prompts.prompt_embeddings, encoded_prompts.prompt_features)
                ):
                    for model_name in X.model_names:
                        model_emb = self.model_embeddings[model_name]
                        # Concatenate based on input_features
                        features = self._create_features(prompt_emb, prompt_feat, model_emb)
                        all_features.append(features)
                        prompt_indices.append(prompt_idx)
                        model_names_flat.append(model_name)
                
                X_pred = np.array(all_features)  # [n_prompts * n_models, feature_dim]
            
            with Timer("xgboost_predict", verbosity="start+end", parent=predict_timer):
                dpred = xgb.DMatrix(X_pred)
                raw_predictions = self._xgb_model.predict(dpred)  # [n_prompts * n_models]
            
            with Timer("organize_predictions", verbosity="start+end", parent=predict_timer):
                # Organize predictions by model
                predictions_dict: dict[str, list[float]] = {model: [] for model in X.model_names}
                
                for pred, prompt_idx, model_name in zip(raw_predictions, prompt_indices, model_names_flat):
                    predictions_dict[model_name].append(pred)
                
                # Add back per-model average, then convert scaled log-lengths to raw lengths
                predictions_dict_np: dict[str, np.ndarray] = {}
                for model, preds in predictions_dict.items():
                    preds_scaled = np.array(preds) + self._model_avg_lengths.get(model, 0.0)  # [n_prompts]
                    preds_log = self.scaler.inverse_transform(preds_scaled)  # [n_prompts]
                    predictions_dict_np[model] = np.exp(preds_log)  # [n_prompts]
            
            return LengthPredictionOutputData(predictions=predictions_dict_np)

    def get_history(self) -> TrainingHistory:
        """Get training history."""
        return TrainingHistory.from_entries(self._history_entries)

    def get_state_dict(self) -> dict[str, Any]:
        """
        Get state dictionary for saving the model.
        
        Returns:
            State dictionary containing all model parameters and configuration
        """
        if self._xgb_model is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        if not self.embedding_model.is_initialized:
            raise RuntimeError("Embedding model not initialized")
        
        # Serialize XGBoost model to bytes
        xgb_model_bytes = self._serialize_xgb_model(self._xgb_model)
        
        return {
            "embedding_model_name": self.embedding_model_name,
            "print_every": self.print_every,
            "preprocessor_version": self.preprocessor.version,
            "prompt_embedding_dim": self._prompt_embedding_dim,
            "prompt_features_dim": self._prompt_features_dim,
            "model_embedding_dim": self._model_embedding_dim,
            "scaler_state": self.scaler.get_state_dict(),
            "prompt_features_scaler_state": self._prompt_features_scaler.get_state_dict(),
            "xgb_model_bytes": xgb_model_bytes,
            "embedding_type": self.embedding_model.embedding_type,
            "embedding_spec": self.embedding_spec.model_dump() if self.embedding_spec is not None else None,
            "min_model_comparisons": self.min_model_comparisons,
            "embedding_model_epochs": self.embedding_model_epochs,
            "embedding_model_state_dict": self.embedding_model.get_state_dict(),
            "seed": self.seed,
            "model_avg_lengths": self._model_avg_lengths,
            # XGBoost hyperparameters
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "input_features": self.input_features,
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any], instance: "GbLengthPredictionModel | None" = None) -> "GbLengthPredictionModel":
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
            embedding_spec = embedding_spec_adapter.validate_python(state_dict["embedding_spec"]) if state_dict["embedding_spec"] is not None else None
            
            model = cls(
                embedding_model_name=state_dict["embedding_model_name"],
                embedding_spec=embedding_spec,
                min_model_comparisons=state_dict["min_model_comparisons"],
                embedding_model_epochs=state_dict["embedding_model_epochs"],
                max_depth=state_dict["max_depth"],
                learning_rate=state_dict["learning_rate"],
                colsample_bytree=state_dict["colsample_bytree"],
                colsample_bylevel=state_dict["colsample_bylevel"],
                reg_alpha=state_dict["reg_alpha"],
                reg_lambda=state_dict["reg_lambda"],
                print_every=state_dict["print_every"],
                seed=state_dict["seed"],
                input_features=state_dict.get("input_features", ["prompt_features", "model_embedding", "prompt_embedding"]),
            )
        
        model.embedding_model = EmbeddingModelBase.load_from_state_dict(state_dict["embedding_model_state_dict"])

        if model._prompt_embedding_dim is None:
            model._initialize_dimensions(
                prompt_embedding_dim=state_dict["prompt_embedding_dim"],
                prompt_features_dim=state_dict["prompt_features_dim"],
                model_embedding_dim=state_dict["model_embedding_dim"],
            )
        
        model._scaler = SimpleScaler.from_state_dict(state_dict["scaler_state"])
        model._prompt_features_scaler = SimpleScaler.from_state_dict(state_dict["prompt_features_scaler_state"])
        
        # Load XGBoost model from bytes
        model._xgb_model = model._deserialize_xgb_model(state_dict["xgb_model_bytes"])
        model._model_avg_lengths = state_dict.get("model_avg_lengths", {})
        
        return model

    def _compute_model_avg_lengths(
        self,
        preprocessed_data: PreprocessedLengthPredictionTrainingDataWithEmbeddings,
    ) -> dict[str, float]:
        """
        Compute per-model average scaled log-length from training data.

        The model will learn to predict residuals from these averages, encoding
        the per-model verbosity baseline separately from the prompt-driven deviation.

        Args:
            preprocessed_data: Preprocessed training data (training split only)

        Returns:
            Dict mapping model name to its mean scaled log-length across all training samples
        """
        model_lengths: dict[str, list[float]] = {}
        for sample in preprocessed_data.samples:
            name_a = preprocessed_data.model_encoder.decode(sample.model_id_a)
            name_b = preprocessed_data.model_encoder.decode(sample.model_id_b)
            model_lengths.setdefault(name_a, []).append(sample.log_response_length_a)
            model_lengths.setdefault(name_b, []).append(sample.log_response_length_b)
        return {name: float(np.mean(lengths)) for name, lengths in model_lengths.items()}

    def _prepare_xgboost_data(
        self, 
        preprocessed_data: PreprocessedLengthPredictionTrainingDataWithEmbeddings,
        model_avg_lengths: dict[str, float],
    ) -> xgb.DMatrix:
        """
        Prepare data for XGBoost training.
        
        Args:
            preprocessed_data: Preprocessed training data
            model_avg_lengths: Per-model average scaled log-length (training-split averages);
                               labels are stored as residuals relative to these averages
            
        Returns:
            DMatrix with features [n_samples, feature_dim] and labels [n_samples]
        """
        features_list = []  # [n_samples, feature_dim]
        lengths_list = []  # [n_samples]
        
        for sample in preprocessed_data.samples:
            name_a = preprocessed_data.model_encoder.decode(sample.model_id_a)
            name_b = preprocessed_data.model_encoder.decode(sample.model_id_b)
            avg_a = model_avg_lengths.get(name_a, 0.0)
            avg_b = model_avg_lengths.get(name_b, 0.0)

            # Create feature vector for model A (label is residual from per-model average)
            features_a = self._create_features(
                sample.prompt_embedding, 
                sample.prompt_features, 
                sample.model_embedding_a
            )
            features_list.append(features_a)
            lengths_list.append(sample.log_response_length_a - avg_a)
            
            # Create feature vector for model B (label is residual from per-model average)
            features_b = self._create_features(
                sample.prompt_embedding, 
                sample.prompt_features, 
                sample.model_embedding_b
            )
            features_list.append(features_b)
            lengths_list.append(sample.log_response_length_b - avg_b)
        
        X = np.array(features_list)  # [n_samples, feature_dim]
        y = np.array(lengths_list)  # [n_samples]
        
        return xgb.DMatrix(X, label=y)
    
    def _create_features(
        self, 
        prompt_embedding: np.ndarray, 
        prompt_features: np.ndarray, 
        model_embedding: np.ndarray,
    ) -> np.ndarray:
        result = []
        
        if "prompt_features" in self.input_features:
            result.append(prompt_features)
        if "model_embedding" in self.input_features:
            result.append(model_embedding)
        if "prompt_embedding" in self.input_features:
            result.append(prompt_embedding)
            
        if len(result) == 0:
            raise ValueError("No input features specified")
        
        return np.concatenate(result)

    def _serialize_xgb_model(self, model: xgb.Booster) -> bytes:
        """
        Serialize XGBoost model to bytes.
        
        Args:
            model: XGBoost Booster to serialize
            
        Returns:
            Serialized model as bytes
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            tmp_path = tmp.name
        
        try:
            model.save_model(tmp_path)
            with open(tmp_path, 'rb') as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _deserialize_xgb_model(self, model_bytes: bytes) -> xgb.Booster:
        """
        Deserialize XGBoost model from bytes.
        
        Args:
            model_bytes: Serialized model bytes
            
        Returns:
            Loaded XGBoost Booster
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            tmp.write(model_bytes)
            tmp.flush()
            tmp_path = tmp.name
        
        try:
            model = xgb.Booster()
            model.load_model(tmp_path)
            return model
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _train_epoch(
        self, 
        epoch: int,
        params: dict[str, Any],
        dtrain: xgb.DMatrix,
        dval: xgb.DMatrix | None,
        rounds_timer: Timer,
    ) -> "GbLengthPredictionModel.EpochResult":
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
                evals=evals,
                evals_result=evals_result,
                verbose_eval=False,
            )
            
            # Get predictions for metrics
            train_pred = self._xgb_model.predict(dtrain)  # [n_train_samples]
            train_labels = dtrain.get_label()  # [n_train_samples]
            train_metrics = compute_length_prediction_metrics(train_pred, train_labels, self.scaler)
            
            # Validation metrics
            val_metrics = None
            if dval is not None:
                val_pred = self._xgb_model.predict(dval)  # [n_val_samples]
                val_labels = dval.get_label()  # [n_val_samples]
                val_metrics = compute_length_prediction_metrics(val_pred, val_labels, self.scaler)
            
            # Prepare additional metrics for logging
            additional_metrics = {
                "train_avg_relative_error": train_metrics["avg_relative_error"],
                "train_avg_relative_ratio": train_metrics["avg_relative_ratio"],
                "train_stddev_ratio": train_metrics["stddev_ratio"],
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
            }
            if val_metrics is not None:
                additional_metrics.update({
                    "val_avg_relative_error": val_metrics["avg_relative_error"],
                    "val_avg_relative_ratio": val_metrics["avg_relative_ratio"],
                    "val_stddev_ratio": val_metrics["stddev_ratio"],
                    "val_rmse": val_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                })
            
            entry = TrainingHistoryEntry(
                epoch=epoch,
                total_loss=train_metrics["rmse"],
                val_loss=val_metrics["rmse"] if val_metrics is not None else None,
                train_accuracy=train_metrics["accuracy"],
                val_accuracy=val_metrics["accuracy"] if val_metrics is not None else None,
                additional_metrics=additional_metrics,
            )
            self._history_entries.append(entry)
            
            self.append_entry_to_log(entry, log_timings_from=self.last_timer)
        
        return self.EpochResult(
            epoch=epoch,
            train_rmse=train_metrics["rmse"],
            val_rmse=val_metrics["rmse"] if val_metrics is not None else None,
            train_accuracy=train_metrics["accuracy"],
            val_accuracy=val_metrics["accuracy"] if val_metrics is not None else None,
            train_relative_error=train_metrics["avg_relative_error"],
            val_relative_error=val_metrics["avg_relative_error"] if val_metrics is not None else None,
            train_mae=train_metrics["mae"],
            val_mae=val_metrics["mae"] if val_metrics is not None else None,
            duration=timer.elapsed_time,
        )

    def _log_epoch_result(self, result: "GbLengthPredictionModel.EpochResult") -> None:
        if self.print_every is None:
            return
        
        if not result.epoch % self.print_every == 0:
            return
        
        if result.val_rmse is None or result.val_accuracy is None or result.val_relative_error is None or result.val_rmse is None:
            print(f"Round {result.epoch:>4}: rmse = {result.train_rmse:.4f}, "
                  f"accuracy = {(result.train_accuracy*100):.2f}%, "
                  f"rel_err = {(result.train_relative_error*100):.2f}%, "
                  f"mae = {result.train_mae:.1f} - {result.duration:.2f}s")
        else:
            print(f"Round {result.epoch:>4}: rmse = {result.train_rmse:.4f}/{result.val_rmse:.4f}, "
                  f"accuracy = {(result.train_accuracy*100):.2f}%/{(result.val_accuracy*100):.2f}%, "
                  f"rel_err = {(result.train_relative_error*100):.2f}%/{(result.val_relative_error*100):.2f}%, "
                  f"mae = {result.train_mae:.1f}/{result.val_mae:.1f} - {result.duration:.2f}s")

    @dataclass
    class EpochResult:
        epoch: int
        train_rmse: float
        val_rmse: float | None
        train_accuracy: float
        val_accuracy: float | None
        train_relative_error: float
        val_relative_error: float | None
        train_mae: float
        val_mae: float | None
        duration: float
