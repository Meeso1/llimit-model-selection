"""Simple benchmark model for response length prediction using per-model OLS regression."""

from typing import Any, Self

import numpy as np

from src.data_models.data_models import InputData, TrainingData
from src.data_models.length_prediction.length_prediction_data_models import (
    LengthPredictionOutputData,
    PreprocessedLengthPredictionSample,
)
from src.models.length_prediction.length_prediction_model_base import LengthPredictionModelBase
from src.preprocessing.length_prediction_preprocessor import LengthPredictionPreprocessor
from src.preprocessing.scoring_feature_extraction import get_feature_descriptions, inverse_transform_prompt_features
from src.preprocessing.simple_scaler import SimpleScaler
from src.utils.data_split import ValidationSplit, _compute_split_indices
from src.utils.length_prediction_metrics import compute_length_prediction_metrics
from src.utils.string_encoder import StringEncoder
from src.utils.timer import Timer
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry


class SimpleLengthPredictionModel(LengthPredictionModelBase):
    """
    Benchmark length prediction model using per-model OLS linear regression.

    Features are selected by name from the 45 prompt features produced by the
    existing preprocessing pipeline (see get_feature_descriptions()). With an
    empty feature list the regression reduces to predicting the per-model mean
    response length. No model embeddings are required.

    For models not seen during training, predictions fall back to a global
    regression fitted on all training data pooled.
    """

    def __init__(
        self,
        input_features: list[str] | None = None,
        use_scaled_features: bool = True,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        min_model_comparisons: int = 20,
        run_name: str | None = None,
        print_every: int | None = None,
    ) -> None:
        super().__init__(run_name)

        self.input_features: list[str] = input_features if input_features is not None else []
        self.use_scaled_features = use_scaled_features
        self.embedding_model_name = embedding_model_name
        self.min_model_comparisons = min_model_comparisons
        self.print_every = print_every

        self.preprocessor = LengthPredictionPreprocessor(
            embedding_model_name=embedding_model_name,
            min_model_comparisons=min_model_comparisons,
        )

        self._feature_indices: list[int] = []
        self._output_scaler: SimpleScaler | None = None
        self._prompt_features_scaler: SimpleScaler | None = None
        self._model_encoder: StringEncoder | None = None
        # model_name -> (weights [n_features], bias)
        self._per_model_params: dict[str, tuple[np.ndarray, float]] = {}
        self._global_params: tuple[np.ndarray, float] | None = None
        self._history_entries: list[TrainingHistoryEntry] = []
        self.last_timer: Timer | None = None

    def get_config_for_logging(self) -> dict[str, Any]:
        return {
            "model_type": "simple_length_prediction",
            "input_features": self.input_features,
            "use_scaled_features": self.use_scaled_features,
            "embedding_model_name": self.embedding_model_name,
            "preprocessor_version": self.preprocessor.version,
            "min_model_comparisons": self.min_model_comparisons,
        }

    def train(
        self,
        data: TrainingData,
        validation_split: ValidationSplit | None = None,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        with Timer("train", verbosity="start+end") as train_timer:
            self.last_timer = train_timer

            with Timer("preprocess", verbosity="start+end", parent=train_timer):
                preprocessed = self.preprocessor.preprocess(data)

            self._output_scaler = SimpleScaler.from_state_dict(preprocessed.output_scaler_state)
            self._prompt_features_scaler = SimpleScaler.from_state_dict(preprocessed.prompt_features_scaler_state)
            self._model_encoder = preprocessed.model_encoder
            self._feature_indices = self._resolve_feature_indices(self.input_features)

            self.init_logger_if_needed()

            samples = preprocessed.samples
            val_fraction = validation_split.val_fraction if validation_split is not None else 0
            seed = validation_split.seed if validation_split is not None else 42

            if val_fraction > 0:
                train_idx, val_idx = _compute_split_indices(len(samples), val_fraction, seed)
                train_samples = [samples[i] for i in train_idx]
                val_samples = [samples[i] for i in val_idx]
            else:
                train_samples = samples
                val_samples = None

            with Timer("fit_ols", verbosity="start+end", parent=train_timer):
                self._per_model_params, self._global_params = self._fit_all_models(
                    train_samples, preprocessed.model_encoder
                )

            with Timer("compute_metrics", verbosity="start+end", parent=train_timer):
                train_preds, train_actuals = self._predict_samples(
                    train_samples, preprocessed.model_encoder
                )
                train_metrics = compute_length_prediction_metrics(
                    train_preds, train_actuals, self._output_scaler
                )

                val_metrics = None
                if val_samples is not None:
                    val_preds, val_actuals = self._predict_samples(
                        val_samples, preprocessed.model_encoder
                    )
                    val_metrics = compute_length_prediction_metrics(
                        val_preds, val_actuals, self._output_scaler
                    )

            additional_metrics: dict[str, float] = {
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
                epoch=1,
                total_loss=train_metrics["rmse"],
                val_loss=val_metrics["rmse"] if val_metrics is not None else None,
                train_accuracy=train_metrics["accuracy"],
                val_accuracy=val_metrics["accuracy"] if val_metrics is not None else None,
                additional_metrics=additional_metrics,
            )
            self._history_entries.append(entry)
            self.append_entry_to_log(entry, log_timings_from=self.last_timer)

            if self.print_every is not None:
                self._log_metrics(train_metrics, val_metrics)

        final_metrics: dict[str, Any] = {
            "train_accuracy": train_metrics["accuracy"],
            "train_mae": train_metrics["mae"],
        }
        if val_metrics is not None:
            final_metrics["val_accuracy"] = val_metrics["accuracy"]
            final_metrics["val_mae"] = val_metrics["mae"]

        self.finish_logger_if_needed(
            final_metrics=final_metrics,
            log_timings_from=self.last_timer,
        )

    def predict(self, X: InputData, batch_size: int = 32) -> LengthPredictionOutputData:
        if self._output_scaler is None or self._global_params is None or self._model_encoder is None:
            raise RuntimeError("Model not trained or loaded yet")

        encoded = self.preprocessor.preprocess_for_inference(
            prompts=X.prompts,
            model_names=X.model_names,
            model_encoder=self._model_encoder,
            scaler=self._prompt_features_scaler,
        )
        # encoded.prompt_features: [n_prompts, 45]

        prompt_features_list = [encoded.prompt_features[i] for i in range(len(X.prompts))]
        features = self._select_features(prompt_features_list)  # [n_prompts, n_selected]

        predictions: dict[str, np.ndarray] = {}
        for model_name in X.model_names:
            params = self._per_model_params.get(model_name, self._global_params)
            weights, bias = params
            preds_scaled = features @ weights + bias  # [n_prompts]
            preds_log = self._output_scaler.inverse_transform(preds_scaled)  # [n_prompts]
            predictions[model_name] = np.exp(preds_log)  # [n_prompts]

        return LengthPredictionOutputData(predictions=predictions)

    def get_history(self) -> TrainingHistory:
        return TrainingHistory.from_entries(self._history_entries)

    def get_state_dict(self) -> dict[str, Any]:
        if self._output_scaler is None or self._global_params is None or self._model_encoder is None:
            raise RuntimeError("Model not trained or loaded yet")

        return {
            "input_features": self.input_features,
            "use_scaled_features": self.use_scaled_features,
            "embedding_model_name": self.embedding_model_name,
            "min_model_comparisons": self.min_model_comparisons,
            "print_every": self.print_every,
            "feature_indices": self._feature_indices,
            "output_scaler_state": self._output_scaler.get_state_dict(),
            "prompt_features_scaler_state": self._prompt_features_scaler.get_state_dict(),
            "model_encoder_state": self._model_encoder.get_state_dict(),
            "per_model_params": {
                name: {"weights": params[0], "bias": params[1]}
                for name, params in self._per_model_params.items()
            },
            "global_params": {
                "weights": self._global_params[0],
                "bias": self._global_params[1],
            },
        }

    @classmethod
    def load_state_dict(
        cls,
        state_dict: dict[str, Any],
        instance: "SimpleLengthPredictionModel | None" = None,
    ) -> Self:
        if instance is not None:
            if not isinstance(instance, cls):
                raise TypeError(f"instance must be {cls.__name__}, got {type(instance).__name__}")
            model = instance
        else:
            model = cls(
                input_features=state_dict["input_features"],
                use_scaled_features=state_dict["use_scaled_features"],
                embedding_model_name=state_dict["embedding_model_name"],
                min_model_comparisons=state_dict["min_model_comparisons"],
                print_every=state_dict.get("print_every"),
            )

        model._feature_indices = state_dict["feature_indices"]
        model._output_scaler = SimpleScaler.from_state_dict(state_dict["output_scaler_state"])
        model._prompt_features_scaler = SimpleScaler.from_state_dict(state_dict["prompt_features_scaler_state"])
        model._model_encoder = StringEncoder.load_state_dict(state_dict["model_encoder_state"])
        model._per_model_params = {
            name: (np.asarray(p["weights"]), float(p["bias"]))
            for name, p in state_dict["per_model_params"].items()
        }
        global_p = state_dict["global_params"]
        model._global_params = (np.asarray(global_p["weights"]), float(global_p["bias"]))

        return model

    def _resolve_feature_indices(self, feature_names: list[str]) -> list[int]:
        """Map feature names to their indices in the [45] prompt_features vector."""
        numeric_descs, boolean_descs = get_feature_descriptions()
        all_descs = numeric_descs + boolean_descs
        name_to_index = {desc.name: i for i, desc in enumerate(all_descs)}

        indices = []
        for name in feature_names:
            if name not in name_to_index:
                available = [desc.name for desc in all_descs]
                raise ValueError(f"Unknown feature name: '{name}'. Available features: {available}")
            indices.append(name_to_index[name])
        return indices

    def _select_features(
        self,
        prompt_features_list: list[np.ndarray],  # list of [45]
    ) -> np.ndarray:  # [n, n_selected] or [n, 0]
        """Select (and optionally unscale) the configured feature columns."""
        if not self.use_scaled_features:
            if self._prompt_features_scaler is None:
                raise RuntimeError("prompt_features_scaler not set -- model not trained yet")
            prompt_features_list = inverse_transform_prompt_features(
                prompt_features_list, self._prompt_features_scaler
            )

        prompt_features = np.stack(prompt_features_list)  # [n, 45]

        if not self._feature_indices:
            return np.zeros((len(prompt_features_list), 0), dtype=np.float64)
        return prompt_features[:, self._feature_indices].astype(np.float64)  # [n, n_selected]

    def _fit_ols(
        self,
        X: np.ndarray,  # [n_samples, n_features] -- n_features may be 0
        y: np.ndarray,  # [n_samples]
    ) -> tuple[np.ndarray, float]:
        """Fit y = X @ w + b via least squares. Returns (weights, bias)."""
        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        X_aug = np.hstack([X.astype(np.float64), ones])  # [n_samples, n_features+1]
        result, _, _, _ = np.linalg.lstsq(X_aug, y.astype(np.float64), rcond=None)
        weights = result[:-1]  # [n_features]
        bias = float(result[-1])
        return weights, bias

    def _fit_all_models(
        self,
        samples: list[PreprocessedLengthPredictionSample],
        model_encoder: StringEncoder,
    ) -> tuple[dict[str, tuple[np.ndarray, float]], tuple[np.ndarray, float]]:
        """Fit per-model and global OLS regressions."""
        features = self._select_features(
            [s.prompt_features for s in samples]
        )  # [n_samples, n_feat]

        model_X: dict[str, list[np.ndarray]] = {}
        model_y: dict[str, list[float]] = {}
        all_X: list[np.ndarray] = []
        all_y: list[float] = []

        for i, sample in enumerate(samples):
            feat = features[i]  # [n_feat]
            for model_id, log_length in [
                (sample.model_id_a, sample.log_response_length_a),
                (sample.model_id_b, sample.log_response_length_b),
            ]:
                model_name = model_encoder.decode(model_id)
                model_X.setdefault(model_name, []).append(feat)
                model_y.setdefault(model_name, []).append(log_length)
                all_X.append(feat)
                all_y.append(log_length)

        per_model_params: dict[str, tuple[np.ndarray, float]] = {}
        for model_name in model_X:
            per_model_params[model_name] = self._fit_ols(
                np.stack(model_X[model_name]),
                np.array(model_y[model_name]),
            )

        global_params = self._fit_ols(np.stack(all_X), np.array(all_y))
        return per_model_params, global_params

    def _predict_samples(
        self,
        samples: list[PreprocessedLengthPredictionSample],
        model_encoder: StringEncoder,
    ) -> tuple[np.ndarray, np.ndarray]:  # predictions [n*2], actuals [n*2]
        """Predict on preprocessed samples, returning scaled log-length arrays."""
        features = self._select_features(
            [s.prompt_features for s in samples]
        )  # [n_samples, n_feat]

        preds: list[float] = []
        actuals: list[float] = []

        for i, sample in enumerate(samples):
            feat = features[i]
            for model_id, log_length in [
                (sample.model_id_a, sample.log_response_length_a),
                (sample.model_id_b, sample.log_response_length_b),
            ]:
                model_name = model_encoder.decode(model_id)
                params = self._per_model_params.get(model_name, self._global_params)
                weights, bias = params
                preds.append(float(feat @ weights + bias))
                actuals.append(log_length)

        return np.array(preds), np.array(actuals)

    def _log_metrics(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        if val_metrics is None:
            print(
                f"Train: rmse={train_metrics['rmse']:.4f}, "
                f"accuracy={train_metrics['accuracy']*100:.2f}%, "
                f"rel_err={train_metrics['avg_relative_error']*100:.2f}%, "
                f"mae={train_metrics['mae']:.1f}"
            )
        else:
            print(
                f"Train/Val: "
                f"rmse={train_metrics['rmse']:.4f}/{val_metrics['rmse']:.4f}, "
                f"accuracy={train_metrics['accuracy']*100:.2f}%/{val_metrics['accuracy']*100:.2f}%, "
                f"rel_err={train_metrics['avg_relative_error']*100:.2f}%/{val_metrics['avg_relative_error']*100:.2f}%, "
                f"mae={train_metrics['mae']:.1f}/{val_metrics['mae']:.1f}"
            )
