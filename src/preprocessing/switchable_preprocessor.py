"""Base class for preprocessors that support sensitivity analysis switches.

Preprocessors that inherit from SwitchablePreprocessor gain four context
managers which temporarily modify how preprocess_for_inference() behaves.
The concrete preprocessor is responsible for calling _apply_prompt_switches()
at the appropriate point inside preprocess_for_inference().
"""

from contextlib import contextmanager
from typing import Iterator
import numpy as np


class SwitchablePreprocessor:
    """Mixin base class providing sensitivity-analysis context managers.

    Subclasses must:
    1. Call super().__init__() to initialise switch state.
    2. Call self._apply_prompt_switches(emb, feat) inside
       preprocess_for_inference() after computing the raw arrays.
    """

    def __init__(self) -> None:
        self._do_shuffle_prompts: bool = False
        self._shuffle_seed: int | None = None
        self._set_emb_to_mean: bool = False
        self._set_feat_to_mean: bool = False
        self._permuted_feature_idx: int | None = None
        self._permuted_feature_seed: int | None = None

    @contextmanager
    def shuffled_prompts(self, seed: int | None = None) -> Iterator[None]:
        """Shuffle prompt embeddings and features across the batch during inference.

        Breaks the correspondence between prompt identity and model scores.
        Each call with seed=None uses a different random permutation.
        """
        self._do_shuffle_prompts = True
        self._shuffle_seed = seed
        try:
            yield
        finally:
            self._do_shuffle_prompts = False
            self._shuffle_seed = None

    @contextmanager
    def set_prompt_embedding_to_mean(self) -> Iterator[None]:
        """Replace all prompt embeddings with their batch mean during inference."""
        self._set_emb_to_mean = True
        try:
            yield
        finally:
            self._set_emb_to_mean = False

    @contextmanager
    def set_prompt_features_to_mean(self) -> Iterator[None]:
        """Replace all prompt feature vectors with their batch mean during inference."""
        self._set_feat_to_mean = True
        try:
            yield
        finally:
            self._set_feat_to_mean = False

    @contextmanager
    def permuted_feature(self, feature_idx: int, seed: int | None = None) -> Iterator[None]:
        """Permute a single prompt feature column across the batch during inference."""
        self._permuted_feature_idx = feature_idx
        self._permuted_feature_seed = seed
        try:
            yield
        finally:
            self._permuted_feature_idx = None
            self._permuted_feature_seed = None

    def _apply_prompt_switches(
        self,
        emb: np.ndarray,  # [n_prompts, embedding_dim]
        feat: np.ndarray,  # [n_prompts, n_features]
    ) -> tuple[np.ndarray, np.ndarray]:  # [n_prompts, embedding_dim], [n_prompts, n_features]
        """Apply any active switches to prompt embeddings and features.

        Must be called by the subclass inside preprocess_for_inference() after
        the raw arrays have been computed.
        """
        if self._do_shuffle_prompts:
            rng = np.random.default_rng(self._shuffle_seed)
            idx = rng.permutation(len(emb))
            emb = emb[idx]
            feat = feat[idx]

        if self._set_emb_to_mean:
            emb = np.broadcast_to(emb.mean(axis=0, keepdims=True), emb.shape).copy()

        if self._set_feat_to_mean:
            feat = np.broadcast_to(feat.mean(axis=0, keepdims=True), feat.shape).copy()

        if self._permuted_feature_idx is not None:
            rng = np.random.default_rng(self._permuted_feature_seed)
            feat = feat.copy()
            feat[:, self._permuted_feature_idx] = rng.permutation(
                feat[:, self._permuted_feature_idx]
            )

        return emb, feat
