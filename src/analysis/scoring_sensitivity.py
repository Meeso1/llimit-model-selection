"""Sensitivity and interpretability analysis for scoring models.

All functions accept a trained ScoringModelBase, raw TrainingData, and a
ValidationSplit. They call model.evaluate() or model.predict() through the
standard interface; no internal model state is accessed directly.

Switch-dependent functions (everything except compute_ranking_consistency)
require model.get_preprocessor() to return a SwitchablePreprocessor.
"""

import numpy as np

from src.models.scoring.scoring_model_base import ScoringModelBase
from src.data_models.data_models import InputData, TrainingData
from src.preprocessing.scoring_feature_extraction import get_feature_descriptions
from src.preprocessing.switchable_preprocessor import SwitchablePreprocessor
from src.utils.data_split import ValidationSplit


def get_prompt_feature_names() -> list[str]:
    """Return the ordered list of prompt feature names used by all scoring preprocessors."""
    numeric_descs, boolean_descs = get_feature_descriptions()
    return [d.name for d in numeric_descs] + [d.name for d in boolean_descs]


def compute_prompt_sensitivity(
    model: ScoringModelBase,
    data: TrainingData,
    split: ValidationSplit,
    n_repeats: int = 5,
    seed: int = 42,
) -> float:
    """Measure how much the model relies on prompt content.

    Shuffles prompt embeddings and features across the batch, breaking the
    correspondence between prompt identity and model scores. Returns the val
    accuracy drop (baseline_val - shuffled_val), averaged over n_repeats
    shuffles. Larger values indicate higher prompt reliance.

    Args:
        model: Trained scoring model.
        data: Evaluation data.
        split: Train/val split definition.
        n_repeats: Number of shuffled evaluations to average.
        seed: Base seed for reproducibility.

    Returns:
        Mean val accuracy drop due to prompt shuffling.
    """
    preprocessor = _require_switchable(model)

    _, baseline_val = model.evaluate(data, split)
    rng = np.random.default_rng(seed)
    shuffled_vals: list[float] = []

    for _ in range(n_repeats):
        repeat_seed = int(rng.integers(0, 2 ** 31))
        with preprocessor.shuffled_prompts(seed=repeat_seed):
            _, val_acc = model.evaluate(data, split)
        shuffled_vals.append(val_acc)

    return float(baseline_val - np.mean(shuffled_vals))


def compute_prompt_embedding_sensitivity(
    model: ScoringModelBase,
    data: TrainingData,
    split: ValidationSplit,
) -> float:
    """Measure accuracy drop when prompt embeddings are replaced with their batch mean.

    Args:
        model: Trained scoring model.
        data: Evaluation data.
        split: Train/val split definition.

    Returns:
        Val accuracy drop (baseline_val - modified_val).
    """
    preprocessor = _require_switchable(model)

    _, baseline_val = model.evaluate(data, split)
    with preprocessor.set_prompt_embedding_to_mean():
        _, modified_val = model.evaluate(data, split)

    return float(baseline_val - modified_val)


def compute_prompt_features_sensitivity(
    model: ScoringModelBase,
    data: TrainingData,
    split: ValidationSplit,
) -> float:
    """Measure accuracy drop when prompt features are replaced with their batch mean.

    Args:
        model: Trained scoring model.
        data: Evaluation data.
        split: Train/val split definition.

    Returns:
        Val accuracy drop (baseline_val - modified_val).
    """
    preprocessor = _require_switchable(model)

    _, baseline_val = model.evaluate(data, split)
    with preprocessor.set_prompt_features_to_mean():
        _, modified_val = model.evaluate(data, split)

    return float(baseline_val - modified_val)


def compute_feature_importance(
    model: ScoringModelBase,
    data: TrainingData,
    split: ValidationSplit,
    seed: int = 42,
) -> dict[str, float]:
    """Compute permutation importance for each prompt feature.

    Each feature is permuted independently across the batch and the resulting
    val accuracy drop is recorded. A larger drop indicates higher importance.

    Args:
        model: Trained scoring model.
        data: Evaluation data.
        split: Train/val split definition.
        seed: Seed for permutation reproducibility.

    Returns:
        Dict mapping feature name to val accuracy drop (baseline_val - permuted_val).
    """
    preprocessor = _require_switchable(model)

    _, baseline_val = model.evaluate(data, split)
    feature_names = get_prompt_feature_names()
    importances: dict[str, float] = {}

    for idx, name in enumerate(feature_names):
        with preprocessor.permuted_feature(feature_idx=idx, seed=seed):
            _, permuted_val = model.evaluate(data, split)
        importances[name] = float(baseline_val - permuted_val)

    return importances


def compute_ranking_consistency(
    model: ScoringModelBase,
    data: TrainingData,
    split: ValidationSplit,
    n_splits: int = 10,
    split_fraction: float = 0.5,
    seed: int = 42,
) -> float:
    """Measure how consistently the model ranks models across prompt subsets.

    Uses the validation portion of data. Computes model rankings (by mean
    score) on random subsets of validation prompts and returns the mean
    pairwise Spearman rank correlation. Values close to 1.0 indicate stable
    rankings across different prompt samples.

    Does not require the preprocessor to implement SwitchablePreprocessor.

    Args:
        model: Trained scoring model.
        data: Evaluation data.
        split: Train/val split definition (only val portion is used).
        n_splits: Number of random subsets to evaluate.
        split_fraction: Fraction of val entries per subset.
        seed: Seed for reproducibility.

    Returns:
        Mean pairwise Spearman rank correlation in [-1, 1].
    """
    from src.utils.data_split import train_val_split

    _, val_data = train_val_split(data, split.val_fraction, split.seed)
    if val_data is None:
        return 0.0

    rng = np.random.default_rng(seed)
    valid_entries = [e for e in val_data.entries if e.winner in ("model_a", "model_b")]
    n = len(valid_entries)

    all_model_names = list(
        {e.model_a for e in valid_entries} | {e.model_b for e in valid_entries}
    )

    rankings: list[np.ndarray] = []
    for _ in range(n_splits):
        subset_size = max(1, int(n * split_fraction))
        indices = rng.choice(n, size=subset_size, replace=False)
        subset = [valid_entries[i] for i in indices]

        unique_prompts = list(dict.fromkeys(e.user_prompt for e in subset))
        output = model.predict(InputData(prompts=unique_prompts, model_names=all_model_names))
        scores = output.scores  # dict[str, np.ndarray[n_prompts]]

        mean_scores = np.array([
            scores[m].mean() if m in scores else 0.0
            for m in all_model_names
        ])
        rankings.append(mean_scores)

    correlations: list[float] = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            correlations.append(_spearman_correlation(rankings[i], rankings[j]))

    return float(np.mean(correlations)) if correlations else 0.0


def _require_switchable(model: ScoringModelBase) -> SwitchablePreprocessor:
    """Get the model's preprocessor and verify it is a SwitchablePreprocessor.

    Raises:
        NotImplementedError: If the model does not expose a preprocessor.
        TypeError: If the preprocessor does not inherit from SwitchablePreprocessor.
    """
    preprocessor = model.get_preprocessor()
    if not isinstance(preprocessor, SwitchablePreprocessor):
        raise TypeError(
            f"{type(model).__name__} preprocessor ({type(preprocessor).__name__}) "
            "does not inherit from SwitchablePreprocessor. "
            "Sensitivity analysis requires shuffled_prompts(), "
            "set_prompt_embedding_to_mean(), set_prompt_features_to_mean(), "
            "and permuted_feature() context managers."
        )
    return preprocessor


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:  # [n], [n] -> scalar
    """Compute Spearman rank correlation between two arrays."""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    if denom == 0.0:
        return 0.0
    return float(np.dot(rx, ry) / denom)
