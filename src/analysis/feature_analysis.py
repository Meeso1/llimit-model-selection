import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

from src.preprocessing.simple_scaler import SimpleScaler


_LOG_CLIP_THRESHOLD = 1e-6


def _compute_scaled_final(
    feature_values: np.ndarray,  # [n_samples]
    logarythmic: bool,
    many_zeros: bool,
) -> np.ndarray:  # [n_nonzero] or [n_samples]
    """
    Replicate the preprocessing pipeline (log → scale → softplus) on non-zero values.
    Mirrors extract_and_transform_all_prompt_features in scoring_feature_extraction.py.
    Zero values are excluded (they become -1 in the model and are not informative here).
    Returns the final values for non-zero inputs only.
    """
    non_zero_mask = feature_values != 0  # [n_samples]
    values = feature_values.copy().astype(float)  # [n_samples]

    if logarythmic:
        clipped = values <= _LOG_CLIP_THRESHOLD
        values = np.log(np.where(~clipped, values, _LOG_CLIP_THRESHOLD))  # [n_samples]

    # Scale is fitted on non-zero values when many_zeros, otherwise on all values
    to_scale = values[non_zero_mask] if many_zeros else values  # [n_nonzero] or [n_samples]
    scaled = SimpleScaler().fit_unbalanced([to_scale]).transform_unbalanced([to_scale])[0]  # [n_nonzero] or [n_samples]

    return np.log(1 + np.exp(scaled)) if many_zeros else scaled  # [n_nonzero] or [n_samples]


def display_numeric_feature(
    feature_values: np.ndarray,  # [n_samples]
    target_values: np.ndarray,  # [n_samples]
    feature_name: str,
    logarythmic: bool = False,
    many_zeros: bool = False,
    axes: tuple[plt.Axes, plt.Axes, plt.Axes] | None = None,
) -> None:
    assert len(feature_values) == len(target_values)

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(25, 5))

    # Drop entries where target is 0
    indexes = np.where(target_values > 0)[0]
    feature_values = feature_values[indexes]
    target_values = target_values[indexes]

    sorted_raw = np.sort(feature_values)  # [n_samples]

    mode = max(np.unique(feature_values), key=lambda x: np.sum(feature_values == x))
    mode_count = np.sum(feature_values == mode)
    mean = np.mean(feature_values)
    std = np.std(feature_values)

    print(f"Mode: {mode}, count: {mode_count} ({mode_count / len(feature_values) * 100:.2f}%)")
    print(f"Mean: {mean:.2f}, std: {std:.2f}")
    print(f"\tin +/- 1 std range: {np.sum(np.abs(feature_values - mean) <= std) / len(feature_values) * 100:.2f}%")

    if many_zeros:
        zero_pct = np.sum(feature_values == 0) / len(feature_values) * 100
        print(f"Zero values: {zero_pct:.2f}%")

    if logarythmic:
        clipped_mask = feature_values <= _LOG_CLIP_THRESHOLD  # [n_samples]
        clipped_nonzero_mask = clipped_mask & (feature_values != 0) if many_zeros else clipped_mask  # [n_samples]
        clip_pct = np.sum(clipped_nonzero_mask) / len(feature_values) * 100
        print(f"Clipped when log-transforming (value <= {_LOG_CLIP_THRESHOLD}): {clip_pct:.2f}%")
        transformed_values = np.log(np.where(~clipped_mask, feature_values, _LOG_CLIP_THRESHOLD))  # [n_samples]
        transform_label = f"log({feature_name})"
    else:
        transformed_values = feature_values  # [n_samples]
        transform_label = feature_name

    # When many_zeros, show only non-zero values in the transformed plot so the
    # distribution of non-trivial values is visible without the zero spike.
    if many_zeros:
        non_zero_mask = feature_values != 0
        sorted_transformed = np.sort(transformed_values[non_zero_mask])  # [n_nonzero]
        transform_label += " (non-zero only)"
    else:
        sorted_transformed = np.sort(transformed_values)  # [n_samples]

    if len(sorted_transformed) > 0:
        t_mean = np.mean(sorted_transformed)
        t_std = np.std(sorted_transformed)
        print(f"Mean (transformed): {t_mean:.2f}, std (transformed): {t_std:.2f}")
        print(f"\tin +/- 1 std range (transformed): {np.sum(np.abs(sorted_transformed - t_mean) <= t_std) / len(sorted_transformed) * 100:.2f}%")

    mutual_info = mutual_info_regression(feature_values.reshape(-1, 1), target_values)
    log_mutual_info = mutual_info_regression(feature_values.reshape(-1, 1), np.log(np.where(target_values > 0, target_values, 1)))
    print(f"Mutual info: {mutual_info[0]:.2f} (for log target: {log_mutual_info[0]:.2f})")

    sorted_final = np.sort(_compute_scaled_final(feature_values, logarythmic, many_zeros))  # [n_nonzero] or [n_samples]
    final_label = f"{feature_name} (scaled+softplus, non-zero)" if many_zeros else f"{feature_name} (scaled)"

    axes[0].plot(sorted_raw)
    axes[0].set_title(f"{feature_name} (raw)")
    axes[1].plot(sorted_transformed)
    axes[1].set_title(transform_label)
    axes[2].plot(sorted_final)
    axes[2].set_title(final_label)


def display_boolean_feature(
    feature_values: np.ndarray,  # [n_samples]
    target_values: np.ndarray,  # [n_samples]
    feature_name: str,
):
    assert len(feature_values) == len(target_values)

    # Drop entries where target is 0
    indexes = np.where(target_values > 0)[0]
    feature_values = feature_values[indexes]
    target_values = target_values[indexes]

    unique_values = np.unique(feature_values)
    value_counts = {
        value: np.sum(feature_values == value) for value in unique_values
    }

    print(f"Feature: {feature_name}")
    print(f"Value counts:")
    for value, count in value_counts.items():
        print(f"\t{value}: {count} ({count / len(feature_values) * 100:0.2f}%)")

    mutual_info = mutual_info_regression(feature_values.reshape(-1, 1), target_values)
    log_mutual_info = mutual_info_regression(feature_values.reshape(-1, 1), np.log(np.where(np.array(target_values) > 0, target_values, 1)))
    print(f"Mutual info: {mutual_info[0]:0.2f} (for log target: {log_mutual_info[0]:0.2f})")
