import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression


def display_numeric_feature(
    feature_values: np.ndarray,  # [n_samples]
    target_values: np.ndarray,  # [n_samples]
    feature_name: str,
    axes: tuple[plt.Axes, plt.Axes] = None,
):
    assert len(feature_values) == len(target_values)

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Drop entries where target is 0
    indexes = np.where(target_values > 0)[0]
    feature_values = feature_values[indexes]
    target_values = target_values[indexes]

    sorted_values = np.sort(feature_values)

    mode = max(np.unique(feature_values), key=lambda x: np.sum(feature_values == x))
    mode_count = np.sum(feature_values == mode)
    mean = np.mean(feature_values)
    std = np.std(feature_values)

    log_values = np.log(np.where(np.array(feature_values) > 0, feature_values, 1))
    sorted_log_values = np.sort(log_values)

    mode_log = np.log(mode)
    log_mean = np.mean(log_values)
    log_std = np.std(log_values)

    print(f"Mode: {mode} (log = {mode_log}), count: {mode_count} ({mode_count / len(feature_values) * 100:0.2f}%)")
    print(f"Mean: {mean:0.2f}, std: {std:0.2f}")
    print(f"\tin +/- 1 std range: {np.sum(np.abs(feature_values - mean) <= std) / len(feature_values) * 100:0.2f}%")
    print(f"Mean (log): {log_mean:0.2f}, std (log): {log_std:0.2f}")
    print(f"\tin +/- 1 std range (log): {np.sum(np.abs(log_values - log_mean) <= log_std) / len(feature_values) * 100:0.2f}%")

    mutual_info = mutual_info_regression(feature_values.reshape(-1, 1), target_values)
    log_mutual_info = mutual_info_regression(feature_values.reshape(-1, 1), np.log(np.where(np.array(target_values) > 0, target_values, 1)))
    print(f"Mutual info: {mutual_info[0]:0.2f} (for log target: {log_mutual_info[0]:0.2f})")

    axes[0].plot(sorted_values)
    axes[0].set_title(f"{feature_name}")
    axes[1].plot(sorted_log_values)
    axes[1].set_title(f"log({feature_name})")


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
