import matplotlib.pyplot as plt
import numpy as np


def plot_loss(axes: plt.Axes, values: list[float | None], title: str):
    # Filter out None values while keeping track of original indices
    valid_indices = [i for i, v in enumerate(values) if v is not None]
    valid_values = np.array([v for v in values if v is not None])
    t = np.array(valid_indices)
    
    log_loss = np.log(valid_values)
    sliding_mean = np.convolve(log_loss, np.ones(10)/10, mode='valid')
    axes.plot(t, log_loss, label="log(loss)")
    axes.plot(t[9:], sliding_mean, label="Sliding mean (10)")

    min_sliding_mean = np.min(sliding_mean)
    axes.axhline(y=min_sliding_mean, color='r', linestyle='--', label=f'Min: {min_sliding_mean:.3f}')

    axes.set_ylabel("log(loss)")
    axes.set_title(title)
    axes.legend()


def plot_accuracy(axes: plt.Axes, values: list[float | None], title: str):
    # Filter out None values while keeping track of original indices
    valid_indices = [i for i, v in enumerate(values) if v is not None]
    valid_values = np.array([v for v in values if v is not None])
    t = np.array(valid_indices)
    
    # Inverted logarithmic scale: -log(1 - accuracy)
    # This transforms [0, 1] to [0, inf], emphasizing improvements near 1.0
    inv_log_acc = -np.log(1 - valid_values + 1e-10)  # Add small epsilon to avoid log(0)
    sliding_mean = np.convolve(inv_log_acc, np.ones(10)/10, mode='valid')
    axes.plot(t, inv_log_acc, label="-log(1-accuracy)")
    axes.plot(t[9:], sliding_mean, label="Sliding mean (10)")

    max_sliding_mean = np.max(sliding_mean)
    max_accuracy = 1 - np.exp(-max_sliding_mean)
    axes.axhline(y=max_sliding_mean, color='r', linestyle='--', label=f'Max: {max_accuracy*100:.1f}%')
    
    # Set y-axis labels to show accuracy percentages at logarithmic positions
    min_acc = np.min(valid_values)
    max_acc = np.max(valid_values)
    accuracy_levels = _generate_accuracy_levels(min_acc, max_acc)
    
    y_positions = [-np.log(1 - acc + 1e-10) for acc in accuracy_levels]
    y_labels = [f"{acc*100:.1f}%" for acc in accuracy_levels]
    axes.set_yticks(y_positions)
    axes.set_yticklabels(y_labels)
    
    axes.set_ylabel("Accuracy")
    axes.set_title(title)
    axes.legend()
    
    
def _generate_accuracy_levels(min_acc: float, max_acc: float, n_levels: int = 6) -> list[float]:
    """
    Dynamically generate accuracy levels following the pattern: 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, etc.
    Returns only levels that fall within or near the [min_acc, max_acc] range.
    """
    min_acc = max(float(min_acc), 0.0)
    max_acc = min(float(max_acc), 1.0)

    if max_acc - min_acc < 1e-6:
        return [min_acc]

    # Generate candidate levels following the pattern: 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, etc.
    candidates = [0.0, 0.5]  # Start with 0.0 and 0.5
    
    # Generate levels: 0.9, 0.99, 0.999, 0.9999, ...
    # and 0.95, 0.995, 0.9995, 0.99995, ...
    for num_nines in range(1, 10):  # Go up to very high accuracy
        # 0.9, 0.99, 0.999, etc.
        level_9 = 1.0 - 10**(-num_nines)
        candidates.append(level_9)
        
        # 0.95, 0.995, 0.9995, etc. (only if num_nines >= 1)
        level_95 = 1.0 - 0.5 * 10**(-num_nines)
        candidates.append(level_95)
    
    candidates.append(1.0)  # Add 1.0 at the end
    
    # Filter to only include levels within a reasonable range
    filtered = [
        level for level in candidates
        if min_acc <= level <= max_acc
    ]
    
    # If we have too few levels, include boundary values
    if len(filtered) < 2:
        filtered = sorted(set([min_acc, max_acc]))
    
    # Limit to approximately n_levels, keeping a good spread
    if len(filtered) > n_levels:
        # Keep first, last, and evenly spaced intermediate levels
        step = len(filtered) / (n_levels - 1)
        indices = [int(i * step) for i in range(n_levels - 1)] + [len(filtered) - 1]
        filtered = [filtered[i] for i in sorted(set(indices))]
    
    return sorted(filtered)
