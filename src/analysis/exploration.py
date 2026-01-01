import numpy as np
import matplotlib.pyplot as plt

from src.data_models.data_models import TrainingData
from src.utils.string_encoder import StringEncoder


def get_wins_matrix(data: TrainingData, encoder: StringEncoder | None = None) -> tuple[np.ndarray, StringEncoder]:
    """
    Returns a matrix of wins for each model. Element (i, j) is the number of times model i beat model j.
    """
    if encoder is None:
        encoder = StringEncoder()
        encoder.fit([name for entry in data.entries for name in [entry.model_a, entry.model_b]])

    result = np.zeros((encoder.size, encoder.size))
    for entry in data.entries:
        model_id_a = encoder.encode(entry.model_a)
        model_id_b = encoder.encode(entry.model_b)

        if model_id_a is None or model_id_b is None:
            continue

        if entry.winner == "model_a":
            result[model_id_a, model_id_b] += 1
        elif entry.winner == "model_b":
            result[model_id_b, model_id_a] += 1
        
    return result, encoder


def get_losses_matrix(data: TrainingData, encoder: StringEncoder | None = None) -> tuple[np.ndarray, StringEncoder]:
    """
    Returns a matrix of losses for each model. Element (i, j) is the number of times model i lost to model j.
    """
    wins_matrix, encoder = get_wins_matrix(data, encoder)
    return wins_matrix.T, encoder


def get_ties_matrix(data: TrainingData, encoder: StringEncoder | None = None) -> tuple[np.ndarray, StringEncoder]:
    if encoder is None:
        encoder = StringEncoder()
        encoder.fit([name for entry in data.entries for name in [entry.model_a, entry.model_b]])

    result = np.zeros((encoder.size, encoder.size))
    for entry in data.entries:
        model_id_a = encoder.encode(entry.model_a)
        model_id_b = encoder.encode(entry.model_b)
        
        if model_id_a is None or model_id_b is None:
            continue

        if entry.winner == "tie":
            result[model_id_a, model_id_b] += 1
            result[model_id_b, model_id_a] += 1

    return result, encoder


def get_both_bad_matrix(data: TrainingData, encoder: StringEncoder | None = None) -> tuple[np.ndarray, StringEncoder]:
    if encoder is None:
        encoder = StringEncoder()
        encoder.fit([name for entry in data.entries for name in [entry.model_a, entry.model_b]])

    result = np.zeros((encoder.size, encoder.size))
    for entry in data.entries:
        model_id_a = encoder.encode(entry.model_a)
        model_id_b = encoder.encode(entry.model_b)

        if model_id_a is None or model_id_b is None:
            continue

        if entry.winner == "both_bad":
            result[model_id_a, model_id_b] += 1
            result[model_id_b, model_id_a] += 1

    return result, encoder


def plot_model_by_model_matrix(
    matrix: np.ndarray, 
    model_encoder: StringEncoder, 
    title: str | None = None,
    axes: plt.Axes | None = None
) -> None:
    if matrix.shape != (model_encoder.size, model_encoder.size):
        raise ValueError(f"Matrix shape {matrix.shape} does not match model encoder size {model_encoder.size}")

    figure = None
    if axes is None:
        figure, axes = plt.subplots(figsize=(10, 10))

    axes.imshow(matrix, cmap='viridis')
    axes.set_title(title)
    axes.set_xlabel('Model')
    axes.set_ylabel('Model')
    axes.set_xticks(range(model_encoder.size), model_encoder.names, rotation=90)
    axes.set_yticks(range(model_encoder.size), model_encoder.names, rotation=0)

    if figure is not None:
        plt.show()
        