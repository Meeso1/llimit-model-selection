import torch
import numpy as np

from src.data_models.data_models import TrainingData
from src.utils.string_encoder import StringEncoder


def compute_pairwise_accuracy(
    scores_a: torch.Tensor,  # [batch_size]
    scores_b: torch.Tensor,  # [batch_size]
    labels: torch.Tensor,  # [batch_size] - 1 if a should win, -1 if b should win
) -> float:
    """
    Computes accuracy for pairwise comparisons.
    
    Accuracy is defined as the percentage of pairs where the model's prediction
    (which model has higher score) matches the human evaluation.
    
    Args:
        scores_a: Scores for model A
        scores_b: Scores for model B
        labels: Ground truth labels (1 if A wins, -1 if B wins)
    
    Returns:
        Accuracy as a float in [0, 1]
    """
    # Prediction: 1 if score_a > score_b, -1 otherwise
    predictions = torch.sign(scores_a - scores_b)  # [batch_size]
    
    # Handle ties (when scores are exactly equal) - treat as incorrect
    predictions = torch.where(predictions == 0, -labels, predictions)
    
    # Compare predictions to labels
    correct = (predictions == labels).float()  # [batch_size]
    
    accuracy = correct.mean().item()
    return accuracy


def compute_embedding_accuracy(
    sample_embeddings: np.ndarray,  # [n_samples, embedding_dim]
    sample_model_names: list[str],  # [n_samples]
    model_embeddings: dict[str, np.ndarray],  # model_name -> [embedding_dim]
) -> float:
    """
    Computes accuracy for embedding models.
    
    For each sample embedding, finds the closest model embedding (by Euclidean distance)
    and checks if it matches the actual model that generated the sample.
    
    Accuracy is defined as the percentage of samples where the closest model embedding
    matches the actual model.
    
    Args:
        sample_embeddings: Embeddings for individual prompt-response pairs (numpy array)
        sample_model_names: Actual model name for each sample
        model_embeddings: Dictionary mapping model names to their embeddings (numpy arrays)
    
    Returns:
        Accuracy as a float in [0, 1]
    """
    if len(sample_embeddings) == 0:
        return 0.0
    
    if len(sample_model_names) != len(sample_embeddings):
        raise ValueError(
            f"sample_model_names length ({len(sample_model_names)}) "
            f"must match sample_embeddings first dimension ({len(sample_embeddings)})"
        )
    
    if len(model_embeddings) == 0:
        raise ValueError("model_embeddings cannot be empty")
    
    # Get embedding dimension
    embedding_dim = sample_embeddings.shape[1]
    
    # Validate that all model embeddings have the same dimension
    for model_name, model_emb in model_embeddings.items():
        if model_emb.shape != (embedding_dim,):
            raise ValueError(
                f"Model embedding for '{model_name}' has shape {model_emb.shape}, "
                f"expected ({embedding_dim},)"
            )
    
    # Stack model embeddings into a matrix for efficient distance computation
    model_names_list = list(model_embeddings.keys())
    model_embeddings_matrix = np.stack([model_embeddings[name] for name in model_names_list])  # [n_models, embedding_dim]
    
    # Compute distances from each sample to each model embedding
    # Using broadcasting: sample_embeddings [n_samples, 1, embedding_dim] - model_embeddings_matrix [1, n_models, embedding_dim]
    # Then compute L2 norm: [n_samples, n_models]
    distances = np.linalg.norm(
        sample_embeddings[:, np.newaxis, :] - model_embeddings_matrix[np.newaxis, :, :],
        axis=2
    )  # [n_samples, n_models]
    
    # Find the closest model for each sample
    closest_model_indices = np.argmin(distances, axis=1)  # [n_samples]
    closest_model_names = [model_names_list[idx] for idx in closest_model_indices]
    
    # Check if closest model matches actual model
    correct = np.array([
        closest == actual
        for closest, actual in zip(closest_model_names, sample_model_names)
    ])
    
    accuracy = correct.mean()
    return float(accuracy)


def compute_comparisons_accuracy(
    data: TrainingData,
    scores: np.ndarray,
    encoder: StringEncoder,
) -> float:
    """
    Computes accuracy for win/lose comparisons.
    
    Accuracy is defined as the percentage of comparisons where the model's prediction
    (which model has higher score) matches the human evaluation.
    
    This skips ties and both_bad comparisons.
    """
    correct = 0
    counted = 0
    for entry in data.entries:
        model_a_id = encoder.encode(entry.model_a)
        model_b_id = encoder.encode(entry.model_b)
        
        if model_a_id is None or model_b_id is None:
            continue

        if entry.winner == "model_a":
            if scores[model_a_id] > scores[model_b_id]:
                correct += 1
            counted += 1
        elif entry.winner == "model_b":
            if scores[model_b_id] > scores[model_a_id]:
                correct += 1
            counted += 1

    return correct / counted
