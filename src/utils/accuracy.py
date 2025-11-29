import torch


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

