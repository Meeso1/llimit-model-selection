"""Utilities for transformer model pooling strategies."""

from typing import Literal
import torch
import huggingface_hub
import json


PoolingMethod = Literal["mean", "cls", "last_token"]


def detect_pooling_method(model_name: str) -> PoolingMethod:
    """
    Attempt to detect the pooling method from a model's config.
    
    Models from sentence-transformers and compatible libraries often include
    a 1_Pooling/config.json file that specifies the pooling method.
    
    Args:
        model_name: HuggingFace model name/path
        
    Returns:
        Detected pooling method ('mean', 'cls', or 'last_token')
        Falls back to 'mean' if undetectable (safest general choice)
    """
    try:
        config_path = huggingface_hub.hf_hub_download(
            repo_id=model_name,
            filename="1_Pooling/config.json",
        )
        with open(config_path) as f:
            config = json.load(f)
        
        if config.get("pooling_mode_mean_tokens", False):
            return "mean"
        elif config.get("pooling_mode_cls_token", False):
            return "cls"
        elif config.get("pooling_mode_lasttoken", False):
            return "last_token"
    except Exception:
        # File doesn't exist, can't be downloaded, or can't be parsed
        pass
    
    # Default to mean pooling - works well for most embedding models
    return "mean"


def pool_embeddings(
    last_hidden_state: torch.Tensor,  # [batch_size, seq_len, hidden_size]
    attention_mask: torch.Tensor,  # [batch_size, seq_len]
    pooling_method: PoolingMethod,
) -> torch.Tensor:
    """
    Pool token embeddings into a single sentence embedding.
    
    Args:
        last_hidden_state: Token embeddings from transformer  # [batch_size, seq_len, hidden_size]
        attention_mask: Attention mask indicating valid tokens  # [batch_size, seq_len]
        pooling_method: Pooling strategy to use
        
    Returns:
        Pooled sentence embeddings  # [batch_size, hidden_size]
    """
    if pooling_method == "cls":
        return _pool_cls_token(last_hidden_state)
    elif pooling_method == "last_token":
        return _pool_last_token(last_hidden_state, attention_mask)
    elif pooling_method == "mean":
        return _pool_mean(last_hidden_state, attention_mask)
    else:
        raise ValueError(f"Invalid pooling method: {pooling_method}")  # pyright: ignore[reportUnreachable]


def _pool_cls_token(
    last_hidden_state: torch.Tensor,  # [batch_size, seq_len, hidden_size]
) -> torch.Tensor:
    """
    Use the [CLS] token (first token) as the sentence embedding.
    
    This is appropriate for models trained with NSP (Next Sentence Prediction)
    or similar objectives where the [CLS] token is specifically optimized.
    
    Args:
        last_hidden_state: Token embeddings  # [batch_size, seq_len, hidden_size]
        
    Returns:
        CLS token embeddings  # [batch_size, hidden_size]
    """
    return last_hidden_state[:, 0, :]


def _pool_last_token(
    last_hidden_state: torch.Tensor,  # [batch_size, seq_len, hidden_size]
    attention_mask: torch.Tensor,  # [batch_size, seq_len]
) -> torch.Tensor:
    """
    Use the last non-padding token as the sentence embedding.
    
    This is appropriate for decoder-only / causal models where information
    flows left-to-right and accumulates in the final token.
    
    Args:
        last_hidden_state: Token embeddings  # [batch_size, seq_len, hidden_size]
        attention_mask: Attention mask  # [batch_size, seq_len]
        
    Returns:
        Last token embeddings  # [batch_size, hidden_size]
    """
    # Get the index of the last non-padding token for each sequence
    seq_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
    batch_indices = torch.arange(
        last_hidden_state.size(0),
        device=last_hidden_state.device,
    )  # [batch_size]
    
    return last_hidden_state[batch_indices, seq_lengths, :]  # [batch_size, hidden_size]


def _pool_mean(
    last_hidden_state: torch.Tensor,  # [batch_size, seq_len, hidden_size]
    attention_mask: torch.Tensor,  # [batch_size, seq_len]
) -> torch.Tensor:
    """
    Average all token embeddings (excluding padding) as the sentence embedding.
    
    This is the most common and robust pooling method for sentence embeddings.
    It works well for most sentence-transformer and embedding-focused models.
    
    Args:
        last_hidden_state: Token embeddings  # [batch_size, seq_len, hidden_size]
        attention_mask: Attention mask  # [batch_size, seq_len]
        
    Returns:
        Mean-pooled embeddings  # [batch_size, hidden_size]
    """
    # Expand attention mask to match embedding dimensions
    mask_expanded = attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()
    ).float()  # [batch_size, seq_len, hidden_size]
    
    # Sum embeddings (zeroing out padding tokens)
    sum_embeddings = torch.sum(
        last_hidden_state * mask_expanded,
        dim=1,
    )  # [batch_size, hidden_size]
    
    # Count non-padding tokens per sequence
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [batch_size, hidden_size]
    
    # Compute mean
    return sum_embeddings / sum_mask  # [batch_size, hidden_size]

