"""PyTorch utility functions."""

from typing import Any
import torch


def state_dict_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Move all tensors in a state dictionary to CPU.
    
    This follows PyTorch best practices for saving models without affecting
    the original module's device placement. The canonical approach is:
    `{k: v.cpu() for k, v in model.state_dict().items()}`
    
    This function handles potential non-tensor values gracefully and supports
    nested dictionaries for compatibility with complex state structures.
    
    Args:
        state_dict: State dictionary potentially containing tensors
        
    Returns:
        New state dictionary with all tensors moved to CPU
        
    References:
        https://docs.pytorch.org/xla/master/learn/migration-to-xla-on-tpus.html
    """
    result = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.cpu()
        elif isinstance(value, dict):
            result[key] = state_dict_to_cpu(value)
        else:
            result[key] = value
    return result

