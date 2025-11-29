# Models

This directory contains documentation for different model implementations.

## Implemented Models

### Dense Network Model
- **File**: [dense_network_model.md](models/dense_network_model.md)
- **Type**: Feedforward neural network with learned model embeddings
- **Input**: Fixed-size prompt embeddings + Model IDs
- **Output**: Scores in [-1, 1] for each (prompt, model) combination
- **Training**: Margin ranking loss on pairwise comparisons
- **Inference**: Can score any number of models simultaneously