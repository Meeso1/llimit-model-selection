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

### Simple Scoring Model
- **File**: [simple_scoring_model.md](models/simple_scoring_model.md)
- **Type**: Simple baseline with one learnable score per model (prompt-agnostic)
- **Input**: Model IDs only (ignores prompts)
- **Output**: Same scores for all prompts (based on learned model scores)
- **Training**: Neural network with margin ranking loss + tie/both_bad penalties
- **Inference**: Very fast (lookup table)

### ELO Scoring Model
- **File**: [elo_scoring_model.md](models/elo_scoring_model.md)
- **Type**: ELO rating system baseline (prompt-agnostic)
- **Input**: Model IDs only (ignores prompts)
- **Output**: Same scores for all prompts (based on ELO ratings)
- **Training**: Iterative ELO rating updates (no neural network)
- **Inference**: Very fast (lookup table)