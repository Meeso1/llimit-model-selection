# DnEmbeddingModel

## Overview

`DnEmbeddingModel` is a feedforward neural network for prompt routing. Unlike `DenseNetworkModel`, which uses a learned integer embedding for each model, this model uses a **pre-trained embedding model** (e.g., a triplet-finetuned sentence transformer) to produce dense vector representations of LLM models. These model embeddings are compared against prompt embeddings by the scoring network.

## Architecture

### Input Modalities

The scoring network receives three inputs, concatenated (or projected first, see below):

- **Prompt embedding** — dense vector from a sentence transformer (e.g., `all-MiniLM-L6-v2`, 384-dim)
- **Prompt features** — 45 scalar features extracted from prompt text (complexity, domain, style, etc.)
- **Model embedding** — dense vector from the trained embedding model (configurable dim)

### Network (`_DenseNetwork`)

All three inputs are combined and passed through a sequence of hidden blocks. Each block applies:

```
Linear → LayerNorm → LeakyReLU(0.1) → Dropout
```

Optionally, a **residual skip connection** wraps each block:
- Identity if `in_dim == out_dim`
- `Linear(in_dim, out_dim, bias=False)` projection otherwise

The final output is a scalar score per `(prompt, model)` pair. A `tanh` is applied at inference time to constrain scores to `[-1, 1]`.

### Input Projections (`input_proj_dim`)

Each of the three input modalities is first independently projected to `input_proj_dim` via `Linear + LeakyReLU`, then concatenated. This aligns the modalities (which differ in scale and space) before mixing them, so the trunk always receives a homogeneous `input_proj_dim * 3`-dimensional input regardless of the raw embedding sizes.

### Weight Initialization

All linear layers (including skip projections and input projections) are initialized with Kaiming Normal (`a=0.1`, `nonlinearity='leaky_relu'`), matching the activation function.

### Gradient Clipping

Training applies `clip_grad_norm_(max_norm=1.0)` after each backward pass to prevent gradient explosions in deeper configurations.

## Configuration

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hidden_dims` | `list[int]` | `[256, 128, 64]` | Hidden layer sizes |
| `dropout` | `float` | `0.2` | Dropout probability in each block |
| `use_skip_connections` | `bool` | `False` | Whether to add residual connections around each block |
| `input_proj_dim` | `int` | `64` | Project each input modality to this dim before concatenation |
| `optimizer_spec` | `OptimizerSpecification` | AdamW lr=0.001 | Optimizer and scheduler config |
| `balance_model_samples` | `bool` | `True` | Weighted sampling to balance model representation |
| `embedding_model_name` | `str` | `"all-MiniLM-L6-v2"` | Sentence transformer for prompt embedding |
| `embedding_spec` | `EmbeddingSpec \| None` | `None` | Spec to create the model embedding model |
| `min_model_comparisons` | `int` | `20` | Min comparisons required to include a model |
| `embedding_model_epochs` | `int` | `10` | Epochs to train the embedding model |
| `base_model_name` | `str \| None` | `None` | Base model to use as starting point, in `"model_type/model_name"` format |
| `seed` | `int` | `42` | Random seed |
| `ranking_loss_type` | `"margin_ranking"` \| `"bradley_terry"` | `"margin_ranking"` | Pairwise ranking loss: margin hinge or Bradley-Terry (sigmoid cross-entropy) |

### JSON Spec Fields (`dn_embedding`)

```json
{
  "model_type": "dn_embedding",
  "hidden_dims": [512, 256, 128, 64],
  "use_skip_connections": true,
  "input_proj_dim": 128,
  "optimizer": { ... },
  "embedding_spec": { ... },
  ...
}
```

## Training Objective

The pairwise ranking loss is configurable via `ranking_loss_type`:

- **`margin_ranking`** (default): `MarginRankingLoss(margin=0.1)` — encourages `score(winner) > score(loser) + margin`.
- **`bradley_terry`**: Bradley-Terry (sigmoid cross-entropy) on the score difference — models P(winner) = sigmoid(score_winner - score_loser), never saturates so gradients keep flowing.

Each training sample is a `(prompt, model_a, model_b, winner)` comparison. The network scores both `(prompt, model_a)` and `(prompt, model_b)`; the chosen loss compares the two scores.

## Base Model

When `base_model_name` is set (e.g. `"gradient_boosting/gradient-boosting-scoring"`), the DN model learns the **residual** on top of the base model's predictions:

- **Training**: The base model's scores for each training pair are cached upfront. During each forward pass, the effective score is `base_score + dn_score`; the ranking loss is applied on effective scores.
- **Inference**: `final_score = base_score + tanh(dn_score)`, consistent with the existing tanh-clipped inference output.

The base model's state is persisted in the saved model file, so it is fully self-contained when loading. In the JSON spec, set `"base_model": "model_type/model_name"` (same format as `GradientBoostingSpecification`).

## Embedding Model

The model embedding model is trained jointly (or loaded from a checkpoint) and produces fixed-size vectors for each LLM. At inference time, these embeddings are used directly—no lookup table, so the model can score LLMs not seen at training time if their embedding is available. See `docs/models/attention_embedding_model.md` for the default attention-based embedding model.

## Limitations

- Inference requires either a trained embedding for the target model, or a "default" fallback embedding.
- Conversation history is unavailable during inference, so context-based prompt features use empty history.
- Old saved models (before skip connections / LayerNorm were added) are not forward-compatible with the new `_DenseNetwork` architecture and must be retrained.
