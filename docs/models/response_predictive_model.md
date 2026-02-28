# Response Predictive Model

## Overview

The Response Predictive Model is a three-component neural architecture that learns to predict and score response representations. Unlike simpler scoring models that learn direct mappings from (prompt, model) to score, this model explicitly learns "what kind of response will this model produce for this prompt?" and then scores that predicted response.

**Key Innovation:** The model is trained with dense regression supervision on real response representations, not just binary comparison labels. This provides a richer training signal than traditional pairwise ranking approaches.

### Core Components

1. **ResponseEncoder** (trainable): Maps real response embeddings + scalar features to a learned representation space
2. **ResponsePredictor**: Given a prompt and model embedding, predicts a response representation in that same learned space
3. **ResponseScorer**: Given a prompt and a response representation, assigns a quality score

During training, real responses are available: the ResponseEncoder produces "target" representations, and the ResponsePredictor learns to match them. During inference, only the ResponsePredictor's output is used (real responses are not available).

## Architecture

### Component 1: ResponseEncoder (Trainable MLP)

Maps real response data to a learned representation space.

**Input:** `concat([response_embedding, response_scalar_features])`
- `response_embedding`: `[d_response_emb]` — from frozen sentence transformer (384 from `all-MiniLM-L6-v2`)
- `response_scalar_features`: `[d_response_features]` — 32 scalar features (interaction, lexical, structural)

**Architecture:**
```
input_dim = d_response_emb + d_response_features  # e.g., 384 + 32 = 416
layers:
  Linear(input_dim, 256) → LeakyReLU(0.1) → Dropout(0.2)
  Linear(256, response_repr_dim)  # e.g., 128
```

**Output:** `response_repr` of shape `[response_repr_dim]`

**Why trainable?** A frozen target (e.g., raw sentence transformer embedding) is arbitrary. A trainable encoder learns a representation constrained by two objectives:
1. **Predictability**: Must be predictable from (prompt, model_embedding)
2. **Usefulness for scoring**: Must contain information useful for quality assessment

These constraints together define an optimal representation space for this task.

### Component 2: ResponsePredictor (MLP)

Predicts what response representation a given model will produce for a given prompt.

**Input:** `concat([prompt_embedding, prompt_features, model_embedding])`
- `prompt_embedding`: `[d_prompt_emb]` — from frozen sentence transformer (384)
- `prompt_features`: `[d_prompt_features]` — 45 scalar features from prompt analysis
- `model_embedding`: `[d_model_emb]` — from embedding model (64-128)

**Architecture:**
```
input_dim = d_prompt_emb + d_prompt_features + d_model_emb  # e.g., 384 + 45 + 64 = 493
layers:
  Linear(input_dim, 512) → LeakyReLU(0.1) → Dropout(0.2)
  Linear(512, 256) → LeakyReLU(0.1) → Dropout(0.2)
  Linear(256, response_repr_dim)  # e.g., 128
```

**Output:** `predicted_response_repr` of shape `[response_repr_dim]`

**Loss:** Cosine embedding loss between predicted and encoder-produced representations. Cosine similarity is used because we care about the *direction* of the representation (what aspects it captures), not exact magnitude.

### Component 3: ResponseScorer (MLP)

Scores response representations in the context of prompts.

**Input:** `concat([prompt_embedding, prompt_features, response_repr])`
- `prompt_embedding`: `[d_prompt_emb]` — same prompt embedding as ResponsePredictor
- `prompt_features`: `[d_prompt_features]` — same 45 prompt features
- `response_repr`: `[response_repr_dim]` — either predicted or real representation

**Architecture:**
```
input_dim = d_prompt_emb + d_prompt_features + response_repr_dim  # e.g., 384 + 45 + 128 = 557
layers:
  Linear(input_dim, 256) → LeakyReLU(0.1) → Dropout(0.2)
  Linear(256, 128) → LeakyReLU(0.1) → Dropout(0.2)
  Linear(128, 1) → Tanh  # Output constrained to [-1, 1]
```

**Output:** scalar score in [-1, 1]

**Loss:** `MarginRankingLoss(margin=0.1)` — same as existing scoring models

**Critical design:** The ResponseScorer does **NOT** receive the model embedding directly. It only "knows" about the model through the response representation. This forces scoring to go through the response prediction pathway.

## Training Strategy

### Joint Training

All three components (ResponseEncoder, ResponsePredictor, ResponseScorer) train simultaneously with three loss functions:

```
total_loss = scoring_loss + alpha * prediction_loss + beta * predictability_loss
```

Where:
- `scoring_loss`: Margin ranking loss on pairwise model comparisons
- `prediction_loss`: Trains the **predictor** to match the encoder's output — encoder is treated as a fixed target (its output is detached)
- `predictability_loss`: Trains the **encoder** to produce representations the predictor can currently match — predictor output is detached
- `alpha`: `prediction_loss_weight` hyperparameter (default: 1.0)
- `beta`: `predictability_loss_weight` hyperparameter (default: 0.2)

**Why two separate losses instead of one?** A single symmetric cosine loss `cosine(pred_repr, real_repr)` creates a bidirectional gradient coupling that causes representation collapse: the encoder learns to ignore response content and simply match whatever the predictor produces, since that minimises the loss even without meaningful representations. By splitting into two asymmetric losses with stop-gradients, each component is trained by clearly separated signals:

| Component | `scoring_loss` gradient | `prediction_loss` gradient | `predictability_loss` gradient |
|-----------|-------------------------|---------------------------|-------------------------------|
| **Encoder** | "produce representations useful for ranking" (via real reprs in mixed scorer input) | none (encoder detached) | "move toward what the predictor can currently produce" |
| **Predictor** | "produce representations useful for ranking" (via predicted reprs in mixed scorer input) | "move toward what the encoder currently produces" | none (predictor detached) |
| **Scorer** | "rank correctly" | none | none |

Setting `predictability_loss_weight = 0` fully disconnects the encoder from the predictor's gradients (encoder is shaped only by the scoring objective).

### Mixed-Representation Training

During training, the ResponseScorer sees a mix of real (encoder-produced) and predicted representations. This bridges the distribution gap between training and inference.

**Schedule (linear decay with floor):**
```python
current_ratio = max(min_real_repr_ratio, real_repr_ratio - real_repr_decay_per_epoch * (epoch - 1))
```

With defaults `real_repr_ratio=0.8`, `real_repr_decay_per_epoch=0.04`, and `min_real_repr_ratio=0.1`, the ratio decays from 0.8 toward 0.1 (floor). The floor ensures the encoder always receives some scoring gradient through the real-representation branch of the mixer, preventing it from freezing entirely once the ratio would otherwise reach zero.

**Per-sample mixing:**
```python
use_real = torch.rand(batch_size, generator=seeded_generator) < current_ratio  # [batch]
repr_a = torch.where(use_real.unsqueeze(-1), real_repr_a, pred_repr_a)
repr_b = torch.where(use_real.unsqueeze(-1), real_repr_b, pred_repr_b)
```

**Rationale:** Early in training, the ResponsePredictor is noisy, so using real representations helps the ResponseScorer learn a good quality function. Later, switching to predicted representations lets the scorer adapt to the distribution it'll see at inference.

### Forward Pass Per Batch

```python
# 1. Predict response representations for both models
pred_repr_a = network.forward_predict(prompt_emb, prompt_features, model_emb_a)
pred_repr_b = network.forward_predict(prompt_emb, prompt_features, model_emb_b)

# 2. Encode real responses (training only)
real_repr_a = network.forward_encode_response(response_emb_a, response_features_a)
real_repr_b = network.forward_encode_response(response_emb_b, response_features_b)

# 3. Prediction loss: train predictor to match encoder (encoder output detached)
pred_loss = cosine_loss(pred_repr_a, real_repr_a.detach(), target_ones) \
          + cosine_loss(pred_repr_b, real_repr_b.detach(), target_ones)

# 4. Predictability loss: nudge encoder toward what predictor can produce (predictor output detached)
predictability_loss = cosine_loss(pred_repr_a.detach(), real_repr_a, target_ones) \
                    + cosine_loss(pred_repr_b.detach(), real_repr_b, target_ones)

# 5. Mix representations based on current ratio (seeded for reproducibility)
use_real = torch.rand(batch_size, generator=generator) < current_ratio
repr_a = torch.where(use_real.unsqueeze(-1), real_repr_a, pred_repr_a)
repr_b = torch.where(use_real.unsqueeze(-1), real_repr_b, pred_repr_b)

# 6. Score (mixed) representations
score_a = network.forward_score(prompt_emb, prompt_features, repr_a)
score_b = network.forward_score(prompt_emb, prompt_features, repr_b)
scoring_loss = margin_ranking_loss(score_a, score_b, labels)

# 7. Total loss and backprop
total_loss = scoring_loss + alpha * pred_loss + beta * predictability_loss
total_loss.backward()
optimizer.step()
```

## Feature Extraction

### Prompt Features (45 features)

Extracted using `src/preprocessing/scoring_feature_extraction.py`:
- Task type indicators (10 features)
- Complexity features (8 features)
- Domain indicators (12 features)
- Linguistic style (6 features)
- Context features (4 features)
- Output format expectations (5 features)

### Response Features (32 features)

Extracted using `src/preprocessing/model_embedding_feature_extraction.py`:

**Interaction Features (6 features):**
- Cosine similarity between prompt and response embeddings
- Euclidean distance between embeddings
- Dot product of embeddings
- Character length ratio (response/prompt)
- Token count ratio
- Characters per token in response

**Lexical Features (11 features):**
- Type-token ratio (vocabulary richness)
- Hapax legomena ratio
- Average/std word length
- Word length distribution (4 bins)
- Uppercase/digit/whitespace ratios

**Structural Features (15 features):**
- Sentence/paragraph counts
- Average/std/max sentence length
- Punctuation ratios (6 types)
- Code blocks, bullet points, numbered lists

All numeric features are normalized using `SimpleScaler` (separate scalers for prompt and response features).

## Metrics & Diagnostics

### Standard Metrics
- **total_loss**: Combined loss (scoring + α × prediction)
- **train_accuracy**: Pairwise accuracy on training data
- **val_loss** / **val_accuracy**: Validation metrics

### Additional Metrics (in `additional_metrics`)

| Metric | Range | Description |
|--------|-------|-------------|
| `scoring_loss` | [0, ∞) | MarginRankingLoss component |
| `prediction_loss` | [0, ∞) | CosineEmbeddingLoss for predictor (encoder detached) |
| `predictability_loss` | [0, ∞) | CosineEmbeddingLoss for encoder predictability (predictor detached) |
| `prediction_quality` | [0, 1] | Mean cosine similarity between predicted and real representations, rescaled: `(1 + cos_sim) / 2`. 0.5 = random, 1.0 = perfect |
| `scorer_real_repr_accuracy` | [0, 1] | Accuracy when using **real** representations (scorer performance in isolation) |
| `repr_mean_variance` | [0, ∞) | Mean per-dimension variance of encoder outputs (monitors collapse, should stay > 0.01) |
| `current_real_repr_ratio` | [0, 1] | Current mixed-representation ratio for this epoch |

**Validation variants:** All metrics except `current_real_repr_ratio` are also logged with `val_` prefix.

### Diagnostic Checks

1. **`prediction_quality` should increase** over epochs. If it stays near 0.5, the ResponsePredictor isn't learning.
2. **`scorer_real_repr_accuracy` > `train_accuracy`**: If the scorer can't score well even with real representations, the ResponseEncoder isn't producing useful representations.
3. **Gap between `scorer_real_repr_accuracy` and `train_accuracy`**: Represents the "prediction quality tax" — accuracy lost due to imperfect response prediction.
4. **`repr_mean_variance` should stay > 0.01**: If it drops toward 0, representation collapse is occurring.
5. **Smooth transition**: As `current_real_repr_ratio` decreases, verify `train_accuracy` doesn't drop suddenly.

## Why This Could Work

### Richer Training Signal
Current scoring models learn from binary comparison labels only: "model A beat model B." This is just one bit of information per sample. The ResponsePredictor is trained with dense regression against real response representations, extracting much more information per sample.

### Natural Decomposition
Separating "what will the response look like?" from "how good is that response?" is principled. Each module has a clear objective:
- ResponsePredictor: learns (prompt, model) → expected response characteristics
- ResponseScorer: learns a quality function over (prompt, response characteristics)

### Prompt-Dependent Model Differentiation
Existing model embeddings are static per model. The ResponsePredictor produces prompt-dependent representations — different aspects of a model's behavior activate for different prompts. This captures how model quality varies by task type.

### Transferable Quality Function
The ResponseScorer operates on response representations, not model identities. It learns "what makes a good response to this prompt" — knowledge that could generalize across models.

### Learned Representation Space
Because the ResponseEncoder is trainable, the representation space is shaped by actual training objectives. It learns to emphasize aspects of responses that are both predictable and useful for scoring, while discarding noise.

## Potential Challenges

### Prediction Difficulty
Predicting any representation of a response from (prompt, model embedding) is extremely hard. Responses have high variance. If the ResponsePredictor is too noisy, the ResponseScorer can't extract useful signal.

### Error Propagation
The ResponseScorer never sees real response representations at inference. If predicted representations differ significantly from real ones (distribution shift), the scorer may perform poorly. Mixed-representation training mitigates this.

### Representation Collapse
Degenerate solution: encoder maps all responses to the same point, predictor trivially predicts that point. This is prevented by the scoring loss (identical representations can't differentiate quality) and monitored via `repr_mean_variance`.

### Information Bottleneck
The response representation must be both predictable and informative. These goals may conflict — what's easiest to predict might not be most useful for scoring.

## Hyperparameters

### Key Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `response_repr_dim` | 128 | 32-256 | Lower = easier to predict, higher = more expressive |
| `prediction_loss_weight` | 1.0 | 0.1-2.0 | Weight for predictor→encoder loss (alpha); trains predictor |
| `predictability_loss_weight` | 0.2 | 0.0-1.0 | Weight for encoder→predictor nudge (beta); 0 = encoder shaped only by scoring |
| `encoder_hidden_dims` | [256] | [128]-[256, 128] | Encoder capacity |
| `predictor_hidden_dims` | [512, 256] | [256, 128]-[512, 384, 256] | Larger for harder prediction |
| `scorer_hidden_dims` | [256, 128] | [128, 64]-[256, 128] | Similar to existing scoring heads |
| `dropout` | 0.2 | 0.1-0.3 | Standard regularization |
| `real_repr_ratio` | 0.8 | 0.5-1.0 | Initial probability of using real representations |
| `real_repr_decay_per_epoch` | 0.04 | 0.02-0.08 | Linear decay speed |
| `min_real_repr_ratio` | 0.1 | 0.0-0.3 | Floor for real representation ratio; ensures encoder always gets scoring gradient |

### Embedding Model Parameters

The model uses an embedding model to learn model representations (same as `DnEmbeddingModel` and `TransformerEmbeddingModel`):

- `embedding_model_name`: Sentence transformer for prompt/response embedding (default: `"all-MiniLM-L6-v2"`)
- `embedding_spec`: Configuration for the embedding model (frozen, finetunable, or attention-based)
- `load_embedding_model_from`: Load pre-trained embeddings from another model (format: `"model_type/model_name"`)
- `min_model_comparisons`: Minimum comparisons required per model (default: 20)
- `embedding_model_epochs`: Epochs for training embedding model (default: 10)

## Training Process

### Phase 1: Embedding Model Training
- Trains or loads embeddings for LLM models
- Uses triplet loss or attention mechanisms
- Produces fixed-dimensional vectors (e.g., 64 or 128)
- Skipped if `load_embedding_model_from` is specified

### Phase 2: Joint Training
- Trains ResponseEncoder, ResponsePredictor, and ResponseScorer simultaneously
- Each batch:
  1. Predict response representations for both models
  2. Encode real responses (via ResponseEncoder)
  3. Compute prediction loss (cosine embedding)
  4. Mix real and predicted representations (using seeded randomness)
  5. Score mixed representations
  6. Compute scoring loss (margin ranking)
  7. Backpropagate combined loss

### Why Joint Over Sequential?

**Sequential** (train predictor first, freeze, then train scorer):
- Simpler, more stable
- But: predictor learns representations optimized for regression, not scoring
- Representations might be easy to predict but irrelevant for quality assessment

**Joint** (default):
- Gradients from scoring loss flow back through encoder and (when using predicted representations) through predictor
- All components learn representations *useful for scoring*, not just easy to encode/predict
- More challenging but theoretically stronger

## Data Format

### Training Data

The preprocessor (`ResponsePredictivePreprocessor`) processes training data with responses:

```python
ResponsePredictivePair:
  - prompt_embedding: [384]
  - prompt_features: [45]
  - response_embedding_a: [384]
  - response_embedding_b: [384]
  - response_features_a: [32]
  - response_features_b: [32]
  - model_id_a: int
  - model_id_b: int
  - winner_label: int  # 0 = model_a wins, 1 = model_b wins
```

After embedding model training, model embeddings are added:

```python
ResponsePredictivePairWithEmbedding:
  (all fields from ResponsePredictivePair)
  - model_embedding_a: [64]
  - model_embedding_b: [64]
```

### Inference Data

At inference, no responses are available:

```python
ResponsePredictiveInferenceData:
  - prompt_embeddings: [n_prompts, 384]
  - prompt_features: [n_prompts, 45]
  - model_embeddings: [n_models, 64]
```

The model predicts response representations and scores them directly.

## Usage Example

### Training from Spec

```bash
uv run python main.py train training_specs/response_predictive.json
```

Example spec:
```json
{
  "data": {
    "validation_split": 0.2,
    "seed": 42
  },
  "epochs": 30,
  "batch_size": 512,
  "model": {
    "name": "response-predictive",
    "spec": {
      "model_type": "response_predictive",
      "response_repr_dim": 128,
      "encoder_hidden_dims": [256],
      "prediction_loss_weight": 1.0,
      "predictor_hidden_dims": [512, 256],
      "scorer_hidden_dims": [256, 128],
      "dropout": 0.2,
      "real_repr_ratio": 0.8,
      "real_repr_decay_per_epoch": 0.04,
      "min_real_repr_ratio": 0.1,
      "predictability_loss_weight": 0.2,
      "optimizer": {
        "optimizer_type": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.01
      },
      "embedding_model_name": "all-MiniLM-L6-v2",
      "embedding_spec": {
        "embedding_type": "frozen",
        "encoder_model_name": "all-MiniLM-L6-v2",
        "hidden_dims": [128, 64]
      },
      "min_model_comparisons": 20
    }
  }
}
```

### Loading and Using

```python
from src.models.response_predictive_model import ResponsePredictiveModel
from src.data_models.data_models import InputData

# Load trained model
model = ResponsePredictiveModel.load("response-predictive")

# Predict scores
input_data = InputData(
    prompts=["Write a Python function to sort a list"],
    model_names=["gpt-4", "claude-3", "llama-3"],
)
output = model.predict(input_data)
scores = output.scores  # dict[model_name, np.ndarray[n_prompts]]
```

## Implementation Files

- **Model**: `src/models/response_predictive_model.py`
- **Data types**: `src/data_models/response_predictive_types.py`
- **Preprocessor**: `src/preprocessing/response_predictive_preprocessor.py`
- **Response features**: `src/preprocessing/model_embedding_feature_extraction.py`
- **Prompt features**: `src/preprocessing/scoring_feature_extraction.py`
- **Model registration**: `src/models/model_loading.py`
- **CLI types**: `src/scripts/model_types.py`
- **Training**: `src/scripts/train.py`
- **Split utilities**: `src/utils/data_split.py`

## Comparison with Other Models

### vs DnEmbeddingModel
- **DnEmbedding**: Direct (prompt, model) → score mapping with frozen prompt embeddings
- **ResponsePredictive**: Adds explicit response prediction pathway, richer training signal
- **Trade-off**: ResponsePredictive is more complex but has access to response-level supervision

### vs TransformerEmbeddingModel
- **TransformerEmbedding**: Fine-tunes transformer for prompt encoding
- **ResponsePredictive**: Uses frozen prompt encoding but adds response prediction
- **Complementary**: Could be combined (fine-tuned prompt encoding + response prediction)

### vs Attention Embedding Models
- **Attention**: Uses responses to learn model embeddings via set aggregation
- **ResponsePredictive**: Uses responses to improve the scoring function itself
- **Key difference**: Attention embeds models from responses, ResponsePredictive scores responses

## Expected Performance

This model is experimental. Realistic expectations:

1. **vs models that ignore responses** (e.g., frozen embeddings): May outperform due to response-level supervision
2. **vs models that use responses for embeddings** (e.g., attention-based): May underperform, as those models have direct access to response information during inference
3. **Unique advantage**: Uses responses to improve the *scoring function*, not just model representation

The model should be compared against `DnEmbeddingModel` and `TransformerEmbeddingModel` baselines.

### Fallback Strategy

If the full model doesn't outperform baselines, the ResponseScorer alone (trained on real response representations) is still useful. It validates whether the approach is limited by prediction quality or scoring architecture. The `scorer_real_repr_accuracy` metric directly measures this upper bound.

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Response prediction too noisy | High | High | Lower `response_repr_dim`, monitor `prediction_quality` |
| Distribution shift (predicted vs real) | Medium | High | Mixed-representation training with linear decay |
| Representation collapse | Medium | High | Monitor `repr_mean_variance`, scoring loss provides natural pressure |
| Prediction loss dominates | Medium | Medium | Tune `prediction_loss_weight`, monitor both loss components |
| Overfitting (more parameters) | Medium | Low | Dropout, validation monitoring |

## Future Extensions

These are **not** currently implemented but listed for reference:

1. **Transformer-based prompt encoding**: Replace frozen prompt embeddings with fine-tuned transformer
2. **Attention-based ResponsePredictor**: Use cross-attention between prompt tokens and model embedding
3. **Auxiliary losses**: Add objectives like predicting response length or style features
4. **Base model support**: Learn incremental improvements over a base scoring model
5. **KL regularization**: Regularize the response representation space (if collapse becomes an issue)

## Related Documentation

- [Embedding Models](triplet_encoder_models.md) - Model embedding approaches (frozen, finetunable, attention)
- [Attention Embedding Model](attention_embedding_model.md) - Alternative approach using responses for model embeddings
- [Transformer Embedding Model](transformer_embedding_model.md) - Fine-tuned transformer approach
- [Models Overview](../models.md) - All model types
