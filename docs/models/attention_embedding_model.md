# Attention Embedding Model - Implementation

## Overview

The Attention Embedding Model is an LLM fingerprinting system that learns fixed-dimensional embeddings representing models based on their (prompt, response) behavior. The model treats embedding computation as a set embedding problem, using attention mechanisms to aggregate information from multiple prompt-response pairs.

**Key Properties:**
- **Input**: A set of (prompt, response) pairs from a single model
- **Output**: A fixed-dimensional embedding vector representing that model
- **Generalization**: Works for new models not seen during training (no retraining required)
- **Training**: Uses supervised contrastive learning to cluster same-model embeddings

**Model Type**: This is an **embedding component** (implements `EmbeddingModelBase`), similar to `TripletFrozenEncoderModel` and `TripletFinetunableEncoderModel`. It is used within `DnEmbeddingModel` to learn model representations, not as a standalone routing model.

**Architecture Position**: 
- `EmbeddingModelBase` (abstract interface)
  - `TripletModelBase` (triplet-based models)
    - `TripletFrozenEncoderModel`
    - `TripletFinetunableEncoderModel`
  - `AttentionEmbeddingModel` (attention-based model)

## Architecture

The model consists of two main components:

### 1. PairEncoder

Encodes individual (prompt, response) pairs into fixed-dimensional representations.

**Architecture:**
```
Input: 
  - prompt_emb: [batch, d_emb]  # Pre-computed text embedding
  - response_emb: [batch, d_emb]  # Pre-computed text embedding
  - scalar_features: [batch, d_scalar]  # Extracted features

Processing:
  1. Project embeddings:
     - prompt_h = Linear(d_emb → h_emb) → LayerNorm → ReLU → Dropout
     - response_h = Linear(d_emb → h_emb) → LayerNorm → ReLU → Dropout
  
  2. Compute interactions:
     - emb_diff = response_h - prompt_h
     - emb_prod = response_h * prompt_h  (element-wise)
  
  3. Project scalar features:
     - scalar_h = Linear(d_scalar → h_scalar) → LayerNorm → ReLU → Dropout
  
  4. Fuse all features:
     - concat = [prompt_h; response_h; emb_diff; emb_prod; scalar_h]
     - pair_encoding = MLP(concat) → [batch, h_pair]

Output: pair_encoding [batch, h_pair]
```

**Hyperparameters:**
- `d_emb`: Text embedding dimension (determined by sentence transformer, typically 384-1536)
- `d_scalar`: Scalar feature dimension (32 features by default)
- `h_emb`: Projected embedding dimension (default: 256)
- `h_scalar`: Projected scalar dimension (default: 64)
- `h_pair`: Pair encoding output dimension (default: 256)
- `pair_mlp_layers`: Number of MLP layers in fusion (default: 2)
- `dropout`: Dropout rate (default: 0.1)

### 2. SetAggregator

Aggregates multiple pair encodings into a single model embedding using multi-head attention pooling.

**Architecture:**
```
Input: pair_encodings [batch, num_pairs, h_pair]

Processing:
  1. Project keys and values:
     - K = Linear(h_pair → h_pair)
     - V = Linear(h_pair → h_pair)
  
  2. Multi-head attention:
     - Learnable query vectors Q (one per head)
     - Attention scores = softmax(Q @ K^T / sqrt(head_dim))
     - Weighted values = attention_weights @ V
  
  3. Output projection:
     - model_emb = Linear(h_pair → d_out) → LayerNorm

Output: model_embedding [batch, d_out]
```

**Hyperparameters:**
- `h_pair`: Pair encoding dimension (default: 256)
- `d_out`: Output model embedding dimension (default: 128)
- `num_attention_heads`: Number of attention heads (default: 4)

## Feature Extraction

The preprocessor extracts three types of features from each (prompt, response) pair:

### 1. Text Embeddings (Pre-computed)

Uses a pre-trained sentence transformer to embed prompts and responses:
- Default model: `all-MiniLM-L6-v2` (384 dimensions)
- Alternative: `all-mpnet-base-v2` (768 dimensions)
- Or any model from sentence-transformers library

### 2. Interaction Features (6 features)

Capture relationships between prompt and response:
- **Embedding-based**:
  - Cosine similarity between embeddings
  - Euclidean distance between embeddings
  - Dot product of embeddings
- **Length-based**:
  - Character length ratio (response/prompt)
  - Token count ratio (approximate, by whitespace splitting)
  - Characters per token in response

### 3. Stylometric Features (26 features)

Capture the writing style of the response:

**Lexical Features (11 features):**
- Type-token ratio (vocabulary richness)
- Hapax legomena ratio (words appearing once)
- Average word length
- Standard deviation of word length
- Word length distribution (4 bins: 1-3, 4-6, 7-9, 10+)
- Uppercase character ratio
- Digit character ratio
- Whitespace character ratio

**Structural Features (15 features):**
- Number of sentences
- Average sentence length
- Standard deviation of sentence length
- Maximum sentence length
- Number of paragraphs
- Average paragraph length
- Punctuation ratios (period, comma, question, exclamation, colon, semicolon)
- Code block count (triple backticks)
- Bullet point count
- Numbered list count

**Total scalar features: 32** (6 interaction + 11 lexical + 15 structural)

All scalar features are normalized using `StandardScaler` fit on the training set.

## Training

### Loss Function

The model uses **supervised contrastive loss** to learn embeddings:

```python
L = -mean(log(exp(sim(anchor, positive) / τ) / sum(exp(sim(anchor, all) / τ))))
```

Where:
- `sim(a, b)` = cosine similarity between embeddings a and b
- `τ` = temperature parameter (default: 0.07)
- Positives = other embeddings from the same model
- All = all other embeddings in the batch (excluding self)

**Effect**: Embeddings from the same model are pulled together, while embeddings from different models are pushed apart.

### Accuracy Metrics

During training, two accuracy metrics are computed to track progress:

**1. Nearest Neighbor Accuracy:**
- For each embedding in the batch, find its nearest neighbor (excluding itself)
- Check if the nearest neighbor comes from the same model
- Metric = percentage of correct nearest neighbor predictions
- **Good values**: Above 0.7 indicates good clustering

**2. Triplet Accuracy:**
- Sample random triplets: (anchor, positive, negative)
  - Positive = from same model as anchor
  - Negative = from different model
- Check if `distance(anchor, positive) < distance(anchor, negative)`
- Metric = percentage of correctly ordered triplets
- Typically samples 10% of batch size as number of triplets
- **Good values**: Above 0.8 indicates strong separation

These metrics are computed on the training batch after each epoch and provide insight into embedding quality.

### Validation

The model supports optional validation during training:

**Split Strategy:**
- Splits at the **model level** (not pair level)
- Ensures both train and validation sets have complete model representations
- The scaler fitted on the full dataset is shared between train and validation

**Validation Metrics:**
- Same metrics as training: loss, nearest neighbor accuracy, and triplet accuracy
- Computed in evaluation mode (no gradients, no dropout)
- Provides early stopping signal and overfitting detection

**Usage:**
Pass a `ValidationSplit` object to the `train()` method:
```python
from src.utils.data_split import ValidationSplit

model.train(
    data=training_data,
    validation_split=ValidationSplit(val_fraction=0.2, seed=42),
    epochs=50,
    batch_size=32
)
```

Validation metrics are logged alongside training metrics in the format `train/val`:
```
Epoch    1: loss = 0.1234/0.1456, nn_acc = 0.8500/0.8300, triplet_acc = 0.9200/0.9000 - 1.23s
```

### Batch Construction

Training batches are constructed as follows:
1. Sample `models_per_batch` random models (default: 16)
2. For each model, create `embeddings_per_model` embeddings (default: 4) by:
   - Sampling `pairs_per_model` random pairs (default: 32) for each embedding
   - Different pair subsets create different embeddings
3. Encode all pairs with PairEncoder
4. Reshape into `[models_per_batch * embeddings_per_model, pairs_per_model, h_pair]`
5. Aggregate each set of pairs with SetAggregator
6. Compute contrastive loss over all embeddings

**Note:** Creating multiple embeddings per model ensures that each batch contains "positive pairs" (embeddings from the same model), which is essential for supervised contrastive learning. Without this, the loss would be zero because there would be no positive pairs to contrast with negatives.

### Hyperparameters

**Model Architecture:**
- `h_emb`: 128-256 (projected embedding dimension)
- `h_scalar`: 32-64 (projected scalar dimension)
- `h_pair`: 128-256 (pair encoding dimension)
- `d_out`: 64-128 (final embedding dimension)
- `pair_mlp_layers`: 2-3
- `num_attention_heads`: 4-8
- `dropout`: 0.1-0.2

**Training:**
- `temperature`: 0.05-0.10 (lower = harder negatives)
- `learning_rate`: 1e-4 to 1e-3
- `weight_decay`: 1e-5
- `epochs`: 20-50
- `pairs_per_model`: 16-32 (pairs per embedding)
- `models_per_batch`: 8-16 (unique models per batch)
- `embeddings_per_model`: 2-8 (embeddings per model per batch, for contrastive learning)

**Data Filtering:**
- `min_model_comparisons`: 20 (minimum samples per model to include)

## Inference

To compute an embedding for a new model (not seen during training):

1. Collect 20-50+ (prompt, response) pairs from the model
2. Extract text embeddings using the same sentence transformer
3. Extract scalar features using the same feature extraction functions
4. Normalize scalar features using the fitted scaler from training
5. Encode pairs with the trained PairEncoder
6. Aggregate with the trained SetAggregator
7. The output is the model's embedding

**Important**: The preprocessor's state (fitted scaler) must be saved and loaded for inference.

## Usage

### Training with DnEmbeddingModel

The attention embedding model is used as an embedding component within `DnEmbeddingModel`. Create a training specification JSON file:

```json
{
  "data": {
    "max_samples": null,
    "validation_split": 0.1,
    "seed": 42
  },
  "epochs": 30,
  "batch_size": 32,
  "model": {
    "name": "dn-embedding-with-attention",
    "spec": {
      "model_type": "dn_embedding",
      "hidden_dims": [256, 128, 64],
      "optimizer": {
        "optimizer_type": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.00001
      },
      "balance_model_samples": true,
      "embedding_model_name": "all-MiniLM-L6-v2",
      "embedding_spec": {
        "embedding_type": "attention",
        "encoder_model_name": "all-MiniLM-L6-v2",
        "h_emb": 256,
        "h_scalar": 64,
        "h_pair": 256,
        "d_out": 128,
        "pair_mlp_layers": 2,
        "num_attention_heads": 4,
        "dropout": 0.1,
        "temperature": 0.07,
        "pairs_per_model": 32,
        "models_per_batch": 16,
        "embeddings_per_model": 4,
        "optimizer": {
          "optimizer_type": "adamw",
          "learning_rate": 0.0001,
          "weight_decay": 0.00001
        }
      },
      "min_model_comparisons": 20,
      "embedding_model_epochs": 10
    }
  },
  "log": {
    "print_every": 1
  }
}
```

Train with:
```bash
python main.py train --spec training_specs/dn_embedding_with_attention.json
```

This will:
1. Train the attention embedding model to learn model representations
2. Use those embeddings in the dense network for routing decisions

### Loading and Using the Model

```python
from src.models.dn_embedding_model import DnEmbeddingModel

# Load trained DnEmbeddingModel (which contains the attention embedding model)
model = DnEmbeddingModel.load("dn-embedding-with-attention")

# Get computed embeddings for all models seen during training
# These come from the attention embedding component
embeddings = model.model_embeddings
# embeddings: dict[str, np.ndarray] - maps model_id to embedding

# Use for routing
from src.data_models.data_models import InputData

input_data = InputData(
    prompts=["What is the capital of France?"],
    model_names=["gpt-4", "claude-3-opus"]
)
predictions = model.predict(input_data)
# predictions contains scores for each model

# The attention embedding model learned to represent models based on their
# behavioral fingerprints, which the dense network then uses for routing
```

## Data Requirements

**Minimum:**
- At least 10-15 distinct models
- At least 20-50 (prompt, response) pairs per model
- Diverse prompt types (reasoning, creative, factual, code)

**Recommended:**
- 30+ distinct models for good separation
- 100+ pairs per model for stable embeddings
- Mix of model capabilities (weak/strong, general/specialized)

## Implementation Details

### Files

- **Base Interface**: `src/models/embedding_model_base.py`
  - `EmbeddingModelBase`: Abstract base class for all embedding models

- **Model**: `src/models/attention_embedding_model.py`
  - `PairEncoder`: Encodes (prompt, response) pairs
  - `SetAggregator`: Aggregates pairs via attention
  - `AttentionEmbeddingModel`: Main model class (implements `EmbeddingModelBase`)

- **Specification**: `src/models/embedding_specs/attention_embedding_spec.py`
  - `AttentionEmbeddingSpec`: Configuration for attention embedding model
  - Used in `DnEmbeddingModel` via `embedding_spec` parameter

- **Preprocessing**: `src/preprocessing/attention_embedding_preprocessor.py`
  - `AttentionEmbeddingPreprocessor`: Handles feature extraction and normalization
  - Caches preprocessed data for efficiency
  - Includes scaler state in preprocessed data

- **Feature Extraction**: `src/preprocessing/feature_extraction.py`
  - `extract_interaction_features()`: Computes interaction features
  - `extract_lexical_features()`: Computes lexical features
  - `extract_structural_features()`: Computes structural features
  - `extract_all_scalar_features()`: Combines all features
  - All functions accept optional `timer` parameter for profiling

- **Scaler**: `src/preprocessing/simple_scaler.py`
  - `SimpleScaler`: Simple standard scaler with state dict support

- **Data Models**: `src/data_models/attention_embedding_types.py`
  - `ProcessedPair`: Single processed pair with features
  - `ModelSetSample`: Set of pairs from one model
  - `ScalerState`: Fitted scaler state for inference
  - `PreprocessedAttentionEmbeddingData`: Full preprocessed dataset with scaler

- **Data Splitting**: `src/utils/data_split.py`
  - `split_attention_embedding_preprocessed_data()`: Splits preprocessed data at model level
  - `ValidationSplit`: Configuration for train/validation split

### Device Support

The model automatically detects and uses CUDA if available, otherwise falls back to CPU.

### Caching

Preprocessing results are cached to disk (in `preprocessed_data/attention_embedding/`) using a content-based hash key. This allows:
- Fast re-training with same data
- Consistent preprocessing across runs
- Efficient experimentation

Cache keys depend on:
- Dataset content (timestamps)
- Embedding model name
- Minimum model comparisons threshold
- Random seed

## Differences from Original Design Document

The implementation follows the high-level architecture from `attention_embedding_model.md` but differs in some details:

1. **No syntactic features**: We do not use spaCy for POS tagging and dependency parsing (would require additional dependencies). Only lexical and structural features are extracted.

2. **No n-gram features**: Character n-grams (TF-IDF) are not implemented. This can be added later if needed.

3. **Simplified loss**: Only supervised contrastive loss is used. The original document suggested optional classification and tier prediction heads, which are not implemented.

4. **Different scalar feature count**: 32 features instead of 43 (due to missing syntactic features).

5. **Training data**: Uses existing comparison data directly, not separate model fingerprinting datasets.

## Future Improvements

Potential enhancements:
1. Add character n-gram TF-IDF features (100-500 dims)
2. Add syntactic features using spaCy (11 dims)
3. Implement classification and tier prediction auxiliary losses
4. Add additional validation metrics (silhouette score, retrieval MRR)
5. Add support for incremental updates (add new models without full retraining)
6. Experiment with different pooling strategies (mean, max, attention variants)
7. Add model embedding visualization (t-SNE, UMAP)
8. Implement early stopping based on validation metrics

## References

1. McGovern et al. (2024) - "Your Large Language Models Are Leaving Fingerprints"
2. Kumarage & Liu (2023) - "Neural Authorship Attribution on Large Language Models"
3. Zaheer et al. (2017) - "Deep Sets"
4. Lee et al. (2019) - "Set Transformer"
5. Khosla et al. (2020) - "Supervised Contrastive Learning"

