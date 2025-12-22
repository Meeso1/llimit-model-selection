# Triplet-Based Model Behavior Encoders

This document describes the triplet-based model behavior encoders that create meaningful vector representations (embeddings) of large language models based on their behavior, specifically their responses to given prompts.

## 1. Overview

The triplet-based encoders are Stage 1 models in a two-stage training process. They learn to generate embeddings from `(prompt, response)` pairs using triplet margin loss. The goal is to create an embedding space where models with similar performance characteristics (as judged by humans) are located closer to each other.

### Two Implementations

We provide two implementations:

1. **TripletFrozenEncoderModel** (`src/models/triplet_frozen_encoder_model.py`): Uses a frozen sentence transformer for initial embeddings, with trainable dense layers on top. Fast to train, good baseline.

2. **TripletFinetunableEncoderModel** (`src/models/triplet_finetunable_encoder_model.py`): Uses a fine-tunable transformer (e.g., BERT) that is trained end-to-end. More powerful but slower and more resource-intensive.

Both models share a common base class `TripletModelBase` and have identical public interfaces.

### Two-Stage Process

-   **Stage 1 (These Models)**: The encoder is trained to generate embeddings from `(prompt, response)` pairs using triplet margin loss.
-   **Stage 2 (Scoring Model)**: Models like `DenseNetworkModel` use these pre-computed, behavior-based embeddings instead of internal model_id embeddings. This allows scoring any model, including new ones.

The encoders do **not** implement the `ModelBase` interface directly but are self-contained components with their own preprocessing, training, and serialization logic.

## 2. Architecture

### 2.1. TripletFrozenEncoderModel

-   **Frozen Text Encoder**: A sentence transformer model (from `sentence-transformers` library) that is frozen during training. It encodes prompts and responses separately, which are then concatenated.
-   **Trainable Dense Network**: A configurable dense neural network (with LeakyReLU activations and dropout) that transforms the frozen embeddings into a learned embedding space. Architecture specified via `hidden_dims` parameter (e.g., `[256, 128]`).

This design keeps the text encoder frozen (avoiding expensive fine-tuning) while still allowing the model to learn a task-specific embedding space.

### 2.2. TripletFinetunableEncoderModel

-   **Fine-tunable Transformer**: A transformer model (e.g., `bert-base-uncased`) from HuggingFace that is fine-tuned during training. Prompts and responses are concatenated with a `[SEP]` token and encoded together.
-   **Optional Projection Layer**: An optional trainable projection layer (with Tanh activation) that projects the transformer's [CLS] token embedding to a lower dimension.

This design allows the model to adapt the transformer to the specific task, potentially learning better representations at the cost of increased training time and resources.

### 2.3. Common Base Class

Both implementations extend `TripletModelBase` (`src/models/triplet_model_base.py`), which provides:
- Common training loop logic
- Triplet margin loss computation
- KL-divergence regularization
- Validation logic
- Model embedding computation

## 3. Data Flow and Models

Data structures in `src/data_models/triplet_encoder_types.py` handle the data flow:

-   **Input Data**: Training triplets of `(prompt, response)` pairs represented by `TrainingTriplet` dataclass (anchor, positive, negative).
-   **Preprocessed Data**: 
    - For frozen model: `PreprocessedTripletEncoderData` with pre-computed embeddings (`TripletEmbedding`)
    - For fine-tunable model: `PreprocessedTripletEncoderData` with raw text triplets (`TrainingTriplet`)
-   **Output Data**: Fixed-size embedding vectors (`np.ndarray`). The `encode()` method returns embeddings for multiple pairs, while `compute_model_embedding()` returns a single averaged embedding.

## 4. Training Process

### 4.1. Two-Stage Approach

Training must be strictly separated:

1.  **Stage 1**: Train the triplet encoder using triplet margin loss until validation metrics converge. Save the trained encoder.
2.  **Stage 2**: Freeze the encoder. Use it to pre-compute an average embedding for every model. Train the Stage 2 scoring model on these static embeddings.

Jointly training is not viable as it would create an unstable, constantly shifting embedding space.

### 4.2. Triplet Selection Strategy

Training uses `TripletMarginLoss` with strategic triplet selection. Given a pairwise comparison `(prompt, response_A, model_A, response_B, model_B, winner=model_A)`:

-   **Anchor (`A`)**: The winning pair: `(prompt, response_A)`.
-   **Negative (`N`)**: The losing pair from the same comparison: `(prompt, response_B)`. This provides a strong, direct preference signal.
-   **Positive (`P`)**: A hybrid strategy:
    -   **Identity Positive (80% by default)**: A different `(prompt', response')` pair from the same model as the anchor (`model_A`). Teaches the encoder to recognize a model's unique signature.
    -   **Performance Positive (20% by default)**: A winning pair from a different model. Encourages learning a general concept of "good response", pushing winning embeddings from different models closer.

**For tie entries:**
- Tie pairs are used as positives for each other
- Random pairs serve as negatives for tie anchors

The ratio is configurable via `identity_positive_ratio`. All random selections use a seeded RNG for reproducibility.

### 4.3. Validation and Goal Metrics

-   **Validation Split**: Models support validation via the `ValidationSplit` parameter. Currently uses a simple random split. *(Prompt-based splitting not yet implemented)*
-   **Goal Metrics**:
    1.  **Validation Loss**: Primary metric for convergence (includes triplet loss and regularization loss).
    2.  **Triplet Accuracy**: For validation triplet `(A, P, N)`, checks if `distance(A, P) < distance(A, N)`. Intuitive measure of embedding space structure.

### 4.4. Training Length

Training is defined by a fixed number of **epochs**. Monitor validation triplet accuracy and loss to identify the best-performing epoch and prevent overfitting.

## 5. Data Considerations

### 5.1. Preprocessing and Caching

Two preprocessors handle the different model types:

-   **TripletFrozenEncoderPreprocessor** (`src/preprocessing/triplet_frozen_encoder_preprocessor.py`): Pre-computes embeddings using the frozen sentence transformer and caches them.
-   **TripletFinetunableEncoderPreprocessor** (`src/preprocessing/triplet_finetunable_encoder_preprocessor.py`): Constructs text-based triplets without pre-computing embeddings (the model tokenizes on-the-fly).

Both preprocessors:
-   Read raw `TrainingData`
-   Apply filtering rules
-   Construct training triplets based on the defined strategy
-   Use a `Jar` to cache preprocessed data based on a hash of the dataset and parameters

### 5.2. Data Filtering Rules

Before triplet construction, raw data is filtered:

-   **Filter Rare Models**: Models appearing in fewer than a threshold number of comparisons (default: 20, configurable via `min_model_comparisons`) are excluded.
-   **Filter Invalid Entries**: Entries with empty prompts or responses are discarded.
-   **Filter both_bad**: Currently not used in training.

## 6. API and Usage

### 6.1. Initialization

Both models share the same interface. Example with TripletFrozenEncoderModel:

```python
from src.models.triplet_frozen_encoder_model import TripletFrozenEncoderModel
from src.models.optimizers.adamw_spec import AdamWSpec

encoder = TripletFrozenEncoderModel(
    encoder_model_name="all-MiniLM-L6-v2",  # Sentence transformer model (frozen)
    hidden_dims=[256, 128],  # Trainable dense layer dimensions
    optimizer_spec=AdamWSpec(learning_rate=0.001),
    triplet_margin=0.2,
    regularization_weight=0.01,
    min_model_comparisons=20,
    identity_positive_ratio=0.8,
    preprocessor_seed=42,
    print_every=1,
)
```

Example with TripletFinetunableEncoderModel:

```python
from src.models.triplet_finetunable_encoder_model import TripletFinetunableEncoderModel
from src.models.optimizers.adamw_spec import AdamWSpec

encoder = TripletFinetunableEncoderModel(
    transformer_model_name="bert-base-uncased",  # HuggingFace transformer
    projection_dim=128,  # Optional projection (None = use transformer output directly)
    max_length=256,  # Maximum sequence length
    optimizer_spec=AdamWSpec(learning_rate=1e-5),  # Lower LR for fine-tuning
    triplet_margin=0.2,
    regularization_weight=0.01,
    min_model_comparisons=20,
    identity_positive_ratio=0.8,
    preprocessor_seed=42,
    print_every=1,
)
```

### 6.2. Public Method Signatures

Both models expose identical public methods:

```python
from src.data_models.triplet_encoder_types import PromptResponsePair

def train(
    self,
    data: TrainingData,
    validation_split: ValidationSplit | None = None,
    epochs: int = 10,
    batch_size: int = 32
) -> None:
    """Trains the model on the given data."""
    ...

def encode(
    self,
    pairs: list[PromptResponsePair],
) -> np.ndarray: # [n_samples, embedding_dim]
    """Encodes a list of (prompt, response) pairs into embeddings."""
    ...

def compute_model_embedding(
    self,
    pairs: list[PromptResponsePair],
) -> np.ndarray: # [embedding_dim]
    """
    Computes a single, representative embedding for a model by
    averaging the embeddings of its (prompt, response) pairs.
    """
    ...

def get_state_dict(self) -> dict[str, Any]:
    """Returns a serializable state dictionary."""
    ...

@classmethod
def load_state_dict(cls, state_dict: dict[str, Any]) -> "TripletModelBase":
    """Loads a model from a state dictionary."""
    ...
```

### 6.3. Usage Example

```python
from src.data_models.triplet_encoder_types import PromptResponsePair
from src.utils.data_split import ValidationSplit

# Training (same for both models)
encoder.train(
    data=training_data,
    validation_split=ValidationSplit(val_fraction=0.2, seed=42),
    epochs=20,
    batch_size=32,
)

# Encoding pairs
pairs = [
    PromptResponsePair(prompt="What is Python?", response="Python is..."),
    PromptResponsePair(prompt="Explain ML", response="Machine learning is..."),
]
embeddings = encoder.encode(pairs)  # [2, embedding_dim]

# Computing model embedding
model_embedding = encoder.compute_model_embedding(pairs)  # [embedding_dim]

# Serialization
state_dict = encoder.get_state_dict()
# Save to file...

# Loading (use appropriate class)
encoder = TripletFrozenEncoderModel.load_state_dict(state_dict)
```

### 6.4. History and Serialization

-   **Training History**: Models maintain an internal list of `EpochLog` dataclasses containing loss and accuracy metrics for each epoch.
-   **Serialization**: Models are serializable via `get_state_dict` and `load_state_dict`, allowing trained encoders to be saved and loaded independently.

## 7. Advanced Features

### 7.1. Incorporating `tie` Data

When two models tie (`winner="tie"`), their `(prompt, response)` pairs are used as positives for each other. This pulls embeddings of similarly high-performing models closer together.

### 7.2. Regularization Loss

Both models include KL-divergence regularization loss (VAE-style) that encourages the embedding distribution to follow a standard normal distribution. This improves generalization and prevents embedding space collapse.

Total loss:
```
total_loss = triplet_loss + regularization_weight * kl_divergence_loss
```

### 7.3. Model Selection

**Use TripletFrozenEncoderModel when:**
- You want fast training
- You have limited computational resources
- You want a good baseline quickly
- The sentence transformer model is already well-suited for your domain

**Use TripletFinetunableEncoderModel when:**
- You have more computational resources (GPU)
- You want potentially better performance
- Your domain is specialized and could benefit from fine-tuning
- You're willing to invest more training time

### 7.4. Future Enhancements

-   **Compactness Loss**: Additional loss term to encourage embeddings from the same model to be close to their mean.
-   **Prompt-based Validation Split**: Split by `user_prompt` to evaluate generalization to unseen prompts.
-   **Hybrid Approach**: Combine frozen and fine-tunable approaches with separate learning rates.
-   **Contrastive Learning**: Explore other contrastive learning objectives beyond triplet loss.

## 8. Implementation Status

### 8.1. Completed Components

-   **Base Class**: `src/models/triplet_model_base.py` with shared training logic
-   **Frozen Model**: `src/models/triplet_frozen_encoder_model.py`
-   **Fine-tunable Model**: `src/models/triplet_finetunable_encoder_model.py`
-   **Data Types**: `src/data_models/triplet_encoder_types.py`
-   **Preprocessors**: 
    - `src/preprocessing/triplet_frozen_encoder_preprocessor.py`
    - `src/preprocessing/triplet_finetunable_encoder_preprocessor.py`
-   **Documentation**: This document

### 8.2. Pending Work

-   Update `DenseNetworkModel` to accept pre-computed embeddings (Stage 2 integration)
-   Implement prompt-based validation split
-   Add CLI commands for training and using the encoders
-   Empirical comparison of frozen vs fine-tunable models

