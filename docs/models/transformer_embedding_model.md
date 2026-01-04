# Transformer Embedding Model

## Overview

The Transformer Embedding Model fine-tunes a pre-trained transformer (e.g., sentence transformer) to predict which model is best suited for a given prompt. It combines the power of transformer-based text understanding with efficient fine-tuning methods like LoRA.

## Architecture

The model consists of three main components:

1. **Embedding Model**: Learns vector representations for models using triplet loss (can be frozen, finetunable, or attention-based)
2. **Transformer Encoder**: Pre-trained transformer (with optional fine-tuning) that encodes prompts
3. **Scoring Head**: Dense neural network that combines transformer output, prompt features, and model embeddings to produce a score

### Component Details

#### 1. Embedding Model
- Trains embeddings for LLM models (e.g., gpt-4, claude-3, etc.)
- Uses triplet loss or attention mechanisms
- Produces fixed-dimensional vectors representing each model's characteristics
- Can be pre-trained and loaded from a saved state

#### 2. Transformer Encoder
- Pre-trained transformer model (e.g., `sentence-transformers/all-MiniLM-L12-v2`)
- Tokenizes and encodes prompts into dense representations
- **Automatically detects the appropriate pooling method** for the model:
  - **Mean pooling**: Averages all token embeddings (most common for sentence transformers)
  - **CLS token**: Uses the [CLS] token embedding (for BERT-style models)
  - **Last token**: Uses the final token (for decoder-only models)
- Supports various fine-tuning methods:
  - **LoRA**: Low-rank adaptation (most parameter-efficient)
  - **QLoRA**: Quantized LoRA (even more memory efficient)
  - **Last Layers**: Fine-tune only the last N layers
  - **BitFit**: Fine-tune only bias terms
  - **Full**: Fine-tune all parameters

#### 3. Scoring Head
- Configurable multi-layer perceptron (MLP)
- Input: concatenation of `[prompt_embedding, prompt_features, model_embedding]`
- Applies LeakyReLU activation and dropout between layers
- Output: Single scalar score for the prompt-model pair

## Training Process

Training happens in two phases:

### Phase 1: Embedding Model Training
- Trains embeddings for LLM models
- Can be skipped if `load_embedding_model_from` is specified
- Uses triplet loss or attention-based learning
- Filters models by minimum number of comparisons

### Phase 2: Transformer + Scoring Head Training
- Fine-tunes the transformer (if specified) and trains the scoring head
- Uses **margin ranking loss** for pairwise comparison:
  ```
  loss = max(0, -label * (score_a - score_b) + margin)
  ```
  where `label = 1` means model A should win, `-1` means model B should win

## Data Format

### Training
For each prompt, the model processes two models (A and B) that were compared:
- Tokenizes the prompt using the transformer's tokenizer
- Extracts prompt features (length, complexity, etc.)
- Retrieves pre-trained embeddings for both models
- Computes scores for both model A and model B
- Applies margin ranking loss based on which model actually won

### Inference
For each prompt and list of models:
- Tokenizes the prompt
- Extracts prompt features
- For each model:
  - Concatenates `[prompt_embedding, prompt_features, model_embedding]`
  - Passes through the network to get a score
- Returns a dictionary mapping model names to scores

## Fine-tuning Methods

### LoRA (Low-Rank Adaptation)
- Adds trainable low-rank matrices alongside frozen weights
- Very parameter-efficient (0.1-1% of parameters trainable)
- Recommended for most use cases
- Parameters:
  - `rank`: Rank of the low-rank matrices (typically 8-32)
  - `alpha`: Scaling factor (typically 2x rank)
  - `dropout`: LoRA dropout rate
  - `target_modules`: Which modules to apply LoRA to ("auto" for automatic detection)

### QLoRA (Quantized LoRA)
- Quantizes the base model to 4-bit while training LoRA adapters in higher precision
- Most memory-efficient option
- Useful for large models (7B+ parameters)
- Parameters:
  - Same as LoRA, plus:
  - `load_in_4bit`: Use 4-bit quantization
  - `bnb_4bit_compute_dtype`: Computation dtype (e.g., "float16")
  - `bnb_4bit_quant_type`: Quantization type (e.g., "nf4")

### Last Layers
- Only fine-tunes the last N layers of the transformer
- Middle ground between LoRA and full fine-tuning
- Parameters:
  - `num_layers`: Number of final layers to fine-tune

### BitFit
- Only fine-tunes bias terms, freezing all weights
- Extremely parameter-efficient
- Good for small datasets

### Full Fine-tuning
- Fine-tunes all parameters in the transformer
- Most flexible but requires most memory and compute
- Risk of overfitting on small datasets

## Sample Balancing

The model supports weighted sampling to handle imbalanced model representation:
- Computes inverse frequency weights for each model
- For each training pair, assigns weight based on the rarest model
- Uses `WeightedRandomSampler` during training
- Ensures rare models get adequate representation

## Hyperparameters

### Model Architecture
- `transformer_model_name`: HuggingFace model name (e.g., `"sentence-transformers/all-MiniLM-L12-v2"`)
- `hidden_dims`: List of hidden layer sizes for scoring head (e.g., `[256, 128]`)
- `dropout`: Dropout rate for scoring head (default: 0.2)
- `max_length`: Maximum sequence length for tokenization (default: 256)

### Fine-tuning
- `finetuning_spec`: Specification for transformer fine-tuning (see above)

### Optimizer
- `optimizer_spec`: Optimizer configuration (AdamW, Adam, or Muon)
- Typical learning rates:
  - LoRA/QLoRA: 1e-4 to 5e-4
  - Last layers: 1e-5 to 1e-4
  - Full fine-tuning: 1e-5 to 5e-5

### Embedding Model
- `embedding_spec`: Configuration for the model embedding component
- `embedding_model_epochs`: Number of epochs to train the embedding model
- `min_model_comparisons`: Minimum comparisons required to include a model
- `load_embedding_model_from`: Path to pre-trained embedding model (optional)

### Training
- `balance_model_samples`: Whether to use weighted sampling (default: true)
- `seed`: Random seed for reproducibility

## Usage Example

### CLI Training
```bash
python -m src.scripts.cli train --spec-file training_specs/transformer_embedding_lora.json
```

### Python API
```python
from src.models.transformer_embedding_model import TransformerEmbeddingModel
from src.models.finetuning_specs.lora_spec import LoraSpec
from src.models.embedding_specs.frozen_embedding_spec import FrozenEmbeddingSpec
from src.models.optimizers.adamw_spec import AdamWSpec

# Create model
model = TransformerEmbeddingModel(
    transformer_model_name="sentence-transformers/all-MiniLM-L12-v2",
    finetuning_spec=LoraSpec(rank=16, alpha=32, dropout=0.05),
    hidden_dims=[256, 128],
    dropout=0.2,
    max_length=256,
    optimizer_spec=AdamWSpec(learning_rate=0.0001, weight_decay=0.01),
    balance_model_samples=True,
    embedding_spec=FrozenEmbeddingSpec(
        encoder_model_name="all-MiniLM-L6-v2",
        hidden_dims=[128, 64],
        optimizer=AdamWSpec(learning_rate=0.001),
    ),
    min_model_comparisons=1000,
    embedding_model_epochs=10,
    seed=42,
)

# Train
model.train(training_data, validation_split=ValidationSplit(0.2, 42), epochs=20, batch_size=32)

# Save
model.save("my-transformer-model")

# Load
loaded_model = TransformerEmbeddingModel.load("my-transformer-model")

# Predict
scores = loaded_model.predict(input_data, batch_size=32)
```

## Performance Considerations

### Memory Usage
- **LoRA**: Low memory usage (~10-20% more than inference)
- **QLoRA**: Lowest memory usage (4-bit quantization)
- **Full fine-tuning**: High memory usage (full gradients)

### Training Speed
- **LoRA**: Fast (only trains ~1% of parameters)
- **Last layers**: Medium (trains ~10-30% of parameters)
- **Full fine-tuning**: Slow (trains all parameters)

### Pooling Method
The model automatically detects the correct pooling strategy from the transformer's configuration:
- **Mean pooling** is used for most sentence-transformer models (BGE, MiniLM, MPNet, E5)
- **CLS token** is used for classification-trained BERT models
- **Last token** is used for decoder-only/causal models
- Falls back to mean pooling if detection fails (safest default)

This ensures optimal performance across different transformer architectures.

### Recommendations
- Start with LoRA (rank=16, alpha=32) for most tasks
- Use QLoRA if memory is limited
- Use higher ranks (32-64) for more complex tasks
- Use full fine-tuning only with large datasets (>100K samples)
- Prefer sentence-transformer models (e.g., `all-mpnet-base-v2`, `bge-base-en-v1.5`) for best embedding quality

## Validation

The model tracks both training and validation metrics:
- **Loss**: Margin ranking loss
- **Accuracy**: Pairwise accuracy (how often the model correctly predicts the winner)

Validation is performed after each epoch without sample balancing.

## Output

The model outputs a dictionary mapping model names to score arrays:
```python
{
    "gpt-4": array([0.85, 0.92, ...]),      # scores for each prompt
    "claude-3": array([0.78, 0.88, ...]),
    "gpt-3.5-turbo": array([0.65, 0.71, ...]),
}
```

Scores are constrained to [-1, 1] range using `tanh` activation.

## Related Files

- Implementation: `src/models/transformer_embedding_model.py`
- Pooling utilities: `src/utils/transformer_pooling_utils.py`
- Data types: `src/data_models/transformer_embedding_types.py`
- Preprocessor: `src/preprocessing/transformer_embedding_preprocessor.py`
- Training specs: `training_specs/transformer_embedding_*.json`
- Fine-tuning specs: `src/models/finetuning_specs/`

