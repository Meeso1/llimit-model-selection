# CLI Usage

The project provides a command-line interface for training and inference.

## Training

Train a model using a JSON training specification.

### Usage

**From file:**
```bash
python -m src.scripts.cli train --spec-file path/to/spec.json
```

**From stdin:**
```bash
more spec.json | python -m src.scripts.cli train
```

### Training Specification Format

The training specification is a JSON file with the following structure:

```json
{
  "wandb": {
    "project": "project-name",
    "experiment_name": "experiment-name",
    "config_name": "config-name",
    "artifact_name": "artifact-name"
  },
  "model": {
    "name": "model-name",
    "start_state": null,
    "spec": {
      "model_type": "dense_network",
      "embedding_model_name": "all-MiniLM-L6-v2",
      "hidden_dims": [256, 128, 64],
      "model_id_embedding_dim": 32,
      "optimizer": {
        "optimizer_type": "adamw",
        "learning_rate": 0.001,
        "lr_decay_gamma": 0.95,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01
      }
    }
  },
  "data": {
    "dataset": "lmarena_human_preference",
    "max_samples": 10000,
    "valiation_split": 0.2,
    "seed": 42
  },
  "log": {
    "print_every": 10
  },
  "epochs": 10,
  "batch_size": 32
}
```

#### Field Descriptions

**wandb**: Weights & Biases configuration
- `project`: W&B project name
- `experiment_name`: Name for this experiment run
- `config_name`: Configuration name
- `artifact_name`: Name for model artifact in W&B

**model**: Model configuration
- `name`: Name to save the model under (required)
- `start_state`: If provided, loads an existing model with this name as the starting point and ignores `spec` (optional)
- `spec`: Model-specific specification (required if `start_state` is not provided)
  - `model_type`: Type of model - one of:
    - **Scoring models**: `"dense_network"`, `"dn_embedding"`, `"simple_scoring"`, `"elo_scoring"`, `"greedy_ranking"`, `"mcmf_scoring"`, `"least_squares_scoring"`, `"gradient_boosting"`, `"transformer_embedding"`
    - **Length prediction models**: `"dn_embedding_length_prediction"`
  - For `dense_network`:
    - `embedding_model_name`: Sentence transformer model name
    - `hidden_dims`: List of hidden layer sizes
    - `model_id_embedding_dim`: Dimension of model ID embeddings
    - `optimizer`: Optimizer specification
      - `optimizer_type`: One of `"adam"`, `"adamw"`, or `"muon"`
      - `learning_rate`: Learning rate
      - `lr_decay_gamma`: Learning rate decay gamma (optional, can be `null`)
      - Additional optimizer-specific parameters (see `docs/optimizers.md`)
  - For `transformer_embedding`:
    - `transformer_model_name`: HuggingFace transformer model name (e.g., `"sentence-transformers/all-MiniLM-L12-v2"`)
    - `finetuning_spec`: Fine-tuning specification (see below)
    - `hidden_dims`: List of hidden layer sizes for the scoring head
    - `dropout`: Dropout rate (default: 0.2)
    - `max_length`: Maximum sequence length (default: 256)
    - `optimizer`: Optimizer specification
    - `balance_model_samples`: Whether to balance samples by model (default: true)
    - `embedding_spec`: Embedding model specification (see `docs/models.md`)
    - `load_embedding_model_from`: Path to load pre-trained embedding model from (optional)
    - `min_model_comparisons`: Minimum comparisons per model to include (default: 20)
    - `embedding_model_epochs`: Number of epochs to train embedding model (default: 10)
    - `seed`: Random seed (default: 42)
  - See `training_specs/` directory for examples of each model type

**data**: Data configuration
- `dataset`: Dataset to use - one of: `"lmarena_human_preference"` (default), `"chatbot_arena"`, or `"both"` (combines both datasets)
- `max_samples`: Maximum number of samples to use (dataset will be downsampled)
- `valiation_split`: Fraction of data to use for validation (between 0 and 1)
- `seed`: Random seed for reproducibility

**log**: Logging configuration
- `print_every`: Print training progress every N epochs

**epochs**: Number of training epochs

**batch_size**: Batch size for training

#### Fine-tuning Specifications

For `transformer_embedding` models, the `finetuning_spec` field specifies how to fine-tune the transformer:

**LoRA (Low-Rank Adaptation):**
```json
{
  "method": "lora",
  "rank": 16,
  "alpha": 32,
  "dropout": 0.05,
  "target_modules": "auto"
}
```

**QLoRA (Quantized LoRA):**
```json
{
  "method": "qlora",
  "rank": 16,
  "alpha": 32,
  "dropout": 0.05,
  "target_modules": "auto",
  "load_in_4bit": true,
  "bnb_4bit_compute_dtype": "float16",
  "bnb_4bit_quant_type": "nf4"
}
```

**Last Layers:**
```json
{
  "method": "last_layers",
  "num_layers": 2
}
```

**BitFit:**
```json
{
  "method": "bitfit"
}
```

**Full Fine-tuning:**
```json
{
  "method": "full"
}
```

### Example

```json
{
  "wandb": {
    "project": "llm-routing",
    "experiment_name": "dense-net-v1",
    "config_name": "default",
    "artifact_name": "dense-net-model"
  },
  "model": {
    "name": "dense_network_v1",
    "start_state": null,
    "spec": {
      "model_type": "dense_network",
      "embedding_model_name": "all-mpnet-base-v2",
      "hidden_dims": [512, 256, 128],
      "model_id_embedding_dim": 64,
      "optimizer": {
        "optimizer_type": "adamw",
        "learning_rate": 0.0001,
        "lr_decay_gamma": 0.98,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01
      }
    }
  },
  "data": {
    "dataset": "lmarena_human_preference",
    "max_samples": 50000,
    "valiation_split": 0.15,
    "seed": 42
  },
  "log": {
    "print_every": 5
  },
  "epochs": 20,
  "batch_size": 64
}
```

### Transformer Embedding Example (with LoRA)

```json
{
  "data": {
    "dataset": "lmarena_human_preference",
    "max_samples": null,
    "validation_split": 0.2,
    "seed": 42
  },
  "epochs": 20,
  "batch_size": 32,
  "model": {
    "name": "transformer-embedding-lora",
    "spec": {
      "model_type": "transformer_embedding",
      "transformer_model_name": "sentence-transformers/all-MiniLM-L12-v2",
      "finetuning_spec": {
        "method": "lora",
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": "auto"
      },
      "hidden_dims": [256, 128],
      "dropout": 0.2,
      "max_length": 256,
      "optimizer": {
        "optimizer_type": "adamw",
        "learning_rate": 0.0001,
        "weight_decay": 0.01
      },
      "balance_model_samples": true,
      "embedding_spec": {
        "embedding_type": "attention",
        "encoder_model_name": "all-MiniLM-L6-v2",
        "h_emb": 128,
        "h_scalar": 32,
        "h_pair": 128,
        "d_out": 64,
        "pair_mlp_layers": 8,
        "num_attention_heads": 8,
        "dropout": 0.1,
        "temperature": 0.07,
        "pairs_per_model": 64,
        "models_per_batch": 8,
        "embeddings_per_model": 4,
        "optimizer": {
          "optimizer_type": "adamw",
          "learning_rate": 0.0001,
          "weight_decay": 0.995
        }
      },
      "load_embedding_model_from": null,
      "min_model_comparisons": 1000,
      "embedding_model_epochs": 100,
      "seed": 42
    }
  },
  "log": {
    "print_every": 1
  }
}
```

See `training_specs/transformer_embedding_lora.json` and `training_specs/transformer_embedding_lora_test.json` for complete examples.

## Inference

Run inference on a trained model.

### Usage

```bash
python -m src.scripts.cli infer \
  --model dense_network/model-name \
  --models-to-score gpt-3.5-turbo gpt-4 claude-2 \
  --prompts "What is Python?" "Explain machine learning" \
  --batch-size 32 \
  --output-path output.json
```

### Arguments

- `--model`: Type and name of model to use (i.e. `"dense_network/model_name"`)
- `--models-to-score`: List of model names to evaluate (space-separated)
- `--prompts`: List of prompts to evaluate (space-separated)
- `--batch-size`: (Optional) Batch size for inference - defaults to 128
- `--output-path`: (Optional) Path to output JSON file. If not provided, output will be saved to `inference_outputs/` with an auto-generated filename: `{timestamp}_{model_name}.json`

### Output Format

The output is a JSON file with scores for each model:

```json
{
  "gpt-3.5-turbo": [0.5, 0.3],
  "gpt-4": [0.8, 0.7],
  "claude-2": [0.6, 0.5]
}
```

Each model maps to a list of scores, one for each prompt in the order they were provided.

### Example

```bash
python -m src.scripts.cli infer \
  --model dense_network/dense_network_v1 \
  --models-to-score gpt-3.5-turbo gpt-4 claude-2 \
  --prompts "How do I sort a list?" "Explain recursion" \
```

This will create an output file in `inference_outputs/` with a timestamped filename.

