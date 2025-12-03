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
  - `model_type`: Type of model (currently only `"dense_network"`)
  - For `dense_network`:
    - `embedding_model_name`: Sentence transformer model name
    - `hidden_dims`: List of hidden layer sizes
    - `model_id_embedding_dim`: Dimension of model ID embeddings
    - `optimizer`: Optimizer specification
      - `optimizer_type`: One of `"adam"`, `"adamw"`, or `"muon"`
      - `learning_rate`: Learning rate
      - `lr_decay_gamma`: Learning rate decay gamma (optional, can be `null`)
      - Additional optimizer-specific parameters (see `docs/optimizers.md`)

**data**: Data configuration
- `max_samples`: Maximum number of samples to use (dataset will be downsampled)
- `valiation_split`: Fraction of data to use for validation (between 0 and 1)
- `seed`: Random seed for reproducibility

**log**: Logging configuration
- `print_every`: Print training progress every N epochs

**epochs**: Number of training epochs

**batch_size**: Batch size for training

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

## Inference

Run inference on a trained model.

### Usage

```bash
python -m src.scripts.cli infer \
  --model-type dense_network \
  --model-name model-name \
  --models-to-score gpt-3.5-turbo gpt-4 claude-2 \
  --prompts "What is Python?" "Explain machine learning" \
  --batch-size 32 \
  --output-path output.json
```

### Arguments

- `--model-type`: Type of model to use (currently only `"dense_network"`)
- `--model-name`: Name of the saved model to load
- `--models-to-score`: List of model names to evaluate (space-separated)
- `--prompts`: List of prompts to evaluate (space-separated)
- `--batch-size`: Batch size for inference
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
  --model-type dense_network \
  --model-name dense_network_v1 \
  --models-to-score gpt-3.5-turbo gpt-4 claude-2 \
  --prompts "How do I sort a list?" "Explain recursion" \
  --batch-size 16
```

This will create an output file in `inference_outputs/` with a timestamped filename.

