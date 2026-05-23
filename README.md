# llimit-model-selection

A research project for learning to select the best LLM for a given prompt. Models are trained on pairwise human preference data and produce per-prompt scores for each candidate model, enabling routing to the model most likely to produce a high-quality response.

In addition to scoring, the system supports **response length prediction** — predicting how many tokens each model will produce for a given prompt.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## Quick Start

### Training

Write a training specification JSON file and run:

```bash
uv run python -m src.scripts.cli train --spec-file training_specs/my_spec.json
```

Or pipe from stdin:

```bash
cat training_specs/my_spec.json | uv run python -m src.scripts.cli train
```

See `training_specs/` for examples. The minimal structure is:

```json
{
  "model": {
    "name": "my-model",
    "spec": {
      "model_type": "dn_embedding",
      "hidden_dims": [256, 128, 64],
      "optimizer": { "optimizer_type": "adamw", "learning_rate": 0.001 }
    }
  },
  "data": { "validation_split": 0.2, "seed": 42 },
  "log": { "run_name": "my-experiment", "print_every": 1 },
  "epochs": 30,
  "batch_size": 128
}
```

### Inference (CLI)

```bash
uv run python -m src.scripts.cli infer \
  --model dn_embedding/my-model \
  --models-to-score gpt-4 claude-3 llama-3 \
  --prompts "Write a Python sort" "Explain quantum computing"
```

### Inspection

```bash
# List training runs
uv run python -m src.scripts.cli list logs

# Inspect a run (config + final metrics as JSON)
uv run python -m src.scripts.cli inspect my-experiment

# List saved models
uv run python -m src.scripts.cli list models
```

### REST API

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker build -f Dockerfile.api -t llimit-api .
docker run -p 8000:8000 llimit-api
```

Swagger UI is available at `http://localhost:8000/docs`.

**POST** `/infer` — run scoring and/or length prediction:

```json
{
  "scoring_model": "dn_embedding/my-model",
  "model_names": ["gpt-4", "claude-3"],
  "prompts": ["Write a poem", "Explain recursion"]
}
```

## Data

Training uses human preference data from HuggingFace:

- `lmarena_human_preference` — [`lmarena-ai/arena-human-preference-140k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k)
- `chatbot_arena` — `lmsys/chatbot_arena_conversations`
- `both` — combines both datasets

Preprocessing is model-specific and results are cached to `preprocessed_data/`.

## Models

### Prompt-based Scoring Models

These models score each `(prompt, model)` pair and can differentiate recommendations based on prompt content.

| Model type | Description |
|---|---|
| `dn_embedding` | Dense network with learned model embeddings |
| `transformer_embedding` | Fine-tuned transformer + scoring head (LoRA, QLoRA, etc.) |
| `response_predictive` | Predicts response representations before scoring (experimental) |
| `gradient_boosting` | XGBoost with model embeddings |

### Prompt-agnostic Baselines

These models learn a single global score per model, ignoring prompt content entirely.

| Model type | Description |
|---|---|
| `simple_scoring` | Learned score per model (neural network) |
| `elo_scoring` | ELO rating system |
| `greedy_ranking` | Net-wins greedy ranking |
| `mcmf_scoring` | Min-Cost-Max-Flow graph optimization |
| `least_squares_scoring` | Closed-form win-rate fitting |

### Length Prediction Models

| Model type | Description |
|---|---|
| `dn_embedding_length_prediction` | Dense network (residual blocks) for token count regression |
| `gb_length_prediction` | XGBoost for token count regression |
| `simple_length_prediction` | Per-model OLS regression (baseline) |

### LLM Embedding Models

Prompt-based scoring models represent each LLM as a learned embedding vector rather than a simple integer ID. This allows the scoring model to understand relationships between LLMs and, crucially, to **score LLMs not seen during training** — as long as a set of their (prompt, response) pairs is available to compute an embedding.

Three embedding approaches are available:

- **Frozen encoder** (`frozen`): Fixed sentence transformer + trainable dense layers, trained with triplet loss
- **Finetunable encoder** (`finetunable`): Fine-tuned transformer (LoRA, last-layers, etc.), trained with triplet loss
- **Attention embedding** (`attention`): Set-aggregation over (prompt, response) pairs via multi-head attention, trained with supervised contrastive loss

Embeddings trained in one model can be reused in another via `load_embedding_model_from: "model_type/model_name"`, avoiding redundant retraining.

## Documentation

Detailed documentation lives in `docs/`:

- **[cli.md](docs/cli.md)** — full CLI reference: training spec format, inference, log inspection
- **[api.md](docs/api.md)** — REST API reference
- **[models.md](docs/models.md)** — all model types with descriptions
- **[models/](docs/models/)** — per-model detailed docs
- **[length_prediction.md](docs/length_prediction.md)** — length prediction architecture and usage
- **[training_logging.md](docs/training_logging.md)** — local training log format and CLI inspection
