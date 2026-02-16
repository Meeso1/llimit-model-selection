# Training Logging

The training logging mechanism provides a local, file-based alternative to wandb for tracking training runs.

## Overview

Training logs are saved using the jar system in the `training_logs/` directory. Each run gets a unique name with an appended timestamp. Multiple training runs with the same base name are all kept (not replaced).

## Components

### TrainingLogger

The main logger class that handles recording training data. The logger is created internally by models with a run name:

```python
# Pass run_name to model constructor
model = SimpleScoringModel(
    run_name="my-experiment",
    # ... other params
)

# Logger is created internally and used automatically during training
```

If `run_name` is `None`, logging is disabled.

**Timestamp Format**: Run names use unix timestamps (from `time.time()`) as suffixes, matching the jar.py format:
- Base name: `my-experiment`
- Full name: `my-experiment-1771244318`
- Separator: `-` (hyphen)

### LogEntry

Represents a single epoch's metrics:
- `data`: Dictionary of metrics
- `timestamp`: When the entry was logged

### TrainingLog

Container for a complete training run:
- `run_name`: Unique name with timestamp
- `config`: Model configuration dictionary
- `epoch_logs`: List of LogEntry objects
- `final_metrics`: Dictionary of final metrics (e.g., test accuracy, model score statistics)
- `start_time`: When training started
- `end_time`: When training finished

### RunInfo

Information about training runs:
- `base_name`: Run name without timestamp
- `timestamps`: List of all timestamps for this run name
- `latest_timestamp`: Most recent timestamp
- `latest_full_name`: Full name of the latest run

## Usage in Models

Models automatically handle logging through the base class:

```python
class MyModel(ModelBase):
    def train(self, data, ...):
        # Initialize logger at start of training
        self.init_logger_if_needed()
        
        for epoch in range(epochs):
            # Train epoch...
            entry = TrainingHistoryEntry(
                epoch=epoch,
                total_loss=loss,
                train_accuracy=acc,
                # ...
            )
            
            # Log epoch data
            self.log_epoch(entry)
        
        # Log final metrics and finish
        from src.utils.model_scores_stats import compute_model_scores_stats
        
        model_scores = self.get_all_model_scores()
        final_metrics = {
            "model_scores_stats": compute_model_scores_stats(model_scores),
            # Add any other final metrics
        }
        self.finish_logger_if_needed(final_metrics=final_metrics)
```

### Model Score Statistics

Instead of logging full model scores dictionaries, we log summary statistics:

```python
from src.utils.model_scores_stats import compute_model_scores_stats

scores = {"model_a": 1.5, "model_b": 0.8, "model_c": -0.3}
stats = compute_model_scores_stats(scores)
# Returns: {
#     "mean": 0.67,
#     "median": 0.8,
#     "std": 0.74,
#     "min": -0.3,
#     "max": 1.5,
#     "top10%": 1.5,  # 90th percentile
#     "top90%": -0.3,  # 10th percentile  
#     "best_model_name": "model_a",
#     "worst_model_name": "model_c"
# }
```

## Loading Logs

```python
from src.utils.training_logger import load_training_log, list_training_runs

# List all available runs (grouped by base name)
runs = list_training_runs()  # Returns list[RunInfo]
for run_info in runs:
    print(f"{run_info.base_name}: {len(run_info.timestamps)} runs")
    print(f"  Latest: {run_info.latest_timestamp}")

# Load by base name (gets latest run)
log = load_training_log("my-experiment")

# Load specific run by base name + timestamp
log = load_training_log("my-experiment", timestamp=1771244318)
# or with string timestamp:
log = load_training_log("my-experiment", timestamp="1771244318")

# Access data
print(log.config)
print(log.epoch_logs)
print(log.final_metrics)
```

## Storage

- Logs are stored in `training_logs/` by default
- Each run is saved as a pickle file with unix timestamp suffix: `{base_name}-{timestamp}.pkl`
  - Example: `my-experiment-1771244318.pkl`
- Multiple runs with the same base name are kept (all versions saved)
- Uses the `Jar.replace()` method which saves the new file, then removes all but the newest with the same full name
- Timestamp separator is `-` (hyphen), consistent with jar.py format

## Migration from Wandb

The logging mechanism replaces wandb with a similar interface:

| Wandb | New Logging |
|-------|-------------|
| `wandb.init()` | `init_logger_if_needed()` (called automatically in train()) |
| `wandb.log()` | `log_epoch()` |
| `wandb.finish()` | `finish_logger_if_needed(final_metrics=...)` |
| `wandb_details: WandbDetails \| None` | `run_name: str \| None` (in constructor) |
| `get_config_for_wandb()` | `get_config_for_logging()` |

### Training Spec Changes

In training spec JSON files:

```json
{
  "log": {
    "run_name": "my-experiment-default",
    "print_every": 10
  }
}
```

The `wandb` field has been removed, and `run_name` is now part of the `log` configuration.

## Final Metrics

Models should log important final metrics when finishing:

```python
from src.utils.model_scores_stats import compute_model_scores_stats

model_scores = self.get_all_model_scores()
final_metrics = {
    "model_scores_stats": compute_model_scores_stats(model_scores),
    "best_val_accuracy": best_accuracy,
    "best_epoch": best_epoch,
    # Model-specific metrics
}
self.finish_logger_if_needed(final_metrics=final_metrics)
```

This is especially important for non-iterative models (like `GreedyRankingModel`, `McmfScoringModel`, `LeastSquaresScoringModel`) that don't have epoch logs but still want to save their configuration and final results.

## Non-Epoch Models

Models that don't use iterative training (no epochs) can still use logging to save:
- Initial configuration (via `get_config_for_logging()`)
- Final metrics including model score statistics
- Any other relevant results

These models call `init_logger_if_needed()` at the start of training and `finish_logger_if_needed(final_metrics=...)` at the end, just like iterative models, but don't call `log_epoch()`.
