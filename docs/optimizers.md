# Optimizer Specifications

## Overview

The optimizer system provides a flexible way to configure and serialize different optimizer types with their parameters, including learning rate decay. All optimizers are implemented as specifications that can create optimizer instances and be serialized for model saving.

## Architecture

### OptimizerSpecification (Abstract Base)

Base class defining the interface for optimizer specifications:

- **create_optimizer(model)**: Creates optimizer instance for a model
- **create_scheduler(optimizer)**: Creates LR scheduler if decay is enabled
- **get_optimizer_name()**: Returns string identifier for serialization
- **to_dict()**: Serializes to dictionary
- **from_dict(params)**: Deserializes from dictionary

### Learning Rate Decay

All optimizers support exponential LR decay through the `lr_decay_gamma` parameter:
- If `None`, no decay is applied
- If provided, learning rate is multiplied by gamma after each epoch
- Uses PyTorch's `ExponentialLR` scheduler

Example: `lr_decay_gamma=0.95` means LR decreases by 5% each epoch.

## Available Optimizers

### AdamSpec

Standard Adam optimizer with optional weight decay.

**Parameters:**
- `learning_rate` (default: 0.001): Base learning rate
- `lr_decay_gamma` (default: None): Exponential decay factor
- `betas` (default: (0.9, 0.999)): Coefficients for running averages
- `eps` (default: 1e-8): Numerical stability term
- `weight_decay` (default: 0.0): L2 penalty coefficient

**Usage:**
```python
from src.models.optimizers.adam_spec import AdamSpec

optimizer_spec = AdamSpec(
    learning_rate=0.001,
    lr_decay_gamma=0.95,
    weight_decay=0.01,
)
```

### AdamWSpec

AdamW optimizer with decoupled weight decay (recommended for most tasks).

**Parameters:**
- `learning_rate` (default: 0.001): Base learning rate
- `lr_decay_gamma` (default: None): Exponential decay factor
- `betas` (default: (0.9, 0.999)): Coefficients for running averages
- `eps` (default: 1e-8): Numerical stability term
- `weight_decay` (default: 0.01): Decoupled weight decay coefficient

**Usage:**
```python
from src.models.optimizers.adamw_spec import AdamWSpec

optimizer_spec = AdamWSpec(
    learning_rate=0.001,
    lr_decay_gamma=0.95,
)
```

### MuonSpec

Muon optimizer (newer optimizer with momentum).

**Parameters:**
- `learning_rate` (default: 0.02): Base learning rate
- `lr_decay_gamma` (default: None): Exponential decay factor
- `momentum` (default: 0.95): Momentum factor
- `nesterov` (default: True): Whether to use Nesterov momentum

**Usage:**
```python
from src.models.optimizers.muon_spec import MuonSpec

optimizer_spec = MuonSpec(
    learning_rate=0.02,
    lr_decay_gamma=0.98,
)
```

## Serialization

Optimizer specifications are serialized as two components:
1. **Optimizer name**: String identifier (e.g., "adam", "adamw", "muon")
2. **Parameters dict**: All optimizer-specific parameters

This allows models to save and restore optimizer configurations.

**Example:**
```python
# Save
state_dict = {
    "optimizer_name": optimizer_spec.get_optimizer_name(),
    "optimizer_params": optimizer_spec.to_dict(),
    # ... other model state ...
}

# Load
from src.models.optimizers.optimizer_spec import OptimizerSpecification

optimizer_spec = OptimizerSpecification.from_serialized(
    state_dict["optimizer_name"],
    state_dict["optimizer_params"],
)
```

The `from_serialized` static method is part of the `OptimizerSpecification` base class and automatically dispatches to the correct optimizer type.

## Integration with DenseNetworkModel

The `DenseNetworkModel` accepts an `optimizer_spec` parameter:

```python
from src.models.dense_network_model import DenseNetworkModel
from src.models.optimizers.adamw_spec import AdamWSpec

model = DenseNetworkModel(
    optimizer_spec=AdamWSpec(
        learning_rate=0.001,
        lr_decay_gamma=0.95,
    ),
)
```

If not provided, defaults to `AdamWSpec(learning_rate=0.001)`.

## Location

Optimizer specifications are located in:
- `src/models/optimizers/optimizer_spec.py` - Abstract base class (includes factory method)
- `src/models/optimizers/adam_spec.py` - Adam implementation
- `src/models/optimizers/adamw_spec.py` - AdamW implementation
- `src/models/optimizers/muon_spec.py` - Muon implementation

