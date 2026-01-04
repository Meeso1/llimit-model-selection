# Optimizer Specifications

## Overview

The optimizer system provides a flexible way to configure and serialize different optimizer types with their parameters, including learning rate decay. All optimizers are implemented as specifications that can create optimizer instances and be serialized for model saving.

## Architecture

### OptimizerSpecification (Abstract Base)

Base class defining the interface for optimizer specifications. All optimizer specs are Pydantic models for automatic validation and serialization:

- **create_optimizer(model, lr_multipliers=None)**: Creates optimizer instance for a model
  - `model`: The PyTorch module to optimize
  - `lr_multipliers`: Optional dict mapping sub-modules to learning rate multipliers
- **create_optimizer_for_multiple(models, lr_multipliers=None)**: Creates optimizer for multiple models
- **create_scheduler(optimizer)**: Creates LR scheduler if decay is enabled
- **to_dict()**: Serializes to dictionary (uses Pydantic's `model_dump()`)
- **from_serialized(optimizer_name, params)**: Static method to deserialize from dictionary

### Learning Rate Decay

All optimizers support exponential LR decay through the `lr_decay_gamma` parameter:
- If `None`, no decay is applied
- If provided, learning rate is multiplied by gamma after each epoch
- Uses PyTorch's `ExponentialLR` scheduler

Example: `lr_decay_gamma=0.95` means LR decreases by 5% each epoch.

### Per-Module Learning Rate Multipliers

All optimizers support different learning rates for different parts of the model through the `lr_multipliers` parameter. This is useful when different components have different training dynamics (e.g., pre-trained vs. newly initialized).

**How it works:**
- Pass a dictionary mapping module references to learning rate multipliers
- Each module's parameters receive `base_learning_rate × multiplier`
- Parameters not in any specified module use the base learning rate
- All optimizers (Adam, AdamW, Muon) support this feature

**Example:**
```python
# In your model's training code
optimizer = optimizer_spec.create_optimizer(
    model=self.network,
    lr_multipliers={
        self.network.scoring_head: 5.0,  # Scoring head trains 5x faster
        # Can specify multiple modules with different multipliers
    }
)
```

**For Muon optimizer:**
- Multipliers only apply to AdamW parameters (biases, norms, heads)
- Muon parameters (≥2D transformer weights) always use the Muon LR
- This is by design since Muon is specifically tuned for transformer weights

**Typical values:**
- `1.0`: Same learning rate (default for unspecified modules)
- `5.0-10.0`: Newly initialized components train faster (e.g., scoring heads)
- `0.1-0.5`: Components that should train more conservatively

**Why use this:**
- **Pre-trained components** (transformer backbone) often benefit from smaller learning rates
- **Newly initialized components** (scoring heads, adapters) often need larger learning rates to converge
- **Fine-grained control** over different parts of your model's training dynamics

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

# Create optimizer with per-module learning rates
optimizer = optimizer_spec.create_optimizer(
    model=my_model,
    lr_multipliers={
        my_model.output_head: 5.0,  # Output head trains 5x faster
    }
)
```

### MuonSpec

Hybrid Muon+AdamW optimizer designed specifically for transformers. Uses different optimization strategies for different parameter types:
- **torch.optim.Muon** for ≥2D transformer weight matrices (faster convergence)
- **torch.optim.AdamW** for everything else (biases, layer norms, scoring heads)

Automatically detects and separates parameters by type. Uses PyTorch's built-in Muon optimizer (available in PyTorch 2.4+) for transformer weights and AdamW for other parameters.

**Parameters:**
- `learning_rate` (default: 0.02): Learning rate for Muon (transformer weights)
- `adamw_lr` (default: 0.0003): Learning rate for AdamW (other parameters)
- `lr_decay_gamma` (default: None): Exponential decay factor (applied to both optimizers)
- `momentum` (default: 0.95): Momentum factor for Muon
- `nesterov` (default: True): Whether to use Nesterov momentum in Muon
- `weight_decay` (default: 0.01): Weight decay for AdamW
- `betas` (default: (0.9, 0.999)): Beta parameters for AdamW

**Usage:**
```python
from src.models.optimizers.muon_spec import MuonSpec

optimizer_spec = MuonSpec(
    learning_rate=0.02,          # Muon LR for transformer weights
    adamw_lr=0.0003,             # AdamW LR for other params
    lr_decay_gamma=0.98,
)
```

**Requirements:** PyTorch 2.4+ (torch.optim.Muon is built-in)

**Note:** 
- Best suited for transformer-based models with ≥2D weight matrices
- Supports optimizing parameters from multiple models simultaneously
- For parameters without 'transformer' in their name or with <2D, AdamW is automatically used

## Serialization

Optimizer specifications are Pydantic models and can be serialized/deserialized automatically. They are saved as two components:
1. **Optimizer type**: String identifier (e.g., "adam", "adamw", "muon")
2. **Parameters dict**: All optimizer-specific parameters

This allows models to save and restore optimizer configurations.

**Example:**
```python
# Save
state_dict = {
    "optimizer_type": optimizer_spec.optimizer_type,
    "optimizer_params": optimizer_spec.to_dict(),
    # ... other model state ...
}

# Load
from src.models.optimizers.optimizer_spec import OptimizerSpecification

optimizer_spec = OptimizerSpecification.from_serialized(
    state_dict["optimizer_type"],
    state_dict["optimizer_params"],
)
```

The `from_serialized` static method is part of the `OptimizerSpecification` base class and automatically dispatches to the correct optimizer type using Pydantic's `model_validate()`.

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

