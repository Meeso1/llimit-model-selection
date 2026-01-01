# Least Squares Scoring Model

## Overview

The Least Squares Scoring Model computes per-model scores by fitting them to minimize the squared error between score differences and observed win rate differences. Unlike iterative methods (e.g., Bradley-Terry), this uses a closed-form solution via linear algebra.

## Algorithm

### Objective

Find scores `s` that minimize:

```
Σ_{pairs (i,j)} (s_i - s_j - target_ij)²
```

where `target_ij = win_rate_ij - win_rate_ji`

### Key Properties

1. **Independent of comparison counts**: Uses win rates rather than raw counts, so each pair contributes equally regardless of how many times they were compared
2. **Closed-form solution**: No iteration needed - solves a linear system using least squares
3. **Well-differentiated scores**: Unlike MCMF potentials, produces a continuous range of scores

### Implementation

The algorithm:

1. **Build wins matrix**: Count wins for each model pair
2. **Compute win rates**: `win_rate[i,j] = wins[i,j] / (wins[i,j] + wins[j,i])`
3. **Compute targets**: `target[i,j] = win_rate[i,j] - win_rate[j,i]` (antisymmetric)
4. **Build Laplacian system**: 
   - `L[i,i]` = degree of node i (number of models i was compared with)
   - `L[i,j]` = -1 if models i and j were compared, 0 otherwise
5. **Build RHS vector**: `b[i] = Σ_j target[i,j]`
6. **Solve**: Use `np.linalg.lstsq` to solve `L @ s = b` (handles singularity via pseudoinverse)
7. **Normalize**: Scale to mean 0, std 1

### Mathematical Interpretation

The Laplacian system `L @ s = b` arises from the normal equations of the least squares problem. The Laplacian matrix is singular (constant vectors are in its null space), so we use the pseudoinverse to get the minimum-norm solution. Normalization removes this degree of freedom.

## Usage

```python
from src.models.least_squares_scoring_model import LeastSquaresScoringModel
from src.utils.data_split import ValidationSplit

model = LeastSquaresScoringModel(
    min_model_occurrences=1000,
    print_summary=True
)

# Train with validation split
model.train(
    data,
    validation_split=ValidationSplit(val_fraction=0.2, seed=42)
)
scores = model.get_all_model_scores()
```

## Parameters

- `min_model_occurrences`: Minimum number of times a model must appear to be included (default: 1000)
- `print_summary`: Whether to print summary statistics after training (default: True)

## Data Requirements

- Filters out rare models (below `min_model_occurrences`)
- Filters out empty entries, ties, and both_bad comparisons
- Uses only pairwise win/loss data (ignores prompts)

## Metrics

Computes and reports (train / validation if validation split provided):
- **Accuracy**: Fraction of comparisons where higher-scored model won
- **Total squared error**: The objective function value Σ_{pairs (i,j)} (s_i - s_j - target_ij)²
- **Mean squared error**: Average squared error per compared pair
- **RMSE**: Root mean squared error (square root of MSE)
- **Num compared pairs**: Number of model pairs that had at least one comparison
- **Lstsq residual**: Residual from the linear system solver ||L @ s - b||² (train only)
- **Score statistics**: min, 10th percentile, mean, 90th percentile, max

The **total squared error** is the primary metric being minimized by the algorithm on training data.

### Validation

When a validation split is provided:
1. Data is split after filtering (based on `min_model_occurrences`) and fitting the encoder
2. Scores are computed using **only training data**
3. Metrics are evaluated on both training and validation data
4. Validation metrics help assess how well the fitted scores generalize to unseen comparisons

## Comparison to Other Models

### vs MCMF Scoring
- **MCMF**: Uses node potentials from residual graph, can have limited score differentiation
- **Least Squares**: Direct optimization on win rates, better score spread

### vs Bradley-Terry
- **Bradley-Terry**: Iterative maximum likelihood, models probabilities via sigmoid
- **Least Squares**: Closed-form linear solution, directly fits score differences to win rate differences

### vs Elo Scoring
- **Elo**: Sequential updates, order-dependent
- **Least Squares**: Batch solution using all data simultaneously, order-independent

