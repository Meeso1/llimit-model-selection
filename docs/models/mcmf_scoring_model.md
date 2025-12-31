# MCMF Scoring Model

## Overview

The `McmfScoringModel` is a model that learns scores for each LLM using a **Min-Cost-Max-Flow (MCMF)** algorithm, without looking at prompts or responses. It formulates the scoring problem as a network flow optimization problem where model scores are derived from the optimal flow's node potentials.

## Architecture

The model uses a graph-based optimization approach:
- **Network Flow**: Constructs a directed graph with source, sink, and model nodes
- **No neural network**: Scores are computed via optimization, not learned via gradient descent
- **No prompt embeddings**: Completely ignores prompt content
- **No response embeddings**: Completely ignores response content
- **Preprocessor**: `SimpleScoringPreprocessor` extracts model comparisons and encodes model names
- **Non-iterative**: Computes scores in a single pass (ignores epochs parameter)

This model treats the scoring problem as finding an optimal flow through a network that respects the win/loss structure of the comparison data.

## Min-Cost-Max-Flow Formulation

### Network Construction

The MCMF network is constructed as follows:

**Nodes:**
- **Source node**: Starting point for flow
- **Sink node**: Ending point for flow
- **Model nodes**: One node for each LLM

**Edges:**
1. **Source → Model_i**: 
   - Capacity = total number of wins for model i
   - Cost = 0
   
2. **Model_i → Sink**:
   - Capacity = total number of losses for model i
   - Cost = 0
   
3. **Model_i → Model_j**:
   - Capacity = number of times model i beat model j
   - Cost = 1

### Optimization Objective

The algorithm finds the **maximum flow** from source to sink with **minimum total cost**, where:
- **Flow**: Represents the routing of wins/losses through the network
- **Cost**: Penalizes flow through model-to-model edges (cost = 1)
- **Node potentials**: Dual variables from the optimization provide model scores

### Scoring from Node Potentials

After solving the MCMF problem, scores are computed from the flow pattern:
- Models with higher inflow (more wins) get higher scores
- Models with higher outflow (more losses) get lower scores
- The balance of flow through each model node determines its score
- Scores reflect both direct win/loss counts and the structure of pairwise comparisons

## Comparison Types

The model handles only **ranking comparisons**:

1. **model_a_wins**: Model A beat Model B
   - Contributes to A's wins and B's losses
   - Creates a directed edge A → B in the comparison graph
   
2. **model_b_wins**: Model B beat Model A
   - Contributes to B's wins and A's losses
   - Creates a directed edge B → A in the comparison graph

**Ignored comparison types:**
- **tie**: Ignored (no clear winner/loser)
- **both_bad**: Ignored (no positive outcome to model)

This simplified handling keeps the network flow formulation clean and focused on clear ranking relationships.

## Training

### Preprocessing
1. **SimpleScoringPreprocessor** extracts all comparisons from training data
2. Creates a `StringEncoder` to map model names to integer IDs
3. Filters models that appear fewer than `min_model_occurrences` times
4. Returns `PreprocessedTrainingData` with encoded comparisons

### Training Process (Non-iterative)

Unlike iterative models, MCMF scoring computes scores in a single optimization:

1. **Initialization**: Preprocesses data and creates model encoder
2. **Filtering**: Filters comparisons to keep only `model_a_wins` and `model_b_wins`
3. **Graph Construction**: Builds MCMF network as described above
4. **Optimization**: Solves min-cost-max-flow using NetworkX's algorithm
5. **Score Extraction**: Derives scores from the optimal flow's node potentials
6. **Metrics**: Computes accuracy and other statistics on the training data

**Note:** The `epochs` parameter is ignored. If `train()` is called multiple times, scores are recomputed from scratch each time (in case data has changed).

### Network Flow Metrics

The model tracks flow-specific metrics:
- **Total flow**: Amount of flow from source to sink (should equal total wins)
- **Flow cost**: Total cost of the optimal flow (lower = more consistent with data)
- **Accuracy**: Fraction of comparisons where higher-scored model won

## Prediction

During prediction, the model:
1. Returns the same scores for all models regardless of prompt (prompt-agnostic)
2. Scores are based purely on the MCMF-derived values
3. Unknown models (not seen during training) receive a score of 0

This makes predictions very fast but unable to adapt to different prompt types.

## API

### Initialization
```python
from src.models.mcmf_scoring_model import McmfScoringModel

model = McmfScoringModel(
    min_model_occurrences=1000,  # Minimum times a model must appear
    print_summary=True,  # Print summary after computing scores
    wandb_details=None,
)
```

### Training
```python
# Note: epochs parameter is ignored (model is non-iterative)
model.train(
    data=training_data,  # TrainingData with EvaluationEntry objects
    validation_split=None,  # Validation not used (ignored)
    epochs=0,  # Should be 0 (ignored if non-zero)
    batch_size=32,  # Not used (kept for API consistency)
)
```

**Warning messages:**
- If `epochs > 0`: Warns that epochs parameter is ignored
- If `validation_split.val_fraction > 0`: Warns that validation is not used

### Prediction
```python
from src.data_models.data_models import InputData

input_data = InputData(
    prompts=["prompt1", "prompt2"],
    model_names=["model_1", "model_2", "model_3"],
)

output = model.predict(input_data)
# output.scores: dict[str, np.ndarray] - model_name -> scores [n_prompts]
```

### Getting Model Scores
```python
# Get scores for all known models
all_scores = model.get_all_model_scores()
# Returns: dict[str, float] mapping model names to scores

for model_name, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name}: {score:.2f}")
```

## Parameters

### Constructor
- `min_model_occurrences`: Minimum number of times a model must appear to be included (default: 1000)
  - Filters out models with insufficient data
  - Higher values = more stable but fewer models
  - Lower values = more models but potentially noisy scores
  
- `print_summary`: Whether to print a summary after computing scores (default: True)
  - Displays number of models, comparisons, accuracy, and flow metrics
  
- `wandb_details`: Weights & Biases configuration (default: None)
  - Optional integration for experiment tracking

### Train method
- `data`: TrainingData with pairwise comparisons
- `validation_split`: Not used (kept for API compatibility)
- `epochs`: Not used (kept for API compatibility, should be 0)
- `batch_size`: Not used (kept for API compatibility)

## State Persistence

The model can be saved and loaded:

```python
# Save
state_dict = model.get_state_dict()

# Load
model = McmfScoringModel.load_state_dict(state_dict)
```

The state includes:
- Model scores (MCMF-derived scores for all models)
- Model encoder (mapping from model names to IDs)
- Flow metrics (total flow, flow cost)
- Model configuration

## Training Output Example

```
================================================================================
MCMF Scoring Model Summary
================================================================================
Number of models: 89
Number of comparisons: 45678
Accuracy: 67.84%
Total flow: 23456
Flow cost: 12345
Score range: [-156.23, 234.56]
Average score: 0.45
================================================================================
```

## Metrics

The model computes several metrics:

- **accuracy**: Fraction of comparisons where the higher-scored model won
- **n_comparisons**: Number of ranking comparisons used (excludes ties and both_bad)
- **n_correct**: Number of correctly predicted comparisons
- **flow_cost**: Total cost of the min-cost-max-flow solution
- **total_flow**: Total amount of flow from source to sink
- **avg_score**: Average score across all models
- **max_score**: Highest model score
- **min_score**: Lowest model score

## Comparison with Other Scoring Models

### McmfScoringModel
- **Algorithm**: Min-cost-max-flow optimization
- **Training**: Single optimization pass (non-iterative)
- **Convergence**: Instant (one-shot computation)
- **Interpretability**: Scores reflect network flow structure
- **Speed**: Fast (optimization, no iteration)
- **Handling**: Only uses clear win/loss comparisons (ignores ties/both_bad)

### SimpleScoringModel
- **Algorithm**: Neural network with margin ranking loss
- **Training**: Iterative backpropagation with PyTorch
- **Convergence**: Multiple epochs required
- **Interpretability**: Learned scores (abstract)
- **Speed**: Fast but iterative (requires GPU)
- **Handling**: Uses all comparison types (wins/losses/ties/both_bad)

### EloScoringModel
- **Algorithm**: ELO rating system
- **Training**: Iterative rating updates
- **Convergence**: Multiple epochs to stabilize
- **Interpretability**: Standard ELO scale (like chess)
- **Speed**: Very fast (no backpropagation)
- **Handling**: Uses all comparison types with penalties

### GreedyRankingModel
- **Algorithm**: Greedy ranking (net wins)
- **Training**: Single pass (non-iterative)
- **Convergence**: Instant (one-shot computation)
- **Interpretability**: Rank-based scores
- **Speed**: Very fast (simple algorithm)
- **Handling**: Only uses clear win/loss comparisons

All models:
- Ignore prompts (prompt-agnostic)
- Learn a single score per model
- Use same preprocessor

## Use Case

This model is primarily useful as:

1. **Baseline**: Compare against other scoring approaches
2. **Graph-theoretic Approach**: Leverages network flow theory for scoring
3. **Research Tool**: Study how optimization-based scoring compares to learning or ELO
4. **One-shot Training**: No need to tune epochs or learning rates

**Interpretation of results:**
- Higher scores indicate better-performing models
- Flow cost indicates how well scores fit the comparison structure
- Lower flow cost suggests more consistent comparison data
- Scores are relative (only ordering matters)

**Advantages:**
- Non-iterative (instant convergence)
- Theoretically grounded (min-cost-max-flow is well-studied)
- No hyperparameter tuning (no learning rate, no epochs)
- Deterministic results
- Efficient computation via network flow algorithms

**Limitations:**
- Cannot adapt recommendations to different prompt types
- Ignores tie and both_bad comparisons
- Score interpretation is less intuitive than ELO
- No control over convergence speed (one-shot only)
- Requires complete graph construction in memory

## Implementation Details

### NetworkX Integration

The model uses NetworkX's `max_flow_min_cost` function:
```python
import networkx as nx

G = nx.DiGraph()
# ... add nodes and edges with capacity and weight attributes ...
flow_dict = nx.max_flow_min_cost(G, source, sink)
```

### Score Computation

Scores are derived from the flow pattern:
```python
for model_id in range(num_models):
    node = model_id + 1
    inflow = flow_dict[source].get(node, 0)
    outflow = flow_dict[node].get(sink, 0)
    net_to_others = sum(flow to other models)
    net_from_others = sum(flow from other models)
    
    score = inflow - outflow + net_from_others - net_to_others
```

This heuristic captures:
- **Inflow from source**: Models with more wins
- **Outflow to sink**: Models with more losses (penalized)
- **Net flow to/from other models**: Structural position in comparison graph

## Parameter Tuning Tips

### min_model_occurrences
- **Start with 1000** (filters noisy models with few comparisons)
- **Increase** if you have many models and want only well-represented ones
- **Decrease** if you have few models or want to include more data

### print_summary
- **Set to True** for interactive debugging and analysis
- **Set to False** for production or batch processing

## Theoretical Background

### Min-Cost-Max-Flow
- **Maximum flow**: Find the largest amount of flow from source to sink
- **Minimum cost**: Among all maximum flows, choose the one with minimum total cost
- **Dual solution**: Node potentials form the dual of the linear program
- **Applications**: Network routing, assignment problems, matching

### Node Potentials
In the dual formulation of min-cost flow:
- Each node has a potential (price)
- Reduced costs (edge costs adjusted by potentials) satisfy optimality conditions
- Potentials represent "value" of being at each node
- Higher potentials indicate more valuable positions in the flow network

### Connection to Ranking
- Source represents "winning pool"
- Sink represents "losing pool"
- Models act as intermediaries routing wins to losses
- Models that route more efficiently (lower cost paths) get higher scores
- The network structure captures both direct wins and transitive relationships

