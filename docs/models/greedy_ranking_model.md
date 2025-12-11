# Greedy Ranking Model

## Overview

The Greedy Ranking Model finds the best linear ordering (ranking) of LLM models based on pairwise comparison data. Unlike iterative optimization methods (ELO, gradient descent), this model uses a simple, non-iterative greedy algorithm to find a near-optimal ranking that minimizes the number of contradictions in the data.

## Mathematical Foundation

### The Tournament Graph

The comparison data is represented as a weighted directed graph:
- **Nodes**: Each LLM model is a node
- **Edges**: For each pair of models (A, B) with comparisons, we have directed edges
- **Edge weights**: `w(A, B)` = number of times model A won against model B

### The Feedback Arc Set Problem

A perfect score-based ranking corresponds to a linear ordering of models where all edges point "forward" in the ranking (higher-ranked models beat lower-ranked ones). However, the data contains **cycles** (e.g., A beats B, B beats C, C beats A), which cannot be explained by a single score.

The goal is to find the ranking that **minimizes the total weight of backward edges** (edges that contradict the ranking). This is known as the **minimum feedback arc set problem**, which is NP-hard for exact solutions.

### The Greedy Algorithm

We use a simple greedy heuristic that produces near-optimal results:

1. Compute the **net score** for each model: `net_score(i) = Σ_j w(i,j) - Σ_j w(j,i)`
   - This is the total wins minus total losses for model i
2. Sort models by decreasing net score
3. Assign ranks: highest net score → rank 1, etc.

This greedy approach is fast, non-iterative, and empirically produces excellent rankings.

### Parameters
- **Ranking**: A permutation of model indices `[r_1, r_2, ..., r_n]`
  - `r_1` is the highest-ranked model, `r_n` is the lowest
- **Scores**: Derived from ranking position (optional)
  - Can assign scores as `-rank` or use ranking position directly

### Hyperparameters
- `min_model_occurrences`: Minimum number of comparisons a model must have to be included
- `score_normalization`: How to convert rank to a numerical score ("negative_rank", "normalized", "centered")
- `print_summary`: Whether to print a summary after computing ranking

## Computing the Win/Loss Matrix

For each pair of models `(i, j)`, we count the number of wins for each side:

```
W[i, j] = # times model i beat model j (from "model_a_wins" when i=a, or "model_b_wins" when i=b)
```

**Note:** We ignore ties and both_bad outcomes for the ranking computation, as they don't establish a clear winner. These are only used for accuracy calculation.

## Computing Net Scores

For each model `i`, we compute its net score:

```
net_score[i] = Σ_j W[i, j] - Σ_j W[j, i]
```

This is simply: (total wins by model i) - (total losses by model i)

## Constructing the Ranking

The ranking is determined by sorting models by their net scores in descending order:

```
ranking = argsort(net_scores, descending=True)
```

Where:
- `ranking[0]` = model with highest net score (rank 1)
- `ranking[1]` = model with second-highest net score (rank 2)
- etc.

## Converting Ranking to Scores

For compatibility with the model API, we convert the ranking to numerical scores:

```
score[model_i] = -(rank of model_i)
```

This ensures that higher-ranked models have higher (less negative) scores, and models can be compared by their scores.

Alternative scoring schemes:
- Normalized: `score[i] = (n - rank[i]) / n` to get scores in [0, 1]
- Centered: `score[i] = -(rank[i] - (n+1)/2)` to center around 0
- Percentile-based: Use percentile ranks

## Accuracy Metrics

Since this model produces a fixed ranking (not iteratively trained), we compute accuracy metrics once after determining the ranking.

### Ranking Accuracy

For each comparison in the dataset:

1. **Win/Loss comparisons** (model_a_wins / model_b_wins):
   - Correct if the higher-ranked model won
   - `correct if (rank[a] < rank[b] and label='model_a_wins') or (rank[b] < rank[a] and label='model_b_wins')`
   - Note: Lower rank number = higher rank (rank 1 is best)

2. **Tie comparisons**:
   - Correct if both models are in the top half of the ranking
   - `correct if rank[a] <= n/2 and rank[b] <= n/2`

3. **Both_bad comparisons**:
   - Correct if both models are in the bottom half of the ranking
   - `correct if rank[a] > n/2 and rank[b] > n/2`

**Overall accuracy** = (# correct) / (# total comparisons)

**Component accuracies**: Track accuracy separately for win/loss, tie, and both_bad comparisons

### Optimality Metrics

**Minimum disagreements:**
- Number of "backward edges" in the ranking
- A backward edge occurs when a lower-ranked model beat a higher-ranked model
- `disagreements = Σ_{i,j: rank[i] > rank[j]} W[i, j]`
- Lower is better; measures how well the data fits a linear ordering

**Theoretical maximum accuracy:**
- `max_accuracy = (total_comparisons - disagreements) / total_comparisons`
- This is the best possible accuracy any score-based model can achieve on this data

### Ranking Statistics

**Rank distribution:**
- Top-ranked models (top 10%)
- Middle-ranked models
- Bottom-ranked models (bottom 10%)

**Net score statistics:**
- Mean, std, min, max of net scores
- Shows the spread of model performance

## Training Procedure

This model doesn't require iterative training. The entire "training" process is:

```
1. Preprocess data: filter models by min_model_occurrences
2. Construct win/loss matrix W[i,j] from comparison data
3. Compute net scores: net_score[i] = Σ_j W[i,j] - Σ_j W[j,i]
4. Sort models by net score (descending) to get ranking
5. Convert ranking to scores for API compatibility
6. Compute accuracy metrics on the full dataset
7. Save ranking and scores
```

**Time complexity:** O(n² + n log n) where n is the number of models
- O(n²) to construct the win/loss matrix (we iterate over comparisons, not all pairs)
- O(n log n) to sort models by net score

In practice, since we iterate over comparisons (m total), it's O(m + n log n), which is very fast.

## Advantages Over Iterative Methods (ELO, Gradient Descent)

1. **No hyperparameter tuning**: No learning rate, temperature, or regularization to tune
2. **Deterministic**: Always produces the same result for the same data
3. **Fast**: Single pass through the data, no iteration
4. **Interpretable**: Clear meaning—models ranked by net wins
5. **Near-optimal**: Greedy algorithm gives provably good approximations to the NP-hard optimal solution
6. **No overfitting**: No risk of overfitting since there's no iterative optimization
7. **Theoretically grounded**: Directly related to the minimum feedback arc set problem

## Implementation Notes

- **Win/loss matrix construction**: Iterate through comparisons once, populate W[i,j]
  - Can use sparse matrix if memory is a concern (most pairs won't have comparisons)
  - For dense storage: O(n²) memory, acceptable for hundreds of models
- **Net score computation**: Single vectorized operation: `net_scores = W.sum(axis=1) - W.sum(axis=0)`
- **Sorting**: Use numpy argsort, O(n log n)
- **Validation split**: Not needed since there's no training/overfitting
  - Can still compute metrics on different data subsets for analysis
- **Model persistence**: Store the ranking array and net scores
  - Can reconstruct scores from ranking at load time

## Extensions and Variations

### Weighted Comparisons
Could weight comparisons by recency, confidence, or other factors:
```
W[i, j] = Σ (comparison_weight * indicator(i beat j))
```

### Iterative Refinement
Can apply the greedy algorithm iteratively:
1. Compute initial ranking
2. For each model, try swapping it with neighbors
3. Keep swap if it reduces total disagreements
4. Repeat until no improvement

However, this loses the simplicity and speed advantage of the greedy approach.

### Multi-dimensional Rankings
Instead of a single net score, could compute multiple scores:
- Net wins on "reasoning" prompts
- Net wins on "creative" prompts
- etc.

Then rank by a weighted combination or use Pareto-optimal ordering.

## Similarity to Existing Methods

- **Kemeny-Young method**: This greedy algorithm approximates the Kemeny-Young optimal ranking
- **Feedback arc set**: Directly related to the minimum weight feedback arc set problem in graph theory
- **PageRank**: Similar intuition (ranking based on graph structure), but PageRank uses random walk probabilities
- **Swiss-system tournament**: Similar to how rankings are computed in chess tournaments
- **Condorcet method**: Related to voting theory—finds the Condorcet winner if one exists

## Limitations

1. **No prompt-awareness**: Like SimpleScoringModel and ELO, this model ignores prompt content
2. **Linear ordering assumption**: Assumes all models can be linearly ordered, which may not be true if there are significant intransitivities
3. **Equal weight to all comparisons**: Doesn't distinguish between high-confidence and low-confidence comparisons
4. **No uncertainty estimates**: Produces a point estimate ranking without confidence intervals
5. **Greedy approximation**: May not find the truly optimal ranking (which is NP-hard), but empirically very close

