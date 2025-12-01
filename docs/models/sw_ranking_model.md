# Similarity-Weighted (SW) Ranking Model

## Overview
This model implements a non-parametric, instance-based approach to model selection. Instead of learning a fixed set of weights, it predicts the "fitness" (score) of a model for a given prompt by looking at how that model (and its competitors) performed on semantically similar prompts in the training data.

This is effectively a **"Local Elo"** or **"Local Bradley-Terry"** model.

## Architecture & Logic

### 1. Data Structures (The "Model" State)
Since this is a lazy learner, the "training" phase primarily involves building an index.
- **Reference Prompts**: A store of embeddings for all valid training prompts.
- **Reference Outcomes**: A mapping or graph storing the outcomes of comparisons associated with those prompts (e.g., "On prompt #42, GPT-4 beat Llama-2").
- **Global Scores (Fallback)**: Pre-computed global Elo ratings for all models to handle cold-start or sparse neighborhoods.

### 2. Inference Logic (`predict`)
Input: $P$ prompts, $M$ candidate models.
Output: Score matrix $S \in \mathbb{R}^{|P| \times |M|}$ (returned as dictionary).

For each query prompt $x$:
1. **Retrieval**: Find the set $\mathcal{N}_k(x)$ of $k$ nearest neighbors in the Reference Prompts using cosine similarity.
2. **Weighting**: Compute weight $\alpha_i$ for each neighbor $i$ based on similarity.
3. **Local Estimation**:
   - Construct a mini-dataset of pairwise comparisons found in the neighborhood $\mathcal{N}_k(x)$.
   - Filter for comparisons involving models relevant to the global set (or specifically the requested $M$).
   - Fit a weighted Bradley-Terry model (or compute weighted win-rates) to solve for local scores $s_{local}$.
4. **Fallback / Smoothing**:
   - If a model $m \in M$ is not present in the neighborhood, use its global score.
   - Blend local and global scores: $s_{final} = \lambda s_{local} + (1-\lambda) s_{global}$.

## Adaptation to Project Interfaces

### Input / Output
- **Input**: `InputData(prompts=[...], model_names=[...])`
- **Output**: `OutputData(scores={model_name: array_of_scores})`

The model must be able to produce a score for *any* requested model name.
- If `model_name` is known but not in the local neighborhood -> Return Global Score.
- If `model_name` is completely unknown (never seen in training) -> Return 0.0 (or raise error / handle gracefully).

### Serialization (`get_state_dict`)
Unlike parametric models, we must save the reference data.

**Required State:**
1. **`prompt_embeddings`**: Matrix of shape `[N_train, D_emb]` (or a FAISS index structure).
2. **`training_history`**: The raw interaction data. List of `(idx_prompt, model_A_id, model_B_id, winner_id)`.
3. **`model_string_encoder`**: Mapping from string names to integer IDs used in the history.
4. **`global_scores`**: Vector/Dict of pre-computed global ratings.
5. **Hyperparameters**: `k` (neighbors), `tau` (temperature), `lambda` (smoothing).

**Note on Size**: If the training set is massive (140k+ rows), saving the raw embeddings might be heavy (~400MB for 140k * 768 floats). We should ensure efficient storage (e.g., `numpy` arrays or `torch` tensors) rather than raw lists.

## Implementation Steps
1. **`train`**:
   - Flatten `TrainingData` into arrays.
   - Compute embeddings (using `PromptEmbeddingPreprocessor`).
   - Build the search index (e.g., exact search via `torch.matmul` for GPU acceleration, or FAISS).
   - Compute global Elo scores for fallback.
2. **`predict`**:
   - Encode input prompts.
   - Query index for indices and distances.
   - vectorized lookup of history.
   - Compute scores (vectorized logic preferred over per-item loops for speed).
3. **`save/load`**: 
   - Dump the tensor arrays to the jar.
