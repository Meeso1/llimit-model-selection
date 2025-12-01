# Matrix Factorization (MF) Model

## Overview
This model learns to predict the fitness of a model for a prompt by projecting both into a shared latent space. It draws inspiration from collaborative filtering (Matrix Factorization) but adapts it to the "cold-start" prompt scenario by using the prompt's semantic embedding.

## Architecture

The model consists of two main branches that combine to form a score.

$$ \text{Score}(p, m) = \langle \mathbf{u}_p, \mathbf{v}_m \rangle + b_m $$

### 1. Prompt Branch (The "User" side)
Since prompts are new at inference time, we cannot learn a static vector for each prompt ID. Instead, we learn a function mapping the pre-computed text embedding to a latent factor.

- **Input**: Prompt Embedding $e_p \in \mathbb{R}^D$ (from Sentence Transformer).
- **Transformation**: A learnable projection.
  $$ \mathbf{u}_p = f(e_p) = W_{proj} e_p $$
  where $W_{proj} \in \mathbb{R}^{K \times D}$ and $K$ is the rank (latent dimension).

### 2. Model Branch (The "Item" side)
Since the set of LLMs is fixed and relatively small, we learn explicit embeddings for them.

- **Model Factors**: Matrix $V \in \mathbb{R}^{N_{models} \times K}$.
  - $\mathbf{v}_m$ is the row corresponding to model $m$.
- **Model Bias**: Vector $b \in \mathbb{R}^{N_{models}}$.
  - $b_m$ captures the global strength of the model (e.g., GPT-4 is generally better than GPT-2).

### 3. Inference Logic (`predict`)
Input: `InputData` with $P$ prompts and $M$ requested models.

1. **Embed Prompts**: Convert prompt strings to embeddings $E \in \mathbb{R}^{P \times D}$.
2. **Project Prompts**: Compute prompt factors $U = E W_{proj}^T$ (Shape: $P \times K$).
3. **Lookup Models**: 
   - Retrieve factor vectors $V_{subset}$ for the requested $M$ models (Shape: $M \times K$).
   - Retrieve biases $b_{subset}$ (Shape: $M$).
4. **Compute Scores**:
   $$ S = U V_{subset}^T + b_{subset} $$
   (Broadcasting: $(P \times K) \cdot (K \times M) + (1 \times M) \rightarrow P \times M$).
5. **Format**: Return as `dict[model_name, np.ndarray[P]]`.

## Adaptation to Project Interfaces

### Input / Output
- **Input**: `InputData` (list of prompts, list of model names).
- **Output**: `OutputData` containing scores.

**Handling Unknown Models**:
- If a model name in `InputData` was not seen during training, we cannot lookup its vector $v_m$.
- **Strategy**: Assign a "default" score (e.g., global mean bias or 0). Log a warning.

### Serialization (`get_state_dict`)
We need to save the learned parameters and the mapping to interpret model names.

**Required State:**
1. **`model_string_encoder`**: The mapping object (or dict) that converts string names (e.g., "gpt-4") to row indices `0..N-1` in the matrices.
2. **`prompt_projection_layer`**: State dict of the linear layer ($W_{proj}, bias_{proj}$).
3. **`model_factors`**: The tensor $V$.
4. **`model_biases`**: The tensor $b$.
5. **`global_mean`** (optional): If we use a global intercept.

## Implementation Details

### Hyperparameters
- **Rank ($K$)**: Latent dimension (e.g., 32).
- **L2 Regularization**: Essential for $V$ and $W_{proj}$ to generalize.
- **Optimizer**: Adam or AdamW.

### Training Loop (`train`)
- Iterate through batches of `EvaluationEntry`.
- Look up indices for `model_a` and `model_b`.
- Compute scores $s_A, s_B$.
- Loss = `BCEWithLogits(s_A - s_B, target_prob)`.
  - Target prob is 1.0 if A wins, 0.0 if B wins.
