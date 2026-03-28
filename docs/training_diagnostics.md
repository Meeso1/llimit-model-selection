# Training Diagnostics

**File**: `src/analysis/training_diagnostics.py`

Provides `EpochDiagnosticsAccumulator`, a model-agnostic accumulator for per-batch training diagnostics.  It replaces the private `_EpochMetricsAccumulator` that previously existed inside `TransformerEmbeddingModel` and adds new Group 1 interpretability metrics (gradient × input attribution, score variance decomposition, representation collapse detection).

---

## Usage pattern

```python
from src.analysis.training_diagnostics import EpochDiagnosticsAccumulator, split_tensor_with_grad

# start of _train_epoch
diag = EpochDiagnosticsAccumulator()
feature_names = [...]  # one name per column of prompt_features

for batch in dataloader:
    prompt_features = batch["prompt_features"].to(device).requires_grad_(True)  # [batch, n_features]
    model_emb_a = batch["model_emb_a"].to(device).requires_grad_(True)          # leaf

    optimizer.zero_grad()
    prompt_emb = encoder(...)
    prompt_emb.retain_grad()  # non-leaf: needs retain_grad() to capture grad
    scores_a = network(prompt_emb, prompt_features, model_emb_a)
    scores_b = network(prompt_emb, prompt_features, model_emb_b)

    # representation stats (no_grad, before backward)
    with torch.no_grad():
        diag.update_representation_stats({"prompt_proj": proj_a, "model_proj": proj_b})

    # --- backward ---
    loss = compute_loss(scores_a, scores_b, labels)
    loss.backward()

    # diagnostic updates that need gradients (after backward)
    diag.update_grad_norms({"encoder": enc_params, "scoring_head": head_params})
    diag.update_gradient_attribution({
        **split_tensor_with_grad(prompt_features, feature_names),  # 45 individually named scalar features
        "prompt_embedding": prompt_emb,   # whole embedding → single aggregated key
        "model_embedding": model_emb_a,   # whole embedding → single aggregated key
    })
    diag.update_score_variance(
        torch.cat([scores_a.detach(), scores_b.detach()]),
        torch.cat([model_ids_a, model_ids_b]),
    )

    clip_grad_norm_(...)
    optimizer.step()

# end of epoch
additional_metrics = diag.to_dict()  # merge into TrainingHistoryEntry.additional_metrics
```

Each `update_*` method is optional; only groups that are called at least once contribute keys to `to_dict()`.

---

## `split_tensor_with_grad`

```python
def split_tensor_with_grad(
    tensor: torch.Tensor,   # [batch_size, n_features], requires_grad=True, grad populated
    feature_names: list[str],
) -> dict[str, torch.Tensor]  # name -> [batch_size, 1], detached leaf with .grad set
```

Call after `loss.backward()`. Splits both the tensor values and its `.grad` into individually-named single-column leaf tensors, assigning each column's gradient manually. Returns an empty dict if `tensor.grad` is `None`. Pass the result (along with any other inputs) to `update_gradient_attribution`.

---

## `EpochDiagnosticsAccumulator`

### Constructor

```python
EpochDiagnosticsAccumulator()
```

---

### `update_grad_norms`

```python
def update_grad_norms(
    self,
    param_groups: dict[str, Iterable[nn.Parameter]],
) -> None
```

Computes the total L2 gradient norm for each named parameter group.  Parameters with `grad is None` are skipped (handles frozen layers).

**Call after `loss.backward()`, before `clip_grad_norm_`.**

**`to_dict` keys**: `{group_name}_grad_norm` for each group.

---

### `update_representation_stats`

```python
def update_representation_stats(
    self,
    representations: dict[str, torch.Tensor],  # name -> [batch_size, dim]
) -> None
```

For each named tensor, records:
- mean L2 norm across the batch: `repr.norm(dim=1).mean()`
- mean per-dimension variance across the batch: `repr.var(dim=0).mean()`

Low variance signals representation collapse; near-zero norms signal dead projections.

**`to_dict` keys**: `{name}_norm`, `{name}_variance` for each name.

**Note**: Run inside a `torch.no_grad()` context when calling from the training loop.

---

### `update_score_variance`

```python
def update_score_variance(
    self,
    scores: torch.Tensor,     # [batch_size]
    model_ids: torch.Tensor,  # [batch_size]
) -> None
```

Decomposes batch score variance into model-driven and prompt-driven portions.  Pass detached, concatenated scores from both pair arms together with their model IDs:

```python
diag.update_score_variance(
    torch.cat([scores_a.detach(), scores_b.detach()]),
    torch.cat([model_ids_a, model_ids_b]),
)
```

Skipped gracefully when fewer than 2 distinct models appear in the batch or total variance is near zero.

**Interpretation**:
- `score_model_variance_ratio` near 1.0 → the model has learned a prompt-independent ranking (scores differ mainly across models, not across prompts).
- `score_prompt_variance_ratio` near 1.0 → score differences are mostly driven by the prompt, not by which model is being scored.

**`to_dict` keys**: `score_total_variance`, `score_model_variance_ratio`, `score_prompt_variance_ratio`.

---

### `update_gradient_attribution`

```python
def update_gradient_attribution(
    self,
    inputs: dict[str, torch.Tensor],  # name -> [batch_size, dim], grad must be populated
) -> None
```

Accumulates `|grad| * |feat|` averaged over the batch for each named input tensor.  No extra forward or backward pass is needed — grads are populated by the normal training `loss.backward()`.

For each input tensor:
- **dim-1 tensors** (e.g. individual slices from `split_features_for_grad_attr`): attribution is a scalar, emitted as `grad_attr_{name}`.
- **dim>1 tensors** (e.g. whole embeddings): attribution is averaged across dimensions, emitted as a single `grad_attr_{name}` scalar representing mean per-dimension importance.

Entries whose `.grad` is `None` are silently skipped.

**Leaf tensors** (loaded from batch): set `requires_grad_(True)` before the forward pass.  
**Non-leaf tensors** (computed inside the model, e.g. transformer output): call `retain_grad()` on them before the forward pass.

**Call after `loss.backward()`**.

**`to_dict` keys**: `grad_attr_{name}` for each name that had grad populated.

---

### `to_dict`

```python
def to_dict(self) -> dict[str, float]
```

Returns a flat `dict[str, float]` of all accumulated metrics, averaged over their respective batch counts.  Suitable for direct use as `TrainingHistoryEntry.additional_metrics`.

---

## `TransformerEmbeddingModel` integration

`TransformerEmbeddingModel` uses `EpochDiagnosticsAccumulator` directly in its `_train_epoch`.  The previously private `_EpochMetricsAccumulator` nested class and the helper methods `_add_projection_norms_to_accumulator`, `_add_gradient_norms_to_accumulator`, and the module-level `_grad_norm_for_params` function have all been removed.

The existing metric keys (`prompt_emb_proj_norm`, `feat_proj_norm`, `model_proj_norm`, `interaction_norm`, `transformer_grad_norm`, `projection_grad_norm`, `scoring_head_grad_norm`) are preserved exactly.  New keys added to each epoch log:

| Key | Source |
|---|---|
| `prompt_emb_proj_variance` | `update_representation_stats` |
| `feat_proj_variance` | `update_representation_stats` |
| `model_proj_variance` | `update_representation_stats` |
| `interaction_variance` | `update_representation_stats` |
| `score_total_variance` | `update_score_variance` |
| `score_model_variance_ratio` | `update_score_variance` |
| `score_prompt_variance_ratio` | `update_score_variance` |
| `grad_attr_{feature_name}` × 45 | `update_gradient_attribution` (per prompt feature) |
| `grad_attr_prompt_embedding` | `update_gradient_attribution` (LLM embedding, mean over dims) |
| `grad_attr_model_embedding` | `update_gradient_attribution` (model embedding, mean over dims) |

---

## Extending to other models

Any model can use `EpochDiagnosticsAccumulator` by:
1. Instantiating it at epoch start.
2. Calling whichever `update_*` methods apply to its architecture.
3. Merging `diag.to_dict()` into `additional_metrics` at epoch end.

`DnEmbeddingModel` and `ResponsePredictiveModel` are natural candidates for adoption (tracked separately).
