# CLI & Spec Changes

When modifying a model's parameters, update these three files:

## 1. `src/scripts/model_types.py` — `<Model>Specification` class

Add/remove fields in the `Pydantic` spec class for the relevant model. New fields should always have a default matching the model's `__init__` default so that existing JSON spec files continue to work:

```python
class ResponsePredictiveSpecification(ModelSpecBase):
    score_consistency_loss_weight: float = 0.1   # add
    # real_repr_ratio: float = 0.8               # remove
```

Pydantic ignores unknown fields in JSON by default, so removing a field from the spec does not break existing JSON files that still contain it — those keys are silently discarded.

## 2. `src/scripts/train.py` — `_create_starting_<model>` function

Pass new spec fields to the model constructor, remove old ones:

```python
return ResponsePredictiveModel(
    score_consistency_loss_weight=model_spec.score_consistency_loss_weight,  # add
    # real_repr_ratio=model_spec.real_repr_ratio,  # remove
    ...
)
```

## 3. Training spec JSON files (`training_specs/`)

Update JSON files that configure this model type. Remove old keys; add new ones with sensible values:

```json
{
  "score_consistency_loss_weight": 0.1,
  "repr_dist_kl_loss_weight": 0.01
}
```

To find all relevant spec files:
```bash
grep -rl '"model_type": "response_predictive"' training_specs/
```

---

## What does NOT need to change

- `src/scripts/training_spec.py` — the outer `TrainingSpecification` wrapper is model-agnostic and never needs editing for model parameter changes.
- Other model spec classes — only the spec for the model being changed needs updating.
