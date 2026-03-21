---
name: modifying-model
description: Guides making changes to existing scoring/embedding model classes in this codebase — adding or removing hyperparameters, loss terms, training-loop logic, and metrics. Use when modifying a model's __init__, training loop, loss functions, state dict, or logged metrics. Also covers when backward compatibility is required vs when it is not.
---

# Modifying an Existing Model

## Backward compatibility rules

**Preserve backward compat** only when the change is a new _hyperparameter_ that previously had an implicitly fixed value (e.g. a new loss weight that was effectively `0.0`, or a new flag that was effectively `False`). In `load_state_dict`, load it with a safe default:

```python
new_param=state_dict.get("new_param", 0.0),
```

**Do NOT preserve backward compat** when the change affects the model architecture — new/removed layers, different forward-pass structure, changed loss paths. Old checkpoints are expected to fail to load; that is fine.

---

## Touch-points checklist

Work through these in order:

- [ ] `__init__` — add/remove parameters; store as `self.param`
- [ ] `get_config_for_logging` — reflect added/removed parameters for W&B (one-to-one with `self.*`)
- [ ] Training loop (`_train_epoch`) — update loss computation, accumulators, `additional_metrics`
- [ ] Validation loop (`_perform_validation`) — mirror changes from `_train_epoch`; update returned dict and the `additional_metrics.update({...})` block in `_train_epoch` for `val_*` variants
- [ ] `get_state_dict` — add new params, remove old ones
- [ ] `load_state_dict` — add new params via `.get(key, default)` where compat is needed; simply remove old lines otherwise
- [ ] CLI & spec — see [cli-and-spec.md](cli-and-spec.md)
- [ ] Plotting utils — see [plotting.md](plotting.md)
- [ ] Docs — update `docs/models/<model>.md` and the summary line in `docs/models.md`

---

## Metrics naming convention

- Training metrics: plain name, e.g. `scoring_loss`, `score_consistency_loss`
- Validation metrics: `val_` prefix, e.g. `val_score_consistency_loss`
- Both go into `additional_metrics` in `_train_epoch`
- The validation loop returns a flat dict; `_train_epoch` prefixes with `val_` when merging
