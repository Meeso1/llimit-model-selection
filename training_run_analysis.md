# Training Run Analysis

For each spec in `training_specs/final/`, this documents all training runs, config changes between versions, and verifies the spec config is near-optimal.

> **Note on `best_accuracy` for DN length-prediction models:** This metric is `1 − avg_relative_error` and is used internally for best-checkpoint selection. It can be negative when the model is worse than predicting the mean. The tables below show `max_val_acc` (max validation accuracy across all epoch logs) as the primary metric for these models.
>
> **Note on `best_epoch`:** For DN models, `total_epochs` in the logs always equals `best_epoch − 1` because the state dict is captured *before* `_epochs_completed` is incremented. Reverting to the best state sets `_epochs_completed` back to `best_epoch − 1`. It does **not** indicate how many total epochs were trained — that is shown as `n_epochs` (number of epoch log entries).
>
> Runs with **no final metrics** (`best_epoch = —`) were genuinely interrupted before `finish_logger_if_needed` could be called.

---

## `dn-embedding-length-prediction`

**4 versions.** Spec version: `1777075103`

| Timestamp | Spec? | Saved? | Best Epoch | max_val_acc (epoch logs) | Other Metrics |
|-----------|:-----:|:------:|:----------:|---------------------|---------------|
| `1777075103` | ✓ |  | 12 | 0.4761 |  |
| `1777075984` |  |  | — | 0.4499 |  |
| `1777076119` |  |  | 8 | 0.4550 |  |
| `1777102890` |  |  | 91 | 0.4680 |  |

### Config changes (each vs previous)

**`1777075984`** (vs `1777075103` ← spec):
- `dropout`: 0.2000 → 0.2500
- `hidden_dims`: [1024, 1024, 1024, 1024, 512, 512, 512, 256, 128] → [1024, 1024, 1024, 1024, 512, 512, 512, 512, 256, 128]
- `optimizer_params.adamw_lr`: 1.00e-04 → 5.00e-05
- `optimizer_params.learning_rate`: 0.0100 → 0.0050
- `optimizer_params.lr_decay_gamma`: 0.9950 → 0.9900
- `max_val_acc`: 0.4761 → 0.4499 (-0.0262) ↓

**`1777076119`** (vs `1777075984`):
- `hidden_dims`: [1024, 1024, 1024, 1024, 512, 512, 512, 512, 256, 128] → [1024, 1024, 1024, 1024, 512, 512, 512, 256, 128]
- `max_val_acc`: 0.4499 → 0.4550 (+0.0050) ↑

**`1777102890`** (vs `1777076119`):
- `optimizer_params.adamw_lr`: 5.00e-05 → 2.00e-04
- `optimizer_params.learning_rate`: 0.0050 → 0.0200
- `max_val_acc`: 0.4550 → 0.4680 (+0.0130) ↑

### Best run

Metric: `max_val_acc` (higher is better)

Best: `1777075103` → 0.4761

✅ Spec version **is** the best run (0.4761).

> **Note:** `1777075984` has no final metrics (genuinely interrupted, 23 epochs). The other three ran all 200 epochs.

---

## `dn-embedding-length-prediction-chatbot-arena`

**8 versions.** Spec version: `1777074175`

| Timestamp | Spec? | Saved? | Best Epoch | max_val_acc (epoch logs) | Other Metrics |
|-----------|:-----:|:------:|:----------:|---------------------|---------------|
| `1777073650` |  |  | 40 | 0.4576 |  |
| `1777073927` |  |  | 15 | 0.4710 |  |
| `1777074175` | ✓ |  | 13 | 0.4801 |  |
| `1777074451` |  |  | — | 0.2736 |  |
| `1777074586` |  |  | — | 0.3062 |  |
| `1777074605` |  |  | — | 0.4570 |  |
| `1777074680` |  |  | — | 0.4584 |  |
| `1777074775` |  |  | 26 | 0.4574 |  |

### Config changes (each vs previous)

**`1777073927`** (vs `1777073650`):
- `dropout`: 0.2000 → 0.3000
- `embedding_model_name`: all-MiniLM-L6-v2 → BAAI/bge-base-en-v1.5
- `optimizer_params.learning_rate`: 0.0100 → 0.0070
- `max_val_acc`: 0.4576 → 0.4710 (+0.0134) ↑

**`1777074175` ← spec** (vs `1777073927`):
- `optimizer_params.adamw_lr`: 1.00e-04 → 2.00e-04
- `optimizer_params.learning_rate`: 0.0070 → 0.0050
- `optimizer_params.weight_decay`: 0.0100 → 0.0010
- `max_val_acc`: 0.4710 → 0.4801 (+0.0091) ↑

**`1777074451`** (vs `1777074175` ← spec):
- `optimizer_params.adamw_lr`: 2.00e-04 → 1.00e-04
- `optimizer_params.learning_rate`: 0.0050 → 1.00e-04
- `max_val_acc`: 0.4801 → 0.2736 (-0.2064) ↓

**`1777074586`** (vs `1777074451`):
- `optimizer_params.learning_rate`: 1.00e-04 → 0.0010
- `max_val_acc`: 0.2736 → 0.3062 (+0.0326) ↑

**`1777074605`** (vs `1777074586`):
- `optimizer_params.learning_rate`: 0.0010 → 0.0050
- `max_val_acc`: 0.3062 → 0.4570 (+0.1508) ↑

**`1777074680`** (vs `1777074605`):
- `optimizer_params.adamw_lr`: 1.00e-04 → 2.00e-04
- `max_val_acc`: 0.4570 → 0.4584 (+0.0014) ↑

**`1777074775`** (vs `1777074680`):
- `optimizer_params.adamw_lr`: 2.00e-04 → 1.00e-04
- `optimizer_params.learning_rate`: 0.0050 → 0.0030
- `optimizer_params.lr_decay_gamma`: 0.9950 → 0.9900
- `max_val_acc`: 0.4584 → 0.4574 (-9.89e-04) ↓

### Best run

Metric: `max_val_acc` (higher is better)

Best: `1777074175` → 0.4801

✅ Spec version **is** the best run (0.4801).

> **Note:** `1777074451` (113 epochs), `1777074586` (5 epochs), `1777074605` (51 epochs), and `1777074680` (63 epochs) have no final metrics (genuinely interrupted). `1777073650` ran 100 epochs. The remaining three ran all 200 epochs.

---

## `dn-embedding-scoring`

**5 versions.** Spec version: `1777152967`

| Timestamp | Spec? | Saved? | Best Epoch | best_accuracy | Other Metrics |
|-----------|:-----:|:------:|:----------:|---------------------|---------------|
| `1777141814` |  | ✓ | 4 | 0.6221 | `sensitivity/prompt_train` = 0.0155, `sensitivity/prompt_embedding_train` = 0.0026, `sensitivity/prompt_features_train` = 0.0080 |
| `1777151492` |  |  | — | — |  |
| `1777151592` |  | ✓ | 18 | 0.6250 | `sensitivity/prompt_train` = 0.0179, `sensitivity/prompt_embedding_train` = 0.0038, `sensitivity/prompt_features_train` = 0.0102 |
| `1777152967` | ✓ | ✓ | 51 | 0.6253 | `sensitivity/prompt_train` = 0.0250, `sensitivity/prompt_embedding_train` = 8.40e-04, `sensitivity/prompt_features_train` = 0.0088 |
| `1777154143` |  | ✓ | 80 | 0.6261 | `sensitivity/prompt_train` = 0.0346, `sensitivity/prompt_embedding_train` = 0.0070, `sensitivity/prompt_features_train` = 0.0204 |

### Config changes (each vs previous)

**`1777151492`** (vs `1777141814`):
- `dropout`: 0.2000 → 0.4000

**`1777151592`** (vs `1777151492`):
- `optimizer_params.adamw_lr`: 1.00e-04 → 5.00e-05
- `optimizer_params.learning_rate`: 0.0100 → 0.0010

**`1777152967` ← spec** (vs `1777151592`):
- `optimizer_params.adamw_lr`: 5.00e-05 → 2.00e-05
- `optimizer_params.learning_rate`: 0.0010 → 5.00e-04
- `best_accuracy`: 0.6250 → 0.6253 (+2.83e-04) ↑

**`1777154143`** (vs `1777152967`):
- `ranking_loss_type`: bradley_terry → margin_ranking
- `best_accuracy`: 0.6253 → 0.6261 (+7.99e-04) ↑

### Best run

Metric: `best_accuracy` (higher is better)

Best: `1777154143` → 0.6261

✅ Spec version `1777152967` → 0.6253 (Δ from best: -7.99e-04, 0.13%)

---

## `dn-embedding-scoring-chatbot-arena`

**7 versions.** Spec version: `1777150668`

| Timestamp | Spec? | Saved? | Best Epoch | best_accuracy | Other Metrics |
|-----------|:-----:|:------:|:----------:|---------------------|---------------|
| `1777148352` |  | ✓ | 3 | 0.7454 | `sensitivity/prompt_train` = 0.0082, `sensitivity/prompt_embedding_train` = -0.0109, `sensitivity/prompt_features_train` = 0.0056 |
| `1777149116` |  | ✓ | 6 | 0.7455 | `sensitivity/prompt_train` = 0.0278, `sensitivity/prompt_embedding_train` = -0.0025, `sensitivity/prompt_features_train` = 0.0113 |
| `1777149634` |  | ✓ | 35 | 0.7513 | `sensitivity/prompt_train` = 0.0384, `sensitivity/prompt_embedding_train` = 0.0054, `sensitivity/prompt_features_train` = 0.0193 |
| `1777149902` |  | ✓ | 17 | 0.7569 | `sensitivity/prompt_train` = 0.0178, `sensitivity/prompt_embedding_train` = -0.0053, `sensitivity/prompt_features_train` = 0.0175 |
| `1777150344` |  | ✓ | 17 | 0.7505 | `sensitivity/prompt_train` = 0.0191, `sensitivity/prompt_embedding_train` = 0.0012, `sensitivity/prompt_features_train` = 0.0172 |
| `1777150668` | ✓ | ✓ | 10 | 0.7586 | `sensitivity/prompt_train` = 0.0169, `sensitivity/prompt_embedding_train` = -0.0041, `sensitivity/prompt_features_train` = 0.0143 |
| `1777150950` |  | ✓ | 33 | 0.7564 | `sensitivity/prompt_train` = 0.0256, `sensitivity/prompt_embedding_train` = -3.22e-04, `sensitivity/prompt_features_train` = 0.0125 |

### Config changes (each vs previous)

**`1777149116`** (vs `1777148352`):
- `dropout`: 0.2000 → 0.3000
- `best_accuracy`: 0.7454 → 0.7455 (+5.33e-05) ↑

**`1777149634`** (vs `1777149116`):
- `dropout`: 0.3000 → 0.4000
- `optimizer_params.learning_rate`: 0.0100 → 0.0010
- `best_accuracy`: 0.7455 → 0.7513 (+0.0058) ↑

**`1777149902`** (vs `1777149634`):
- `optimizer_params.adamw_lr`: 1.00e-04 → 5.00e-05
- `best_accuracy`: 0.7513 → 0.7569 (+0.0056) ↑

**`1777150344`** (vs `1777149902`):
- `optimizer_params.momentum`: 0.9500 → 0.9900
- `best_accuracy`: 0.7569 → 0.7505 (-0.0064) ↓

**`1777150668` ← spec** (vs `1777150344`):
- `optimizer_params.momentum`: 0.9900 → 0.9500
- `best_accuracy`: 0.7505 → 0.7586 (+0.0081) ↑

**`1777150950`** (vs `1777150668`):
- `dropout`: 0.4000 → 0.4500
- `input_proj_dim`: 64 → 128
- `best_accuracy`: 0.7586 → 0.7564 (-0.0022) ↓

### Best run

Metric: `best_accuracy` (higher is better)

Best: `1777150668` → 0.7586

✅ Spec version **is** the best run (0.7586).

---

## `gb-length-prediction`

Single version. Timestamp: `1777140386`, saved model: `gb-length-prediction-1777140515`, best epoch: `500`

Metrics: `best_accuracy` = 0.4158

---

## `gb-length-prediction-chatbot-arena`

Single version. Timestamp: `1777140662`, saved model: `gb-length-prediction-chatbot-arena-1777140716`, best epoch: `496`

Metrics: `best_accuracy` = 0.4430

---

## `gradient-boosting-scoring`

**4 versions.** Spec version: `1777139846`

| Timestamp | Spec? | Saved? | Best Epoch | best_accuracy | Other Metrics |
|-----------|:-----:|:------:|:----------:|---------------------|---------------|
| `1776980140` |  |  | 79 | 0.6228 | `sensitivity/prompt_train` = 0.0000, `sensitivity/prompt_embedding_train` = 0.0295, `sensitivity/prompt_features_train` = 0.0044 |
| `1777068039` |  |  | 64 | 0.6237 | `sensitivity/prompt_train` = 0.0000, `sensitivity/prompt_embedding_train` = 0.0232, `sensitivity/prompt_features_train` = 0.0045 |
| `1777138712` |  | ✓ | 11 | 0.6226 | `sensitivity/prompt_train` = 0.0000, `sensitivity/prompt_embedding_train` = 0.0044, `sensitivity/prompt_features_train` = 0.0013 |
| `1777139846` | ✓ | ✓ | 86 | 0.6228 | `sensitivity/prompt_train` = 0.0000, `sensitivity/prompt_embedding_train` = 0.0316, `sensitivity/prompt_features_train` = 0.0045 |

### Config changes (each vs previous)

**`1777068039`** (vs `1776980140`):
- `embedding_model_epochs`: 250 → 350
- `input_features`: ['prompt_features', 'model_embedding', 'prompt_embedding'] → ['prompt_embedding', 'model_embedding', 'prompt_features']
- `best_accuracy`: 0.6228 → 0.6237 (+9.30e-04) ↑

**`1777138712`** (vs `1777068039`):
- `input_features`: ['prompt_embedding', 'model_embedding', 'prompt_features'] → ['model_embedding', 'prompt_embedding', 'prompt_features']
- `best_accuracy`: 0.6237 → 0.6226 (-0.0011) ↓

**`1777139846` ← spec** (vs `1777138712`):
- `embedding_spec.models_per_batch`: 16 → 50
- `input_features`: ['model_embedding', 'prompt_embedding', 'prompt_features'] → ['model_embedding', 'prompt_features', 'prompt_embedding']
- `best_accuracy`: 0.6226 → 0.6228 (+1.55e-04) ↑

### Best run

Metric: `best_accuracy` (higher is better)

Best: `1777068039` → 0.6237

✅ Spec version `1777139846` → 0.6228 (Δ from best: -9.30e-04, 0.15%)

---

## `gradient-boosting-scoring-chatbot-arena`

**10 versions.** Spec version: `1777139252`

| Timestamp | Spec? | Saved? | Best Epoch | best_accuracy | Other Metrics |
|-----------|:-----:|:------:|:----------:|---------------------|---------------|
| `1777062917` |  |  | 65 | 0.7532 | `sensitivity/prompt_train` = 0.0000, `sensitivity/prompt_embedding_train` = 0.0413, `sensitivity/prompt_features_train` = 0.0035 |
| `1777068400` |  |  | 66 | 0.7485 | `sensitivity/prompt_train` = 0.0000, `sensitivity/prompt_embedding_train` = 0.0450, `sensitivity/prompt_features_train` = 0.0026 |
| `1777068826` |  |  | 98 | 0.7461 | `sensitivity/prompt_train` = 0.0000, `sensitivity/prompt_embedding_train` = 0.0646, `sensitivity/prompt_features_train` = 0.0064 |
| `1777069021` |  |  | 87 | 0.7487 | `sensitivity/prompt_train` = 0.0000, `sensitivity/prompt_embedding_train` = 0.0587, `sensitivity/prompt_features_train` = 0.0037 |
| `1777069263` |  |  | 92 | 0.7487 | `sensitivity/prompt_train` = 0.0000, `sensitivity/prompt_embedding_train` = 0.0619, `sensitivity/prompt_features_train` = 0.0056 |
| `1777070318` |  |  | 40 | 0.7472 | `sensitivity/prompt_train` = 0.0000, `sensitivity/prompt_embedding_train` = 0.0213, `sensitivity/prompt_features_train` = 0.0027 |
| `1777070478` |  |  | 49 | 0.7468 | `sensitivity/prompt_train` = 0.0000, `sensitivity/prompt_embedding_train` = 0.0316, `sensitivity/prompt_features_train` = 0.0046 |
| `1777070831` |  |  | 52 | 0.7500 | `sensitivity/prompt_train` = 0.0000, `sensitivity/prompt_embedding_train` = 0.0344, `sensitivity/prompt_features_train` = 0.0049 |
| `1777139252` | ✓ | ✓ | 42 | 0.7487 | `sensitivity/prompt_train` = 0.0000, `sensitivity/prompt_embedding_train` = 0.0283, `sensitivity/prompt_features_train` = 0.0032 |
| `1777139617` |  | ✓ | 20 | 0.7457 | `sensitivity/prompt_train` = 0.0000, `sensitivity/prompt_embedding_train` = 0.0140, `sensitivity/prompt_features_train` = 0.0015 |

### Config changes (each vs previous)

**`1777068400`** (vs `1777062917`):
- `embedding_model_epochs`: 100 → 400
- `embedding_spec.encoder_model_name`: BAAI/bge-base-en-v1.5 → all-MiniLM-L6-v2
- `embedding_spec.optimizer.learning_rate`: 1.00e-04 → 2.00e-04
- `embedding_spec.optimizer.lr_decay_gamma`: 0.9950 → None
- `embedding_spec.pair_mlp_layers`: 12 → 8
- `input_features`: ['prompt_features', 'prompt_embedding', 'model_embedding'] → ['model_embedding', 'prompt_embedding', 'prompt_features']
- `best_accuracy`: 0.7532 → 0.7485 (-0.0047) ↓

**`1777068826`** (vs `1777068400`):
- `embedding_spec.dropout`: 0.1000 → 0.1500
- `embedding_spec.models_per_batch`: 16 → 20
- `embedding_spec.optimizer.lr_decay_gamma`: None → 0.9950
- `input_features`: ['model_embedding', 'prompt_embedding', 'prompt_features'] → ['prompt_features', 'model_embedding', 'prompt_embedding']
- `best_accuracy`: 0.7485 → 0.7461 (-0.0024) ↓

**`1777069021`** (vs `1777068826`):
- `embedding_spec.optimizer.learning_rate`: 2.00e-04 → 1.00e-04
- `best_accuracy`: 0.7461 → 0.7487 (+0.0026) ↑

**`1777069263`** (vs `1777069021`):
- `embedding_spec.encoder_model_name`: all-MiniLM-L6-v2 → BAAI/bge-base-en-v1.5
- `input_features`: ['prompt_features', 'model_embedding', 'prompt_embedding'] → ['model_embedding', 'prompt_embedding', 'prompt_features']
- `best_accuracy`: 0.7487 → 0.7487 (+0.0000) =

**`1777070318`** (vs `1777069263`):
- `embedding_spec.pair_mlp_layers`: 8 → 6
- `input_features`: ['model_embedding', 'prompt_embedding', 'prompt_features'] → ['prompt_features', 'model_embedding', 'prompt_embedding']
- `best_accuracy`: 0.7487 → 0.7472 (-0.0015) ↓

**`1777070478`** (vs `1777070318`):
- `embedding_spec.dropout`: 0.1500 → 0.2500
- `input_features`: ['prompt_features', 'model_embedding', 'prompt_embedding'] → ['model_embedding', 'prompt_embedding', 'prompt_features']
- `best_accuracy`: 0.7472 → 0.7468 (-4.30e-04) ↓

**`1777070831`** (vs `1777070478`):
- `embedding_spec.dropout`: 0.2500 → 0.2000
- `input_features`: ['model_embedding', 'prompt_embedding', 'prompt_features'] → ['model_embedding', 'prompt_features', 'prompt_embedding']
- `best_accuracy`: 0.7468 → 0.7500 (+0.0032) ↑

**`1777139252` ← spec** (vs `1777070831`):
- `input_features`: ['model_embedding', 'prompt_features', 'prompt_embedding'] → ['model_embedding', 'prompt_embedding', 'prompt_features']
- `best_accuracy`: 0.7500 → 0.7487 (-0.0013) ↓

**`1777139617`** (vs `1777139252`):
- `embedding_spec.models_per_batch`: 20 → 16
- `best_accuracy`: 0.7487 → 0.7457 (-0.0030) ↓

### Best run

Metric: `best_accuracy` (higher is better)

Best: `1777062917` → 0.7532

⚠️ Spec version `1777139252` → 0.7487 (Δ from best: -0.0045, 0.60%)

---

## `gradient-boosting-scoring-only-llm-embedding`

Single version. Timestamp: `1777203179`, saved model: `gradient-boosting-scoring-only-llm-embedding-1777203184`, best epoch: `3`

Metrics: `best_accuracy` = 0.6218, `sensitivity/prompt_train` = 0.0000, `sensitivity/prompt_val` = 0.0000

---

## `gradient-boosting-scoring-only-llm-embedding-chatbot-arena`

Single version. Timestamp: `1777203216`, saved model: `gradient-boosting-scoring-only-llm-embedding-chatbot-arena-1777203218`, best epoch: `33`

Metrics: `best_accuracy` = 0.7455, `sensitivity/prompt_train` = 0.0000, `sensitivity/prompt_val` = 0.0000

---

## `response-predictive-scoring`

Single version. Timestamp: `1777159739`, saved model: `response-predictive-scoring-1777163436`, best epoch: `28`

Metrics: `best_accuracy` = 0.6209, `sensitivity/prompt_train` = 0.0227, `sensitivity/prompt_embedding_train` = 0.0058, `sensitivity/prompt_features_train` = 0.0093

---

## `response-predictive-scoring-chatbot-arena`

**7 versions.** Spec version: `1777158730`

| Timestamp | Spec? | Saved? | Best Epoch | best_accuracy | Other Metrics |
|-----------|:-----:|:------:|:----------:|---------------------|---------------|
| `1777155647` |  | ✓ | 104 | 0.7466 | `sensitivity/prompt_train` = 0.1804, `sensitivity/prompt_embedding_train` = 0.0778, `sensitivity/prompt_features_train` = 0.0658 |
| `1777156223` |  | ✓ | 71 | 0.7377 | `sensitivity/prompt_train` = 0.0090, `sensitivity/prompt_embedding_train` = 0.0012, `sensitivity/prompt_features_train` = 0.0076 |
| `1777156640` |  | ✓ | 34 | 0.7468 | `sensitivity/prompt_train` = 0.0466, `sensitivity/prompt_embedding_train` = 0.0056, `sensitivity/prompt_features_train` = 0.0125 |
| `1777157419` |  | ✓ | 59 | 0.7459 | `sensitivity/prompt_train` = 0.0549, `sensitivity/prompt_embedding_train` = 0.0193, `sensitivity/prompt_features_train` = 0.0148 |
| `1777157764` |  | ✓ | 74 | 0.7440 | `sensitivity/prompt_train` = 0.0819, `sensitivity/prompt_embedding_train` = 0.0258, `sensitivity/prompt_features_train` = 0.0235 |
| `1777158093` |  | ✓ | 60 | 0.7468 | `sensitivity/prompt_train` = 0.0556, `sensitivity/prompt_embedding_train` = 0.0125, `sensitivity/prompt_features_train` = 0.0198 |
| `1777158730` | ✓ | ✓ | 91 | 0.7498 | `sensitivity/prompt_train` = 0.0993, `sensitivity/prompt_embedding_train` = 0.0402, `sensitivity/prompt_features_train` = 0.0359 |

### Config changes (each vs previous)

**`1777156223`** (vs `1777155647`):
- `optimizer_params.adamw_lr`: 1.00e-04 → 5.00e-05
- `optimizer_params.learning_rate`: 0.0050 → 0.0010
- `best_accuracy`: 0.7466 → 0.7377 (-0.0089) ↓

**`1777156640`** (vs `1777156223`):
- `optimizer_params.adamw_lr`: 5.00e-05 → 1.00e-04
- `optimizer_params.learning_rate`: 0.0010 → 0.0050
- `optimizer_params.momentum`: 0.9000 → 0.9500
- `best_accuracy`: 0.7377 → 0.7468 (+0.0091) ↑

**`1777157419`** (vs `1777156640`):
- `dropout`: 0.3000 → 0.4000
- `encoder_hidden_dims`: [1024, 512, 256, 128] → [1024, 1024, 512, 256, 128]
- `optimizer_params.momentum`: 0.9500 → 0.9000
- `best_accuracy`: 0.7468 → 0.7459 (-8.53e-04) ↓

**`1777157764`** (vs `1777157419`):
- `predictor_hidden_dims`: [512, 512, 512, 512, 512, 512] → [512, 512, 512, 512, 512, 512, 512, 512, 512]
- `scorer_hidden_dims`: [512, 512, 512, 512, 512, 512] → [512, 512, 512, 512, 512, 512, 512, 512, 512]
- `best_accuracy`: 0.7459 → 0.7440 (-0.0020) ↓

**`1777158093`** (vs `1777157764`):
- `prediction_loss_weight`: 0.1000 → 0.2000
- `best_accuracy`: 0.7440 → 0.7468 (+0.0028) ↑

**`1777158730` ← spec** (vs `1777158093`):
- `embedding_model_name`: all-MiniLM-L6-v2 → BAAI/bge-base-en-v1.5
- `best_accuracy`: 0.7468 → 0.7498 (+0.0029) ↑

### Best run

Metric: `best_accuracy` (higher is better)

Best: `1777158730` → 0.7498

✅ Spec version **is** the best run (0.7498).

---

## `transformer-embedding-scoring`

Single version. Timestamp: `1777163514`, saved model: `transformer-embedding-scoring-1777192317`, best epoch: `14`

Metrics: `best_accuracy` = 0.6179, `sensitivity/prompt_train` = 0.0011, `sensitivity/prompt_embedding_train` = 0.0000, `sensitivity/prompt_features_train` = -9.95e-04

---

## `transformer-embedding-scoring-chatbot-arena`

Single version. Timestamp: `1777193088`, saved model: `transformer-embedding-scoring-chatbot-arena-1777198626`, best epoch: `9`

Metrics: `best_accuracy` = 0.7447, `sensitivity/prompt_train` = 0.0058, `sensitivity/prompt_embedding_train` = 0.0000, `sensitivity/prompt_features_train` = 0.0021

---

