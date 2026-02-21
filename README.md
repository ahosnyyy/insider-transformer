# Insider Threat Detection — InsiderTransformerAE

Full Reconstruction Autoencoder Transformer for insider threat detection on the CERT R4.2 dataset.

## Architecture

- **Model**: Encoder-only Transformer (`InsiderTransformerAE`) with full autoencoder reconstruction (no masking)
- **Training**: Full MSE loss on all positions (no masking), train ONLY on normal user behavior
- **Categorical handling**: Categorical features (user, PC, role, department, functional unit) are embedded and serve as context input
- **Loss**: MSE on behavioral features only (personality traits excluded as context-only)
- **Inference**: Single forward pass per sequence → reconstruction error = anomaly
- **Scoring**: `0.5 * mean_day_error + 0.5 * max_day_error` — captures both sustained and spike anomalies
- **Granularity**: Daily aggregation (60 active days per sequence, stride 5 days)

### Model Specifications

| Parameter | Value |
|---|---|
| d_model | 128 |
| n_heads | 8 |
| n_layers | 4 |
| d_ff | 512 |
| Sequence length | 60 days |
| Dropout | 0.2 |
| Reconstruction head | 2-layer (LayerNorm → Linear → GELU → Linear) |
| Granularity | Daily (60 active days) |
| Stride | 5 days |

## Pipeline

```
01_prepare_data.py → 02_feature_engineering.py → 03_train.py → 04_evaluate.py → 05_plot.py
```

---

## Step 1: Data Preparation

Convert raw CSVs to Parquet, ingest into DuckDB with derived tables.

```bash
python scripts/01_prepare_data.py
```

### What it creates

| Table | Description |
|---|---|
| `events` | Unified events from logon, device, email, file, http |
| `users` | Latest LDAP snapshot per user |
| `insiders` | Ground truth labels from answers/insiders.csv |
| `psychometric` | Big Five personality scores (O, C, E, A, N) |
| `user_changes` | Role/dept/FU changes across monthly LDAP snapshots |
| `terminated_users` | Users who disappear from LDAP between months |
| `functional_unit_encoding` | FU name → integer mapping |
| `activity_encoding` | Activity name → integer mapping |

### Outputs

```
data/parquet/                        ← Parquet files (logon, device, email, file, http, ldap, psychometric)
data/processed/insider.duckdb        ← DuckDB database with all tables
```

---

## Step 2: Feature Engineering

Extract features per day, create 60-day sequences, and optionally augment the test set.

```bash
python scripts/02_feature_engineering.py

# Skip augmentation
python scripts/02_feature_engineering.py --no-augment

# Feature extraction only (no sequences)
python scripts/02_feature_engineering.py --skip-sequences
```

Three stages:
1. **Daily aggregation**: Group events by `(user_id, date)` in DuckDB SQL → StandardScaler fitted on train users only
2. **Sequence creation**: 60-day sliding windows with stride 5 → train/val/test splits (70/15/15 by user)
3. **Augmentation**: Jittering, feature dropout, scaling on insider test sequences

### Outputs

```
data/processed/
  features_continuous.npy              ← Scaled continuous + cyclical + binary features (~45)
  features_categorical.npy             ← Integer-encoded categorical IDs (5)
  labels.npy                           ← Binary labels (0=normal, 1=insider)
  user_ids.npy                         ← User IDs per day
  dates.npy                            ← Date per day (datetime64)
  X_{train,val,test}_continuous.npy    ← Sequence splits (continuous)
  X_{train,val,test}_categorical.npy   ← Sequence splits (categorical)
  y_{train,val,test}.npy               ← Sequence labels
  user_ids_{train,val,test}.npy        ← User IDs per sequence
  dates_{train,val,test}.npy           ← Last date per sequence
  X_test_continuous_aug.npy            ← Augmented insider test sequences
  X_test_categorical_aug.npy
  y_test_aug.npy
  user_ids_test.npy
  preprocessing_artifacts.pkl          ← Encoders, scaler, column lists
  pipeline_config.json                 ← Config snapshot for data provenance
```

---

## Step 3: Training

Train InsiderTransformerAE with full reconstruction (no masking). Supports resume, mixed precision, and periodic test evaluation.

```bash
python scripts/03_train.py

# Override hyperparameters
python scripts/03_train.py --epochs 100 --batch-size 128 --lr 0.0003

# Resume from checkpoint
python scripts/03_train.py --resume

# Disable mixed precision
python scripts/03_train.py --no-amp

# Evaluate on test set every N epochs
python scripts/03_train.py --eval-interval 10

# Periodic checkpointing (save latest_model.pt every N epochs)
python scripts/03_train.py --checkpoint-interval 10  # default
python scripts/03_train.py --checkpoint-interval 5   # every 5 epochs
python scripts/03_train.py --checkpoint-interval 0   # disable

# Dry run: test full pipeline (1 epoch, few steps) before full training
python scripts/03_train.py --dry-run
```

### TensorBoard

Training logs are written to `outputs/logs/`. View with:

```bash
tensorboard --logdir outputs/logs
```

Tracked scalars:
- `Train/Loss_Step`, `Train/Loss_Epoch`, `Val/Loss_Epoch`, `Val/Loss_Step`
- `Train/LR_Step`, `Train/LR_Epoch`, `Train/GradNorm_Step`, `Train/GradNorm_Epoch`
- `Test/AUC`, `Test/AUPRC`, `Test/ScoreMean_Normal`, `Test/ScoreMean_Insider`, `Test/ScoreRatio`
- `Test/BestF1_Threshold`, `Test/UserDetectionRate`
- `Test/MeanLatency_Sequences`, `Test/MeanLatency_Hours`, `Test/MedianLatency_Hours`

### Outputs

```
outputs/
  best_model.pt              ← Best checkpoint (by val loss)
  final_model.pt             ← Last epoch checkpoint
  latest_model.pt            ← Periodic checkpoint (every N epochs, default: 10)
  training_history.json      ← Per-epoch train/val loss, LR, grad norm
  logs/                      ← TensorBoard event files
```

---

## Step 4: Evaluation

Compute anomaly scores, threshold-independent metrics, per-threshold breakdowns, per-user and per-scenario results.

```bash
python scripts/04_evaluate.py

# Specific threshold method
python scripts/04_evaluate.py --threshold best_f1

# Compare all threshold methods
python scripts/04_evaluate.py --threshold all

# Exclude specific scenarios
python scripts/04_evaluate.py --exclude-scenarios 2 3

# Load cached scores (skip re-scoring)
python scripts/04_evaluate.py --load-scores

# Evaluate on augmented test set
python scripts/04_evaluate.py --use-augmented
```

### Metrics

- **Threshold-independent**: AUPRC (primary), AUROC, score separation
- **Per-threshold**: F1, Precision, Recall, FPR, FNR, Specificity, Confusion Matrix
- **Threshold methods**: `best_f1` (oracle), `fixed_recall_95/99` (oracle), `percentile_95/99`, `mean_plus_3std`
- **Per-user**: Detection rate, latency (sequences and hours), max/mean score per insider user
- **Detection latency**: Time-based (hours/days) from first insider activity to first detection, lead time before ground-truth end date
- **Per-scenario**: Sequence recall, user recall by CERT scenario type
- **User-level**: Aggregate P/R/F1 treating each user as a single prediction

### SOC Alert Report

The evaluation automatically generates `outputs/soc_report.json` — a structured report designed for SOC analysts or SIEM ingestion. Each flagged insider user produces an alert with:

- **Severity** (HIGH / MEDIUM / LOW) based on detection rate and score magnitude
- **Detection summary**: flagged sequences, latency, anomaly scores
- **Risk indicators**: Human-readable labels for the most anomalous feature groups
- **Top anomalous windows**: The 5 highest-scoring sequences with timestamps
- **Ground-truth window**: Known insider activity period from CERT labels

### Outputs

```
outputs/
  test_anomaly_scores.npy      ← Per-sequence anomaly scores
  calibration_scores.npy       ← Validation set scores for threshold calibration
  evaluation_results.json      ← All metrics, breakdowns, thresholds, latency summary
  soc_report.json              ← Structured SOC alert report per insider user
```

---

## Step 5: Plotting

Generate 10 publication-quality plots from saved outputs. No model loading required. All plots use **seaborn** styling for professional, consistent visualizations with automatic legend positioning.

```bash
python scripts/05_plot.py

# Use augmented test labels
python scripts/05_plot.py --use-augmented

# Plot from dry run outputs
python scripts/05_plot.py --dry-run
```

### Plots

| # | Plot | File |
|---|------|------|
| 1 | Training / validation loss curve | `loss_curve.png` |
| 2 | Learning rate schedule | `lr_schedule.png` |
| 3 | Score distribution (normal vs insider) with KDE | `score_distribution.png` |
| 4 | Precision-Recall curve with operating points | `pr_curve.png` |
| 5 | ROC curve | `roc_curve.png` |
| 6 | Threshold method comparison (F1/P/R bars) | `threshold_comparison.png` |
| 7 | Per-scenario detection breakdown | `scenario_breakdown.png` |
| 8 | Anomaly score scatter | `score_scatter.png` |
| 9 | Confusion matrix (best_f1) | `confusion_matrix.png` |
| 10 | Detection timeline by insider user | `detection_timeline.png` |

**Visualization Features:**
- **Seaborn styling**: All plots use seaborn's `whitegrid` style with consistent color palettes
- **Enhanced distributions**: Score distributions include KDE overlays for better density visualization
- **Smart legends**: All legends use `loc='best'` for automatic optimal positioning
- **Professional aesthetics**: Consistent typography, improved color contrast, and publication-ready formatting

### Outputs

```
outputs/plots/                              ← All 10 PNG files
outputs/plots/detection_timeline_individual/ ← Individual per-user timeline plots
```

---

## Dependencies

All parameters in `config/config.yaml`. Key sections:

| Section | Controls |
|---|---|
| `model` | Transformer dims (d_model=128, d_ff=512), lookback (60 days), dropout (0.2) |
| `training` | Loss (MSE, full reconstruction), epochs (200), batch size (128), LR (0.0003), warmup, early stopping |
| `embeddings` | Embedding dimensions per categorical feature |
| `features` | Feature lists: continuous (~45), cyclical (2), binary (5), categorical (5), context_only (5) |
| `scoring` | Anomaly score method (mean_plus_max), weights |
| `augmentation` | Insider test set augmentation (jittering, feature dropout, scaling) |
| `evaluation` | Threshold methods, primary metric (AUPRC), detection level (user-level) |

---

## Data Requirements

Place CERT R4.2 dataset in `data/raw/`:

```
data/raw/
  logon.csv
  device.csv
  email.csv
  file.csv
  http.csv
  psychometric.csv
  LDAP/
    2010-01.csv ... 2011-05.csv
  answers/
    insiders.csv
```

**Note**: Daily aggregation requires only the above files. No session-based preprocessing needed.

---

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- DuckDB
- NumPy, pandas, scikit-learn, matplotlib, **seaborn**, tqdm, PyYAML
- TensorBoard (optional, for training visualization)
- pytest (for testing)
