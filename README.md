# Insider Threat Detection — InsiderTransformerAE

Hybrid Session→Daily Transformer Autoencoder for insider threat detection on the CERT R4.2 dataset.

## Architecture

- **Model**: Encoder-only Transformer (`InsiderTransformerAE`) with full reconstruction loss (no masking)
- **Features**: Session-aware daily features (~52 continuous + 5 categorical)
  - Daily activity counts (18) + Session-derived statistics (18)
  - Interaction features (4) + Cyclical/binary features (7) + Personality traits (5)
- **Training**: Train ONLY on normal user behavior; anomalous behavior produces high reconstruction error
- **Categorical handling**: Categorical features (user, PC, role, department, functional_unit) are embedded and serve as context input
- **Loss**: MSE on behavioral features only (personality traits excluded as context-only)
- **Inference**: Single forward pass per sequence → reconstruction error = anomaly score
- **Scoring**: `0.5 * mean_day_error + 0.5 * max_day_error` — captures both sustained and spike anomalies
- **Granularity**: Daily aggregation (60 active days per sequence, stride 5 days)

### Model Specifications

| Parameter | Value |
|---|---|
| d_model | 192 |
| n_heads | 6 |
| n_layers | 4 |
| d_ff | 512 |
| Sequence length | 60 days |
| Dropout | 0.2 |
| Reconstruction head | 2-layer (LayerNorm → Linear → GELU → Linear) |
| Granularity | Daily (60 active days) |
| Stride | 5 days |
| Features | ~52 continuous + 5 categorical (session-aware) |

## Project Structure

All scripts are **thin CLI wrappers** that parse arguments and delegate to core classes in `src/`:

| Script | Core Module | Class |
|--------|-------------|-------|
| `scripts/03_train.py` | `src/training/trainer.py` | `Trainer` |
| `scripts/04_evaluate.py` | `src/evaluation/evaluator.py` | `Evaluator` |
| `scripts/05_plot.py` | `src/visualization/plotter.py` | `Plotter` |
| `scripts/06_inference.py` | `src/inference/runner.py` | `InferenceRunner` |

`scripts/01_prepare_data.py` and `scripts/02_feature_engineering.py` are also wrappers, calling functions from `src/data/`.

```
insider-transformer/
├── config/
│   └── config.yaml                    # Central configuration
├── scripts/                           # Thin CLI wrappers (~40-80 lines each)
│   ├── 01_prepare_data.py
│   ├── 02_feature_engineering.py
│   ├── 03_train.py
│   ├── 04_evaluate.py
│   ├── 05_plot.py
│   └── 06_inference.py
├── src/                               # Core logic
│   ├── data/                          # Data loading, feature engineering, sequences, augmentation
│   ├── models/                        # InsiderTransformerAE architecture
│   ├── training/
│   │   └── trainer.py                 # Trainer class (training loop, checkpointing, TensorBoard)
│   ├── evaluation/
│   │   ├── evaluator.py               # Evaluator class (scoring, thresholds, metrics, SOC report)
│   │   ├── scoring.py                 # Anomaly scoring functions
│   │   └── helpers.py                 # Evaluation utilities
│   ├── visualization/
│   │   └── plotter.py                 # Plotter class (11 publication-quality plots)
│   ├── inference/
│   │   └── runner.py                  # InferenceRunner class (feature eng → scoring → reports)
│   └── utils/                         # Config loading, seeding, device selection
├── data/                              # Raw + processed data
├── outputs/                           # Checkpoints, scores, plots, reports
└── docs/                              # Architecture and feature documentation
```

## Pipeline

```
01_prepare_data.py → 02_feature_engineering.py → 03_train.py → 04_evaluate.py → 05_plot.py
                                                                                    ↓
                                                                              06_inference.py
```

**Complete workflow:**
1. **Data Prep**: CSV → Parquet → DuckDB with session detection
2. **Feature Eng**: Session-aware daily features → 60-day sequences
3. **Training**: Full reconstruction Transformer on normal users
4. **Evaluation**: Multi-level metrics (user, sequence, session)
5. **Visualization**: 11 plots including session-level analysis
6. **Inference**: Real-time scoring with date range filtering

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

Four stages:
1. **Daily aggregation**: Group events by `(user_id, date)` in DuckDB SQL + session detection (Logon/Logoff pairing) + per-session feature extraction + daily session stats
2. **Feature computation**: Log1p transforms, z-scores, rolling stats, cyclical encoding, interaction features → StandardScaler fitted on train users only
3. **Sequence creation**: 60-day sliding windows with stride 5 → train/val/test splits (70/15/15 by user)
4. **Augmentation**: Jittering, feature dropout, scaling on insider test sequences

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

Train InsiderTransformerAE with reconstruction loss. Supports resume, mixed precision (opt-in), and periodic test evaluation.

```bash
python scripts/03_train.py                                    # Default (AMP disabled)

# Override hyperparameters
python scripts/03_train.py --epochs 100 --batch-size 128 --lr 0.0003

# Resume from checkpoint
python scripts/03_train.py --resume

# Enable mixed precision for faster training
python scripts/03_train.py --amp

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

Comprehensive evaluation with multi-level metrics (user, sequence, session) and SOC report generation.

```bash
python scripts/04_evaluate.py                                    # Default (AMP disabled)

# Use augmented test set
python scripts/04_evaluate.py --use-augmented

# Enable mixed precision for faster evaluation
python scripts/04_evaluate.py --amp

# Exclude specific scenarios
python scripts/04_evaluate.py --exclude-scenarios 3 4

# Override batch size
python scripts/04_evaluate.py --batch-size 512
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

Generate publication-quality plots from saved outputs. No model loading required. All plots use **seaborn** styling for professional, consistent visualizations with automatic legend positioning.

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
| 11 | Session-level confusion matrix | `confusion_matrix_session.png` |

**Visualization Features:**
- **Seaborn styling**: All plots use seaborn's `whitegrid` style with consistent color palettes
- **Enhanced distributions**: Score distributions include KDE overlays for better density visualization
- **Smart legends**: All legends use `loc='best'` for automatic optimal positioning
- **Professional aesthetics**: Consistent typography, improved color contrast, and publication-ready formatting

### Outputs

```
outputs/plots/                              ← All 11 PNG files
outputs/plots/detection_timeline_individual/ ← Individual per-user timeline plots
```

---

## Step 6: Inference

Score users for insider threat risk using the trained model. Runs the full pipeline: feature engineering → sequencing → scoring → report.

```bash
# Score all users (AMP disabled by default)
python scripts/06_inference.py

# Enable mixed precision for faster inference
python scripts/06_inference.py --amp

# Score a specific user
python scripts/06_inference.py --user-id ACME/user123

# Score a date range (only processes events in that window)
python scripts/06_inference.py --start-date 2011-03-01 --end-date 2011-04-01

# Combine user + date range
python scripts/06_inference.py --user-id ACME/user123 --start-date 2011-03-01 --end-date 2011-04-01

# Custom threshold
python scripts/06_inference.py --threshold 0.85

# Use a different threshold method
python scripts/06_inference.py --threshold-method percentile_99

# Show top 20 riskiest users
python scripts/06_inference.py --top-k 20
```

The script reads from the DuckDB database and uses saved preprocessing artifacts (scaler means/stds, label mappings) from training — no re-fitting. Default threshold is `best_f1` from evaluation results. Date ranges automatically include a buffer period for rolling statistics and sequence history.

### Outputs

```
outputs/
  inference_report.json    ← Per-user risk report (severity, scores, risk indicators, top windows)
  inference_scores.npz     ← Raw scores + user IDs + dates for further analysis
```

### Report Structure

Each user entry includes:
- **Severity** (HIGH / MEDIUM / LOW / NORMAL)
- **Anomaly scores** (max, mean, detection rate)
- **Top anomalous windows** with dates
- **Risk indicators** — human-readable labels for the most anomalous feature groups

---

## Configuration

All parameters in `config/config.yaml`. Key sections:

| Section | Controls |
|---|---|
| `model` | Transformer dims (d_model=128, d_ff=512), lookback (60 days), dropout (0.2) |
| `training` | Loss (MSE), epochs (200), batch size (128), LR (0.0003), warmup, early stopping |
| `embeddings` | Embedding dimensions per categorical feature |
| `features` | Feature lists: continuous, session, interaction, cyclical, binary, categorical, context_only |
| `scoring` | Anomaly score method (mean_plus_max), weights |
| `masking` | Masked reconstruction settings (enabled, mask_ratio, strategy) |
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

**Note**: The pipeline automatically detects sessions from Logon/Logoff events and computes session-level features.

---

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- DuckDB
- NumPy, scikit-learn, matplotlib, seaborn, tqdm, PyYAML
- TensorBoard (optional, for training visualization)
